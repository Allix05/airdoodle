"""Loads user-submitted correction files (exported from the browser app's
"Download corrections" button, saved into training/corrections/*.json) and
turns them into (X, y) arrays in the same 28x28 uint8 format as the rest
of the unified dataset, so they can be folded into the next training run.

Each correction is a hand-verified "the model got this wrong, here's the
real answer" example, which is more valuable per-sample than a random
Quick Draw doodle, so they're upsampled (repeated) before being merged in
-- otherwise a handful of corrections would have no measurable effect
against 300K+ other images.
"""
import base64
import glob
import json
import os

import numpy as np

from classes import CLASSES

CORRECTIONS_DIR = os.path.join(os.path.dirname(__file__), "corrections")
UPSAMPLE = 8  # repeat each hand-verified correction this many times


def unpack_bits(b64_str: str, length: int) -> np.ndarray:
    """Inverse of corrections.js's packBits: little-endian bit order within
    each byte, matching `bytes[i >> 3] |= 1 << (i & 7)`.
    """
    raw = np.frombuffer(base64.b64decode(b64_str), dtype=np.uint8)
    return np.unpackbits(raw, bitorder="little")[:length]


def load_corrections():
    """Returns (X, y): X is (N, 28, 28) uint8, y is (N,) int64 label indices,
    already upsampled. Empty arrays if no correction files are present.
    """
    label_to_idx = {name: i for i, name in enumerate(CLASSES)}
    images, labels = [], []

    paths = sorted(glob.glob(os.path.join(CORRECTIONS_DIR, "*.json")))
    for path in paths:
        with open(path) as f:
            data = json.load(f)
        entries = data.get("corrections", [])
        kept = 0
        for entry in entries:
            label = entry["correctLabel"]
            if label not in label_to_idx:
                print(f"  skipping unknown label '{label}' in {os.path.basename(path)}")
                continue
            bits = unpack_bits(entry["inputPacked"], entry["inputLength"])
            images.append((bits * 255).astype(np.uint8).reshape(28, 28))
            labels.append(label_to_idx[label])
            kept += 1
        print(f"  {os.path.basename(path)}: {kept}/{len(entries)} corrections")

    if not images:
        return np.zeros((0, 28, 28), dtype=np.uint8), np.zeros((0,), dtype=np.int64)

    X = np.repeat(np.stack(images), UPSAMPLE, axis=0)
    y = np.repeat(np.array(labels, dtype=np.int64), UPSAMPLE, axis=0)
    return X, y


if __name__ == "__main__":
    X, y = load_corrections()
    n_classes = len(set(y.tolist())) if len(y) else 0
    print(f"Total: {len(X)} images ({UPSAMPLE}x upsampled) across {n_classes} classes")
