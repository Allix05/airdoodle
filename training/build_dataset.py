"""Assemble the unified training set: MNIST digits + EMNIST letters +
Quick Draw doodle categories, all as 28x28 uint8 grayscale bitmaps with
strokes as bright pixels on a dark background (the convention all three
sources already share, and the same convention the browser-side canvas
preprocessing produces).
"""
import numpy as np
import torchvision

from classes import CLASSES, DIGIT_OFFSET, DOODLE_OFFSET, DOODLES, LETTER_OFFSET
from quickdraw_fetch import load_quickdraw_subset

N_PER_DOODLE_CLASS = 6000
N_PER_LETTER_CLASS = 4000  # EMNIST letters has plenty; cap for balance/speed


def load_mnist():
    ds = torchvision.datasets.MNIST(root="./data", train=True, download=True)
    images = ds.data.numpy().astype(np.uint8)  # (60000, 28, 28), already correct orientation
    labels = ds.targets.numpy().astype(np.int64) + DIGIT_OFFSET
    return images, labels


def load_emnist_letters():
    ds = torchvision.datasets.EMNIST(root="./data", split="letters", train=True, download=True)
    images = ds.data.numpy().astype(np.uint8)  # (N, 28, 28)
    # EMNIST stores images transposed relative to MNIST's convention.
    images = np.transpose(images, (0, 2, 1))
    labels = ds.targets.numpy().astype(np.int64)  # 1..26
    labels = (labels - 1) + LETTER_OFFSET  # -> 0..25, then shift into our label space

    # Balance/cap per letter for speed and class balance.
    out_images, out_labels = [], []
    for letter_idx in range(26):
        mask = labels == (letter_idx + LETTER_OFFSET)
        idx = np.where(mask)[0][:N_PER_LETTER_CLASS]
        out_images.append(images[idx])
        out_labels.append(labels[idx])
    return np.concatenate(out_images), np.concatenate(out_labels)


def load_doodles():
    out_images, out_labels = [], []
    for i, category in enumerate(DOODLES):
        arr = load_quickdraw_subset(category, N_PER_DOODLE_CLASS)
        out_images.append(arr)
        out_labels.append(np.full(len(arr), DOODLE_OFFSET + i, dtype=np.int64))
        print(f"  quickdraw '{category}': {len(arr)} samples")
    return np.concatenate(out_images), np.concatenate(out_labels)


def build_and_save(out_path="data/unified.npz"):
    print("Loading MNIST digits...")
    mnist_x, mnist_y = load_mnist()
    print(f"  {len(mnist_x)} samples")

    print("Loading EMNIST letters...")
    emnist_x, emnist_y = load_emnist_letters()
    print(f"  {len(emnist_x)} samples")

    print("Loading Quick Draw doodles...")
    doodle_x, doodle_y = load_doodles()
    print(f"  {len(doodle_x)} samples")

    X = np.concatenate([mnist_x, emnist_x, doodle_x])
    y = np.concatenate([mnist_y, emnist_y, doodle_y])

    perm = np.random.RandomState(42).permutation(len(X))
    X, y = X[perm], y[perm]

    print(f"Total: {len(X)} samples across {len(CLASSES)} classes")
    np.savez_compressed(out_path, X=X, y=y, classes=np.array(CLASSES))
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    build_and_save()
