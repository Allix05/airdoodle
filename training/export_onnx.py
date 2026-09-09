#!/usr/bin/env python
"""Export the trained checkpoint to ONNX for the browser demo.

    python export_onnx.py --checkpoint checkpoints/best.pt --out ../docs/model.onnx
"""
import argparse
import os

import onnx
import torch

from classes import CLASSES
from model import DoodleNet


class InferenceWrapper(torch.nn.Module):
    """Exports softmax probabilities directly (simpler for the JS side)."""

    def __init__(self, model: DoodleNet):
        super().__init__()
        self.model = model

    def forward(self, x):
        logits = self.model(x)
        return torch.softmax(logits, dim=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--out", default="../docs/model.onnx")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    model = DoodleNet(len(CLASSES))
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()

    wrapper = InferenceWrapper(model)
    wrapper.eval()
    dummy = torch.zeros(1, 1, 28, 28, dtype=torch.float32)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.onnx.export(
        wrapper,
        dummy,
        args.out,
        input_names=["input"],
        output_names=["probs"],
        dynamic_axes={"input": {0: "batch"}, "probs": {0: "batch"}},
        opset_version=args.opset,
    )

    data_file = args.out + ".data"
    if os.path.exists(data_file):
        onnx_model = onnx.load(args.out)
        onnx.save(onnx_model, args.out, save_as_external_data=False)
        os.remove(data_file)

    size_kb = os.path.getsize(args.out) / 1024
    print(f"Exported {args.out} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
