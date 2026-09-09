# AirDoodle

[![CI](https://github.com/Allix05/airdoodle/actions/workflows/ci.yml/badge.svg)](https://github.com/Allix05/airdoodle/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Draw in the air with your webcam and let a neural network guess what it is. Pinch your thumb and index finger together to draw, pick a color, sketch a digit, a letter, or a doodle, then hit **Guess** — a CNN trained on real handwriting and sketch data classifies your final drawing, right there in your browser.

**[Try it live](https://allix05.github.io/airdoodle/)** — 100% client-side: hand tracking and inference both run locally via WebAssembly. Your camera feed never leaves your device.

<!-- SCREENSHOT_PLACEHOLDER -->

## How it works

```
Webcam ──▶ MediaPipe Hand Landmarker ──▶ 21 landmarks
                                              │
                                              ▼
                                  pinch detection (hand.js)
                                              │
                                              ▼
                                strokes accumulated on canvas
                                              │
                                    (you click "Guess!")
                                              ▼
                          crop to content, center, downscale to 28x28
                                       (preprocess.js)
                                              │
                                              ▼
                                 ONNX CNN classifier (44 classes)
                                              │
                                              ▼
                                  guess + confidence bars
```

1. **[`hand.js`](docs/hand.js)** — tracks the pinch gesture (thumb tip to index tip distance, normalized by palm width so it works at any distance from the camera) via MediaPipe's Hand Landmarker, debounced to avoid jittery pen up/down flicker.
2. **Drawing** (`app.js`) — while pinching, the midpoint between thumb and index fingertips draws onto a persistent canvas in your chosen color.
3. **[`preprocess.js`](docs/preprocess.js)** — when you click Guess, the drawing is converted to a binary stroke mask (color-agnostic — any drawn color counts, only shape matters), cropped to its bounding box, centered with a margin, and downscaled to 28x28 grayscale: the exact convention MNIST, EMNIST, and Google's Quick Draw dataset all already share.
4. **[`training/`](training/)** — a PyTorch CNN (`DoodleNet`) trained on a unified dataset built from three real sources, then exported to ONNX and run client-side via [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/).

## Results

`DoodleNet` (a small CNN: 3 conv layers + 2 FC layers, ~1.6M parameters) trained for 12 epochs on 195K training images (212K total across 44 classes, 8% held out for validation) — about 12 minutes on a laptop CPU, no GPU:

| | |
|---|---|
| **Validation accuracy** | **96.2%** across all 44 classes |
| Training loss | 0.58 → 0.11 |
| Training data | 60K MNIST digits + 104K EMNIST letters + 48K Quick Draw doodles (6K/category) |

## What it can recognize

- **Digits** 0-9 (from [MNIST](http://yann.lecun.com/exdb/mnist/))
- **Letters** A-Z (from [EMNIST](https://www.nist.gov/itl/products-and-services/emnist-dataset), letters split)
- **Doodles**: cat, house, tree, car, sun, star, fish, umbrella (from Google's [Quick, Draw!](https://quickdraw.withgoogle.com/data) dataset)

44 classes total, one unified classifier.

## Try it locally

```bash
python -m http.server 8091 --directory docs
# open http://localhost:8091
```

### Run the tests

The drawing/preprocessing logic is pure and unit-tested with Node's built-in test runner — no camera needed:

```bash
node --test
```

### Train your own model

```bash
cd training
pip install -r requirements.txt
python build_dataset.py   # downloads MNIST/EMNIST, fetches a Quick Draw subset via HTTP range requests
python train.py           # trains DoodleNet, saves checkpoints/best.pt
python export_onnx.py     # exports to docs/model.onnx
```

`build_dataset.py` doesn't download entire Quick Draw category files (each is 50-150MB) — it uses HTTP range requests to fetch just the header plus the exact byte range needed for the sample count requested (see [`quickdraw_fetch.py`](training/quickdraw_fetch.py)).

## Project structure

```
docs/                fully static GitHub Pages app
  hand.js               pinch detection + debouncing (pure)
  preprocess.js         canvas -> 28x28 model input (pure)
  app.js                camera capture, drawing, ONNX inference, UI glue
  model.onnx             exported classifier
  tests/                node:test unit tests
training/            PyTorch training pipeline
  classes.py             the unified label space
  quickdraw_fetch.py     partial-download Quick Draw fetcher
  build_dataset.py       assembles the unified dataset
  model.py               DoodleNet CNN
  train.py               training loop
  export_onnx.py         ONNX export
```

## Why these design choices

- **Color-agnostic preprocessing** — the model only ever sees stroke shape (bright pixels on black), never hue, so drawing in any color works identically. Color is purely for your enjoyment.
- **Crop-and-center before downscaling** — matches how MNIST/EMNIST/Quick Draw were themselves constructed, so a drawing made at any position or scale on the canvas maps onto the same input distribution the model was trained on.
- **A single unified classifier** across three very different data sources works because they all already share the 28x28 grayscale bitmap convention — no format conversion needed, just concatenation.

## License

MIT — see [LICENSE](LICENSE).
