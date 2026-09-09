# Corrections

Drop files downloaded from the app's "Download corrections" button here
(e.g. `from-me-2026-09-08.json`, one per contributor/session so files
don't collide). Any `*.json` file in this folder is automatically picked
up by `build_dataset.py` on the next run via `load_corrections.py`, which
unpacks the drawings, upsamples them (each hand-verified correction is
repeated 8x by default -- see `UPSAMPLE` in `load_corrections.py`), and
folds them into the training set alongside MNIST/EMNIST/Quick Draw.

This folder is otherwise empty by default -- that's expected.
