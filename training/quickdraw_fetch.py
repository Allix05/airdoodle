"""Fetch a small random-ish subset of a Quick, Draw! category's 28x28
bitmaps without downloading the entire (~50-150MB) file, using an HTTP
Range request for just the header plus the bytes we need.
"""
from __future__ import annotations

import io
import urllib.request

import numpy as np

BASE_URL = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{}.npy"


def _fetch_range(url: str, start: int, end: int | None) -> bytes:
    range_header = f"bytes={start}-{end}" if end is not None else f"bytes={start}-"
    req = urllib.request.Request(url, headers={"Range": range_header})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read()


def load_quickdraw_subset(category: str, n_samples: int, offset: int = 0) -> np.ndarray:
    """Return an (n_samples, 28, 28) uint8 array for `category`.

    Only downloads the .npy header plus the exact byte range needed for
    `n_samples` starting at `offset`, instead of the whole file.
    """
    url = BASE_URL.format(category)

    # The header is small; 256 bytes is comfortably more than enough for
    # these simple (N, 784) uint8 arrays.
    header_chunk = _fetch_range(url, 0, 255)
    fp = io.BytesIO(header_chunk)
    version = np.lib.format.read_magic(fp)
    shape, fortran_order, dtype = np.lib.format._read_array_header(fp, version)
    header_end = fp.tell()

    if fortran_order:
        raise ValueError(f"unexpected fortran-order array for {category}")

    n_total, row_len = shape
    item_size = dtype.itemsize * row_len
    start_byte = header_end + offset * item_size
    end_byte = header_end + (offset + n_samples) * item_size - 1

    data = _fetch_range(url, start_byte, end_byte)
    arr = np.frombuffer(data, dtype=dtype)
    got = len(arr) // row_len
    arr = arr[: got * row_len].reshape(got, 28, 28)
    return arr
