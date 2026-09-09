// Converts the user's colored drawing canvas into the exact 28x28
// grayscale bitmap format the model was trained on (MNIST/EMNIST/Quick
// Draw's shared "numpy_bitmap" convention: bright strokes on a black
// background, cropped to the drawing's bounding box, then centered in a
// square with a small margin before the final resize -- the same
// centering convention those datasets use).
//
// Pure pixel-array logic (no DOM canvas APIs), so it's unit-testable with
// plain typed arrays.

// `pixels` is a Uint8ClampedArray-like RGBA buffer, `width`/`height` its
// dimensions. Returns a new Uint8Array (width*height) of "is this pixel
// part of a stroke" (255) vs background (0), color-agnostic.
export function toBinaryMask(pixels, width, height, threshold = 24) {
  const mask = new Uint8Array(width * height);
  for (let i = 0; i < width * height; i++) {
    const r = pixels[i * 4], g = pixels[i * 4 + 1], b = pixels[i * 4 + 2];
    mask[i] = (r + g + b) > threshold ? 255 : 0;
  }
  return mask;
}

// Returns {minX, minY, maxX, maxY} of nonzero pixels in `mask`, or null if
// the mask is entirely empty (nothing drawn).
export function boundingBox(mask, width, height) {
  let minX = width, minY = height, maxX = -1, maxY = -1;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (mask[y * width + x] !== 0) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
  }
  if (maxX < 0) return null;
  return { minX, minY, maxX, maxY };
}

// Nearest-neighbor resample of a square crop of `mask` (defined by `box`,
// with `marginFrac` extra padding on each side) into a `size`x`size`
// Float32Array normalized to [0, 1]. This mirrors the standard
// "crop-to-content, center with margin, downscale" preprocessing used for
// MNIST-style digit/letter datasets.
export function toModelInput(mask, width, height, box, size = 28, marginFrac = 0.15) {
  const out = new Float32Array(size * size);
  if (!box) return out; // nothing drawn -> all-zero input

  const boxW = box.maxX - box.minX + 1;
  const boxH = box.maxY - box.minY + 1;
  const side = Math.max(boxW, boxH) * (1 + marginFrac * 2);
  const cx = (box.minX + box.maxX) / 2;
  const cy = (box.minY + box.maxY) / 2;
  const srcMinX = cx - side / 2;
  const srcMinY = cy - side / 2;

  for (let oy = 0; oy < size; oy++) {
    for (let ox = 0; ox < size; ox++) {
      const srcX = Math.floor(srcMinX + ((ox + 0.5) / size) * side);
      const srcY = Math.floor(srcMinY + ((oy + 0.5) / size) * side);
      if (srcX >= 0 && srcX < width && srcY >= 0 && srcY < height) {
        out[oy * size + ox] = mask[srcY * width + srcX] / 255;
      }
    }
  }
  return out;
}

// Convenience: run the full pipeline on raw RGBA pixel data.
export function preprocessDrawing(pixels, width, height, size = 28) {
  const mask = toBinaryMask(pixels, width, height);
  const box = boundingBox(mask, width, height);
  const modelInput = toModelInput(mask, width, height, box, size);
  return { mask, box, modelInput, isEmpty: box === null };
}
