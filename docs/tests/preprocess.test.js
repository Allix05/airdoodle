import { test } from "node:test";
import assert from "node:assert/strict";
import { toBinaryMask, boundingBox, toModelInput, preprocessDrawing } from "../preprocess.js";

function makeRGBA(width, height, fillFn) {
  const pixels = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const [r, g, b] = fillFn(x, y);
      const i = (y * width + x) * 4;
      pixels[i] = r; pixels[i + 1] = g; pixels[i + 2] = b; pixels[i + 3] = 255;
    }
  }
  return pixels;
}

test("toBinaryMask treats black background as 0 and colored strokes as 255", () => {
  const pixels = makeRGBA(4, 1, (x) => (x === 2 ? [0, 255, 255] : [0, 0, 0]));
  const mask = toBinaryMask(pixels, 4, 1);
  assert.deepEqual(Array.from(mask), [0, 0, 255, 0]);
});

test("toBinaryMask is color-agnostic: red, blue, and white all count as stroke", () => {
  const pixels = makeRGBA(3, 1, (x) => [[255, 0, 0], [0, 0, 255], [255, 255, 255]][x]);
  const mask = toBinaryMask(pixels, 3, 1);
  assert.deepEqual(Array.from(mask), [255, 255, 255]);
});

test("boundingBox returns null for an empty (all-background) mask", () => {
  const mask = new Uint8Array(100); // all zero
  assert.equal(boundingBox(mask, 10, 10), null);
});

test("boundingBox finds the tight box around drawn pixels", () => {
  const width = 10, height = 10;
  const mask = new Uint8Array(width * height);
  mask[3 * width + 2] = 255; // (x=2,y=3)
  mask[5 * width + 7] = 255; // (x=7,y=5)
  const box = boundingBox(mask, width, height);
  assert.deepEqual(box, { minX: 2, minY: 3, maxX: 7, maxY: 5 });
});

test("toModelInput of an empty drawing is all zeros", () => {
  const mask = new Uint8Array(28 * 28);
  const out = toModelInput(mask, 28, 28, null);
  assert.equal(out.length, 28 * 28);
  assert.ok(out.every((v) => v === 0));
});

test("toModelInput centers and fills a simple square stroke", () => {
  const width = 100, height = 100;
  const mask = new Uint8Array(width * height);
  // A filled 20x20 square roughly in the corner.
  for (let y = 10; y < 30; y++) {
    for (let x = 10; x < 30; x++) mask[y * width + x] = 255;
  }
  const box = boundingBox(mask, width, height);
  const out = toModelInput(mask, width, height, box, 28, 0.15);

  // Most of the resulting 28x28 image should be "on" since the source
  // region was a solid filled square with a modest margin.
  let onCount = 0;
  for (let i = 0; i < out.length; i++) if (out[i] > 0) onCount++;
  assert.ok(onCount > 28 * 28 * 0.4, `expected a substantial filled area, got ${onCount}/${out.length}`);

  // And it should be roughly centered: the very corners should be
  // background (since we added margin around the tight box).
  assert.equal(out[0], 0, "top-left corner should be background");
  assert.equal(out[27 * 28 + 27], 0, "bottom-right corner should be background");
});

test("preprocessDrawing reports isEmpty for a blank canvas", () => {
  const pixels = makeRGBA(10, 10, () => [0, 0, 0]);
  const result = preprocessDrawing(pixels, 10, 10);
  assert.equal(result.isEmpty, true);
});

test("preprocessDrawing reports not-empty once something is drawn", () => {
  const pixels = makeRGBA(10, 10, (x, y) => (x === 5 && y === 5 ? [255, 255, 255] : [0, 0, 0]));
  const result = preprocessDrawing(pixels, 10, 10);
  assert.equal(result.isEmpty, false);
  assert.equal(result.modelInput.length, 28 * 28);
});
