import { test } from "node:test";
import assert from "node:assert/strict";
import { packBits, unpackBits, loadCorrections, addCorrection, exportCorrectionsJSON } from "../corrections.js";

function fakeStorage() {
  const map = new Map();
  return {
    getItem: (k) => (map.has(k) ? map.get(k) : null),
    setItem: (k, v) => map.set(k, v),
  };
}

test("packBits/unpackBits round-trips a binary array", () => {
  const original = new Float32Array(784);
  for (let i = 0; i < 784; i++) original[i] = i % 3 === 0 ? 1 : 0;
  const packed = packBits(original);
  const unpacked = unpackBits(packed, 784);
  assert.deepEqual(Array.from(unpacked), Array.from(original));
});

test("packBits produces a much smaller representation than raw JSON", () => {
  const arr = new Float32Array(784).fill(1);
  const packed = packBits(arr);
  assert.ok(packed.length < 200, `expected packed string well under 200 chars, got ${packed.length}`);
});

test("loadCorrections returns an empty array when nothing is stored", () => {
  assert.deepEqual(loadCorrections(fakeStorage()), []);
});

test("addCorrection stores an entry and returns the new count", () => {
  const storage = fakeStorage();
  const modelInput = new Float32Array(784).fill(0);
  const count = addCorrection({ correctLabel: "tree", guessedLabel: "9", modelInput }, storage);
  assert.equal(count, 1);
  const list = loadCorrections(storage);
  assert.equal(list.length, 1);
  assert.equal(list[0].correctLabel, "tree");
  assert.equal(list[0].guessedLabel, "9");
  assert.equal(list[0].inputLength, 784);
});

test("addCorrection caps the list at MAX_CORRECTIONS, dropping oldest first", () => {
  const storage = fakeStorage();
  const modelInput = new Float32Array(784).fill(0);
  for (let i = 0; i < 305; i++) {
    addCorrection({ correctLabel: String(i), guessedLabel: "x", modelInput }, storage);
  }
  const list = loadCorrections(storage);
  assert.equal(list.length, 300);
  assert.equal(list[0].correctLabel, "5", "the oldest 5 entries should have been evicted");
  assert.equal(list[list.length - 1].correctLabel, "304");
});

test("exportCorrectionsJSON produces valid JSON with the right count", () => {
  const storage = fakeStorage();
  const modelInput = new Float32Array(784).fill(1);
  addCorrection({ correctLabel: "cat", guessedLabel: "dog", modelInput }, storage);
  addCorrection({ correctLabel: "house", guessedLabel: "tree", modelInput }, storage);
  const json = exportCorrectionsJSON(storage);
  const parsed = JSON.parse(json);
  assert.equal(parsed.count, 2);
  assert.equal(parsed.corrections.length, 2);
  assert.ok(parsed.exportedAt);
});
