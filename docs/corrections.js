// Local, on-device storage of user corrections ("that guess was wrong, it
// was actually X") for possible future model improvement. Pure logic
// (storage is injected) so it's unit-testable without a real browser.
//
// Important honesty note: this does NOT retrain the model in the browser.
// ONNX Runtime Web only runs inference. This just collects labeled
// examples locally so they can be exported and used to fine-tune a future
// version of the model offline.

const STORAGE_KEY = "airdoodle_corrections_v1";
const MAX_CORRECTIONS = 300;

// modelInput values are always exactly 0 or 1 (see preprocess.js), so we
// can pack 28*28=784 of them into 98 bytes instead of ~10KB of JSON floats.
export function packBits(floatArray) {
  const bytes = new Uint8Array(Math.ceil(floatArray.length / 8));
  for (let i = 0; i < floatArray.length; i++) {
    if (floatArray[i] > 0.5) bytes[i >> 3] |= 1 << (i & 7);
  }
  let binary = "";
  for (const b of bytes) binary += String.fromCharCode(b);
  return btoa(binary);
}

export function unpackBits(base64, length) {
  const binary = atob(base64);
  const out = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    const byte = binary.charCodeAt(i >> 3);
    out[i] = byte & (1 << (i & 7)) ? 1 : 0;
  }
  return out;
}

export function loadCorrections(storage) {
  try {
    const raw = storage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

// entry: { correctLabel, guessedLabel, modelInput (Float32Array/array of
// 0/1, length 784) }. Returns the new total count.
export function addCorrection(entry, storage) {
  const list = loadCorrections(storage);
  list.push({
    correctLabel: entry.correctLabel,
    guessedLabel: entry.guessedLabel,
    inputLength: entry.modelInput.length,
    inputPacked: packBits(entry.modelInput),
    timestamp: Date.now(),
  });
  while (list.length > MAX_CORRECTIONS) list.shift();
  storage.setItem(STORAGE_KEY, JSON.stringify(list));
  return list.length;
}

export function exportCorrectionsJSON(storage) {
  const list = loadCorrections(storage);
  return JSON.stringify({ exportedAt: new Date().toISOString(), count: list.length, corrections: list }, null, 2);
}
