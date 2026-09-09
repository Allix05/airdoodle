import { HandLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

import { isPinching, penPoint, Debouncer, LM } from "./hand.js";
import { preprocessDrawing } from "./preprocess.js";
import { addCorrection, loadCorrections, exportCorrectionsJSON } from "./corrections.js";

// Must exactly match training/classes.py's CLASSES order -- this is the
// model's output label space.
const DIGITS = "0123456789".split("");
const LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");
const DOODLES = ["cat", "house", "tree", "car", "sun", "star", "fish", "umbrella"];
const CLASSES = [...DIGITS, ...LETTERS, ...DOODLES];

const COLORS = ["#ffffff", "#ff5b7f", "#ffcc4d", "#4dffb0", "#5b8cff", "#c37bff"];

const video = document.getElementById("video");
const videoWrap = document.querySelector(".video-wrap");
const controlPanel = document.querySelector(".control-panel");
const flashOverlay = document.getElementById("flashOverlay");
const drawCanvas = document.getElementById("drawCanvas");
const drawCtx = drawCanvas.getContext("2d");
const overlay = document.getElementById("overlay");
const overlayCtx = overlay.getContext("2d");
const startOverlay = document.getElementById("startOverlay");
const startBtn = document.getElementById("startBtn");
const loadingOverlay = document.getElementById("loadingOverlay");
const loadingText = document.getElementById("loadingText");
const errorBanner = document.getElementById("errorBanner");
const colorSwatchesEl = document.getElementById("colorSwatches");
const clearBtn = document.getElementById("clearBtn");
const guessBtn = document.getElementById("guessBtn");
const penStatusEl = document.getElementById("penStatus");
const penStatusText = document.getElementById("penStatusText");
const resultBlock = document.getElementById("resultBlock");
const snapshotCanvas = document.getElementById("snapshotCanvas");
const snapshotCtx = snapshotCanvas.getContext("2d");
const resultGuess = document.getElementById("resultGuess");
const topGuessesEl = document.getElementById("topGuesses");
const categoryChipsEl = document.getElementById("categoryChips");
const feedbackAsk = document.getElementById("feedbackAsk");
const feedbackYes = document.getElementById("feedbackYes");
const feedbackNo = document.getElementById("feedbackNo");
const feedbackCorrect = document.getElementById("feedbackCorrect");
const feedbackChipsEl = document.getElementById("feedbackChips");
const feedbackThanks = document.getElementById("feedbackThanks");
const correctionsCountEl = document.getElementById("correctionsCount");
const downloadCorrectionsBtn = document.getElementById("downloadCorrections");

let currentColor = COLORS[0];
COLORS.forEach((color, i) => {
  const btn = document.createElement("button");
  btn.className = "swatch" + (i === 0 ? " active" : "");
  btn.style.background = color;
  btn.style.setProperty("--sw-color", color);
  btn.addEventListener("click", () => {
    currentColor = color;
    [...colorSwatchesEl.children].forEach((c) => c.classList.remove("active"));
    btn.classList.add("active");
  });
  colorSwatchesEl.appendChild(btn);
});

function formatLabel(name) {
  return name.length > 1 ? name[0].toUpperCase() + name.slice(1) : name;
}

CLASSES.forEach((name) => {
  const chip = document.createElement("span");
  chip.className = "chip";
  chip.textContent = formatLabel(name);
  categoryChipsEl.appendChild(chip);
});

CLASSES.forEach((name) => {
  const chip = document.createElement("button");
  chip.className = "feedback-chip";
  chip.textContent = formatLabel(name);
  chip.addEventListener("click", () => {
    if (!lastModelInput) return;
    addCorrection({ correctLabel: name, guessedLabel: lastGuessedLabel, modelInput: lastModelInput }, window.localStorage);
    feedbackCorrect.classList.add("hidden");
    feedbackThanks.classList.remove("hidden");
    updateCorrectionsUI();
  });
  feedbackChipsEl.appendChild(chip);
});

function updateCorrectionsUI() {
  const count = loadCorrections(window.localStorage).length;
  correctionsCountEl.textContent =
    count === 0 ? "No corrections collected yet." : `${count} correction${count === 1 ? "" : "s"} collected.`;
  downloadCorrectionsBtn.disabled = count === 0;
}

function resetFeedbackUI() {
  feedbackAsk.classList.remove("hidden");
  feedbackCorrect.classList.add("hidden");
  feedbackThanks.classList.add("hidden");
}

feedbackYes.addEventListener("click", () => {
  feedbackAsk.classList.add("hidden");
});

feedbackNo.addEventListener("click", () => {
  feedbackAsk.classList.add("hidden");
  feedbackCorrect.classList.remove("hidden");
});

downloadCorrectionsBtn.addEventListener("click", () => {
  const json = exportCorrectionsJSON(window.localStorage);
  const blob = new Blob([json], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "airdoodle-corrections.json";
  a.click();
  URL.revokeObjectURL(url);
});

updateCorrectionsUI();

function showError(message) {
  errorBanner.textContent = message;
  errorBanner.classList.remove("hidden");
}

let handLandmarker = null;
let session = null;
let running = false;
let lastPoint = null;
let lastModelInput = null;
let lastGuessedLabel = null;
const pinchDebouncer = new Debouncer(3, false);

// The draw canvas is kept transparent (not opaque black) so the live
// camera feed shows through behind your strokes -- purely visual, since
// preprocess.js's binary mask only looks at drawn (non-background) pixel
// color, and transparent background pixels read as background either way.
function clearDrawCanvas() {
  drawCtx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
}

function resizeCanvases() {
  const rect = video.getBoundingClientRect();
  const w = Math.round(rect.width);
  const h = Math.round(rect.height);
  for (const c of [drawCanvas, overlay]) {
    if (c.width !== w || c.height !== h) {
      c.width = w;
      c.height = h;
      if (c === drawCanvas) clearDrawCanvas();
    }
  }
}

async function init() {
  try {
    startOverlay.classList.add("hidden");
    loadingOverlay.classList.remove("hidden");

    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false });
    video.srcObject = stream;
    await video.play();

    resizeCanvases();
    clearDrawCanvas();

    loadingText.textContent = "Loading the drawing-recognition model...";
    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/";
    session = await ort.InferenceSession.create("model.onnx", { executionProviders: ["wasm"] });

    loadingText.textContent = "Loading hand-tracking model...";
    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
    );
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        delegate: "GPU",
      },
      runningMode: "VIDEO",
      numHands: 1,
    });

    loadingOverlay.classList.add("hidden");
    running = true;
    requestAnimationFrame(loop);
  } catch (err) {
    loadingOverlay.classList.add("hidden");
    showError("Couldn't start: " + err.message);
  }
}

function drawCursor(px, py, pinching) {
  overlayCtx.beginPath();
  overlayCtx.arc(px, py, pinching ? 10 : 7, 0, Math.PI * 2);
  overlayCtx.fillStyle = pinching ? currentColor : "rgba(200, 210, 235, 0.5)";
  overlayCtx.fill();
  overlayCtx.lineWidth = 2;
  overlayCtx.strokeStyle = "rgba(255,255,255,0.8)";
  overlayCtx.stroke();
}

function loop() {
  if (!running) return;
  resizeCanvases();
  overlayCtx.clearRect(0, 0, overlay.width, overlay.height);

  const nowMs = performance.now();
  const result = handLandmarker.detectForVideo(video, nowMs);
  const hands = result.landmarks || [];
  const landmarks = hands[0];

  if (landmarks) {
    const pinchingRaw = isPinching(landmarks);
    const pinching = pinchDebouncer.update(pinchingRaw);
    penStatusEl.classList.toggle("down", pinching);
    penStatusText.textContent = pinching ? "Drawing..." : "Pinch thumb + index finger to draw";

    const p = penPoint(landmarks);
    const px = p.x * drawCanvas.width;
    const py = p.y * drawCanvas.height;

    if (pinching) {
      drawCtx.strokeStyle = currentColor;
      drawCtx.fillStyle = currentColor;
      drawCtx.lineWidth = Math.max(6, drawCanvas.width * 0.018);
      drawCtx.lineCap = "round";
      drawCtx.lineJoin = "round";
      if (lastPoint) {
        drawCtx.beginPath();
        drawCtx.moveTo(lastPoint.x, lastPoint.y);
        drawCtx.lineTo(px, py);
        drawCtx.stroke();
      } else {
        drawCtx.beginPath();
        drawCtx.arc(px, py, drawCtx.lineWidth / 2, 0, Math.PI * 2);
        drawCtx.fill();
      }
      lastPoint = { x: px, y: py };
    } else {
      lastPoint = null;
    }

    drawCursor(px, py, pinching);
  } else {
    lastPoint = null;
    penStatusEl.classList.remove("down");
    penStatusText.textContent = "Show your hand to the camera";
  }

  requestAnimationFrame(loop);
}

// The draw canvas is displayed mirrored via a CSS transform (so it feels
// natural, like a mirror) but that's purely visual -- the underlying pixel
// buffer is never actually flipped, so toDataURL()/getImageData() return
// the *unmirrored* version, which is backwards relative to what you saw
// yourself drawing on screen. For asymmetric shapes (letters especially)
// that backwards version is what the model would otherwise classify,
// which is wrong. This draws the canvas onto a fresh, actually-flipped
// canvas so everything downstream (classification, the flying photo,
// the snapshot preview) matches what you actually saw.
function createMirroredCanvas(sourceCanvas) {
  const mirrored = document.createElement("canvas");
  mirrored.width = sourceCanvas.width;
  mirrored.height = sourceCanvas.height;
  const mctx = mirrored.getContext("2d");
  mctx.translate(mirrored.width, 0);
  mctx.scale(-1, 1);
  mctx.drawImage(sourceCanvas, 0, 0);
  return mirrored;
}

// Camera-flash + "photo flies over to be processed" effect. Takes the
// (already mirror-corrected) drawing data URL and animates it shrinking
// into the control panel. Resolves once the flight finishes.
function animateCapture(dataUrl) {
  return new Promise((resolve) => {
    const startRect = videoWrap.getBoundingClientRect();
    const destRect = controlPanel.getBoundingClientRect();

    const photo = document.createElement("img");
    photo.src = dataUrl;
    photo.className = "flying-photo";
    photo.style.left = `${startRect.left}px`;
    photo.style.top = `${startRect.top}px`;
    photo.style.width = `${startRect.width}px`;
    photo.style.height = `${startRect.height}px`;
    photo.style.borderRadius = "14px";
    document.body.appendChild(photo);
    // Force the browser to commit the starting position/size before we
    // change the transform below, so the transition has a "from" state to
    // animate away from. Doesn't depend on requestAnimationFrame, which
    // browsers throttle for backgrounded/hidden tabs.
    void photo.offsetWidth;

    flashOverlay.classList.remove("flash");
    void flashOverlay.offsetWidth;
    flashOverlay.classList.add("flash");

    const targetSize = 64;
    const targetX = destRect.left + destRect.width / 2 - targetSize / 2;
    const targetY = destRect.top + 46;
    const dx = targetX - startRect.left;
    const dy = targetY - startRect.top;
    const scale = targetSize / startRect.width;

    photo.style.transform = `translate(${dx}px, ${dy}px) scale(${scale})`;
    photo.style.opacity = "0";
    photo.style.borderRadius = "50%";

    setTimeout(() => {
      photo.remove();
      resolve();
    }, 620);
  });
}

clearBtn.addEventListener("click", () => {
  clearDrawCanvas();
  lastPoint = null;
  resultBlock.classList.remove("visible");
});

guessBtn.addEventListener("click", async () => {
  if (!session) return;
  const mirroredCanvas = createMirroredCanvas(drawCanvas);
  const imageData = mirroredCanvas.getContext("2d").getImageData(0, 0, mirroredCanvas.width, mirroredCanvas.height);
  const { modelInput, isEmpty } = preprocessDrawing(imageData.data, mirroredCanvas.width, mirroredCanvas.height);

  if (isEmpty) {
    resultGuess.textContent = "Draw something first!";
    topGuessesEl.innerHTML = "";
    feedbackAsk.classList.add("hidden");
    feedbackCorrect.classList.add("hidden");
    feedbackThanks.classList.add("hidden");
    resultBlock.classList.add("visible");
    return;
  }

  guessBtn.disabled = true;

  // Run inference while the capture animation plays, so the guess is
  // ready right as the photo "lands" in the panel.
  const inferencePromise = session.run({ input: new ort.Tensor("float32", modelInput, [1, 1, 28, 28]) });
  await animateCapture(mirroredCanvas.toDataURL());
  const outputs = await inferencePromise;
  const probs = outputs.probs.data;

  guessBtn.disabled = false;

  // Render what the model actually sees (the 28x28 preprocessed input).
  const snapImage = snapshotCtx.createImageData(28, 28);
  for (let i = 0; i < 28 * 28; i++) {
    const v = Math.round(modelInput[i] * 255);
    snapImage.data[i * 4] = v;
    snapImage.data[i * 4 + 1] = v;
    snapImage.data[i * 4 + 2] = v;
    snapImage.data[i * 4 + 3] = 255;
  }
  snapshotCtx.putImageData(snapImage, 0, 0);

  const ranked = Array.from(probs)
    .map((p, i) => ({ p, i }))
    .sort((a, b) => b.p - a.p);

  lastModelInput = modelInput;
  lastGuessedLabel = CLASSES[ranked[0].i];
  resultGuess.textContent = formatLabel(lastGuessedLabel);

  topGuessesEl.innerHTML = "";
  ranked.slice(0, 5).forEach(({ p, i }) => {
    const row = document.createElement("div");
    row.className = "guess-row";
    row.innerHTML = `
      <div class="label">${formatLabel(CLASSES[i])}</div>
      <div class="bar-track"><div class="bar-fill" style="width:${(p * 100).toFixed(0)}%"></div></div>
      <div class="pct">${(p * 100).toFixed(0)}%</div>
    `;
    topGuessesEl.appendChild(row);
  });

  resetFeedbackUI();
  resultBlock.classList.remove("visible");
  // Force reflow so re-triggering the class on a fresh guess re-plays the
  // transition even if the block was already visible from a prior guess.
  void resultBlock.offsetWidth;
  resultBlock.classList.add("visible");
});

startBtn.addEventListener("click", init);
