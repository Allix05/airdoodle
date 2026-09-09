import { HandLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

import { isPinching, penPoint, Debouncer, LM } from "./hand.js";
import { preprocessDrawing } from "./preprocess.js";

// Must exactly match training/classes.py's CLASSES order -- this is the
// model's output label space.
const DIGITS = "0123456789".split("");
const LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");
const DOODLES = ["cat", "house", "tree", "car", "sun", "star", "fish", "umbrella"];
const CLASSES = [...DIGITS, ...LETTERS, ...DOODLES];

const COLORS = ["#ffffff", "#ff5b7f", "#ffcc4d", "#4dffb0", "#5b8cff", "#c37bff"];

const video = document.getElementById("video");
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

let currentColor = COLORS[0];
COLORS.forEach((color, i) => {
  const btn = document.createElement("button");
  btn.className = "swatch" + (i === 0 ? " active" : "");
  btn.style.background = color;
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

function showError(message) {
  errorBanner.textContent = message;
  errorBanner.classList.remove("hidden");
}

let handLandmarker = null;
let session = null;
let running = false;
let lastPoint = null;
const pinchDebouncer = new Debouncer(3, false);

function fillCanvasBlack(canvas, ctx) {
  ctx.fillStyle = "#000000";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
}

function resizeCanvases() {
  const rect = video.getBoundingClientRect();
  const w = Math.round(rect.width);
  const h = Math.round(rect.height);
  for (const c of [drawCanvas, overlay]) {
    if (c.width !== w || c.height !== h) {
      c.width = w;
      c.height = h;
      if (c === drawCanvas) fillCanvasBlack(drawCanvas, drawCtx);
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
    fillCanvasBlack(drawCanvas, drawCtx);

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

clearBtn.addEventListener("click", () => {
  fillCanvasBlack(drawCanvas, drawCtx);
  lastPoint = null;
  resultBlock.style.display = "none";
});

guessBtn.addEventListener("click", async () => {
  if (!session) return;
  const imageData = drawCtx.getImageData(0, 0, drawCanvas.width, drawCanvas.height);
  const { modelInput, isEmpty } = preprocessDrawing(imageData.data, drawCanvas.width, drawCanvas.height);

  if (isEmpty) {
    resultBlock.style.display = "block";
    resultGuess.textContent = "Draw something first!";
    topGuessesEl.innerHTML = "";
    return;
  }

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

  const tensor = new ort.Tensor("float32", modelInput, [1, 1, 28, 28]);
  const outputs = await session.run({ input: tensor });
  const probs = outputs.probs.data;

  const ranked = Array.from(probs)
    .map((p, i) => ({ p, i }))
    .sort((a, b) => b.p - a.p);

  resultBlock.style.display = "block";
  resultGuess.textContent = formatLabel(CLASSES[ranked[0].i]);

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
});

startBtn.addEventListener("click", init);
