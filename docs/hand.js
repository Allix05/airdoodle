// Pinch-to-draw hand logic, pure and camera-independent so it's unit
// testable. Landmark indices follow the standard MediaPipe Hand Landmarker
// layout (same as used in the Chordinate project).
const LM = {
  WRIST: 0,
  THUMB_TIP: 4,
  INDEX_MCP: 5,
  INDEX_TIP: 8,
  PINKY_MCP: 17,
};

function dist2D(a, b) {
  const dx = a.x - b.x, dy = a.y - b.y;
  return Math.sqrt(dx * dx + dy * dy);
}

// Distance between thumb and index fingertips, normalized by palm width
// (index_mcp to pinky_mcp) so the pinch threshold works regardless of how
// close or far the hand is from the camera.
export function pinchRatio(landmarks) {
  const palmWidth = dist2D(landmarks[LM.INDEX_MCP], landmarks[LM.PINKY_MCP]);
  if (palmWidth === 0) return Infinity;
  return dist2D(landmarks[LM.THUMB_TIP], landmarks[LM.INDEX_TIP]) / palmWidth;
}

export function isPinching(landmarks, threshold = 0.55) {
  return pinchRatio(landmarks) < threshold;
}

// The point to draw at: the midpoint between thumb tip and index tip.
export function penPoint(landmarks) {
  const a = landmarks[LM.THUMB_TIP], b = landmarks[LM.INDEX_TIP];
  return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
}

// Debounces a raw per-frame boolean (e.g. pinch state) so jitter near the
// threshold doesn't produce flickery pen-up/pen-down toggling.
export class Debouncer {
  constructor(stableFrames = 3, initial = false) {
    this.stableFrames = stableFrames;
    this.current = initial;
    this.pending = initial;
    this.count = 0;
  }

  update(raw) {
    if (raw === this.pending) {
      this.count++;
    } else {
      this.pending = raw;
      this.count = 1;
    }
    if (this.count >= this.stableFrames) {
      this.current = this.pending;
    }
    return this.current;
  }
}

export { LM };
