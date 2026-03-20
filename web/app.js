/**
 * ASL Live Recognizer — app.js
 *
 * Pipeline:
 *   Webcam → MediaPipe HandLandmarker → 21 landmarks
 *   → preProcessLandmarks (center + max-normalize → 42 floats)
 *   → manual NN forward pass (BatchNorm → Dense×4 with mish/softmax)
 *   → dwell + cooldown → append letter to text
 *
 * No TF.js required — inference is pure JS using float32 weights from weights.bin.
 */

import {
  HandLandmarker,
  FilesetResolver,
  DrawingUtils,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

// ── Constants ──────────────────────────────────────────────────────────────────
const LABELS = [
  "A","B","C","D","E","F","G","H","I","J","K","L","M",
  "N","O","P","Q","R","S","T","U","V","W","X","Y","Z",
];
const DWELL_FRAMES = 8;   // consecutive frames before a letter is accepted
const COOLDOWN_MS  = 500; // ms before the same letter can fire again
const BN_EPSILON   = 0.001;

// ── DOM refs ──────────────────────────────────────────────────────────────────
const video      = document.getElementById("webcam");
const canvas     = document.getElementById("overlay");
const ctx        = canvas.getContext("2d");
const letterEl   = document.getElementById("letter");
const confEl     = document.getElementById("confidence");
const dwellBar   = document.getElementById("dwell-bar");
const dwellLabel = document.getElementById("dwell-label");
const typedEl    = document.getElementById("typed-text");
const statusEl   = document.getElementById("status");
const noHandEl   = document.getElementById("no-hand-badge");

// ── App state ─────────────────────────────────────────────────────────────────
let frameStability = 0;
let lastPrediction = null;
let cooldownUntil  = 0;
let typedText      = [];
let handLandmarker, drawingUtils, weights;

// ── Weight loading ────────────────────────────────────────────────────────────
async function loadWeights() {
  const [manifestRes, binRes] = await Promise.all([
    fetch("model/weights.json"),
    fetch("model/weights.bin"),
  ]);
  const manifest = await manifestRes.json();
  const buffer   = await binRes.arrayBuffer();

  const w = {};
  for (const entry of manifest) {
    const arr = new Float32Array(buffer, entry.offset, entry.length / 4);
    w[entry.name] = { data: arr, shape: entry.shape };
  }
  return w;
}

// ── Neural network forward pass ───────────────────────────────────────────────
function batchNorm(x) {
  const gamma = weights.bn_gamma.data;
  const beta  = weights.bn_beta.data;
  const mean  = weights.bn_mean.data;
  const vari  = weights.bn_var.data;
  const out   = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) {
    out[i] = gamma[i] * (x[i] - mean[i]) / Math.sqrt(vari[i] + BN_EPSILON) + beta[i];
  }
  return out;
}

/** Dense: input [in] × kernel [in, out] + bias [out] → output [out] */
function dense(x, kernelKey, biasKey) {
  const kernel = weights[kernelKey].data;
  const bias   = weights[biasKey].data;
  const inSize  = weights[kernelKey].shape[0];
  const outSize = weights[kernelKey].shape[1];
  const out = new Float32Array(outSize);
  for (let j = 0; j < outSize; j++) {
    let sum = bias[j];
    for (let i = 0; i < inSize; i++) {
      sum += x[i] * kernel[i * outSize + j];
    }
    out[j] = sum;
  }
  return out;
}

function mish(x) {
  const out = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) {
    const v = x[i];
    // clamp to avoid exp overflow; mish(v) ≈ v for large v anyway
    const sp = v > 20 ? v : Math.log1p(Math.exp(v));
    out[i] = v * Math.tanh(sp);
  }
  return out;
}

function softmax(x) {
  let max = -Infinity;
  for (let i = 0; i < x.length; i++) if (x[i] > max) max = x[i];
  let sum = 0;
  const out = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) { out[i] = Math.exp(x[i] - max); sum += out[i]; }
  for (let i = 0; i < x.length; i++) out[i] /= sum;
  return out;
}

function predict(input42) {
  let x = batchNorm(input42);
  x = mish(dense(x, "d1_kernel", "d1_bias"));
  x = mish(dense(x, "d2_kernel", "d2_bias"));
  x = mish(dense(x, "d3_kernel", "d3_bias"));
  x = softmax(dense(x, "d4_kernel", "d4_bias"));
  return x;
}

// ── Landmark preprocessing ────────────────────────────────────────────────────
/**
 * Mirrors pre_process_landmarks() from live_asl_webcam.py exactly.
 * Landmarks are MediaPipe normalized [0,1] coords — scale doesn't matter because
 * the max-abs division cancels any constant factor.
 */
function preProcessLandmarks(landmarks) {
  const baseX = landmarks[0].x;
  const baseY = landmarks[0].y;
  const flat  = new Float32Array(42);
  for (let i = 0; i < 21; i++) {
    flat[i * 2]     = landmarks[i].x - baseX;
    flat[i * 2 + 1] = landmarks[i].y - baseY;
  }
  let maxAbs = 0;
  for (let i = 0; i < 42; i++) {
    const a = Math.abs(flat[i]);
    if (a > maxAbs) maxAbs = a;
  }
  if (maxAbs > 0) {
    for (let i = 0; i < 42; i++) flat[i] /= maxAbs;
  }
  return flat;
}

// ── Dwell + cooldown ──────────────────────────────────────────────────────────
function updateDwellAndTyping(letter) {
  if (letter === lastPrediction) {
    frameStability++;
  } else {
    frameStability = 1;
    lastPrediction = letter;
  }

  const pct = Math.min((frameStability / DWELL_FRAMES) * 100, 100);
  dwellBar.style.width = pct + "%";

  const now = Date.now();
  if (frameStability >= DWELL_FRAMES && now >= cooldownUntil) {
    typedText.push(letter);
    cooldownUntil = now + COOLDOWN_MS;
    renderTypedText();
  }
}

function renderTypedText() {
  typedEl.textContent = typedText.join("");
}

// ── Main frame loop ───────────────────────────────────────────────────────────
function processFrame(timestampMs) {
  const results = handLandmarker.detectForVideo(video, timestampMs);

  // Mirror the canvas for a natural "selfie" view
  ctx.save();
  ctx.scale(-1, 1);
  ctx.translate(-canvas.width, 0);
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  ctx.restore();

  const hasHand = results.landmarks && results.landmarks.length > 0;
  noHandEl.style.display = hasHand ? "none" : "block";

  if (hasHand) {
    const landmarks = results.landmarks[0];

    // Draw skeleton (landmarks are in unmirrored space; flip them for canvas)
    const mirroredLandmarks = landmarks.map(lm => ({ ...lm, x: 1 - lm.x }));
    drawingUtils.drawLandmarks(mirroredLandmarks, { color: "#00FF00", radius: 4 });
    drawingUtils.drawConnectors(mirroredLandmarks, HandLandmarker.HAND_CONNECTIONS, {
      color: "#00FF00",
      lineWidth: 2,
    });

    // Inference uses original (unmirrored) landmarks — geometry is the same
    const input42 = preProcessLandmarks(landmarks);
    const probs   = predict(input42);

    let topIdx = 0;
    for (let i = 1; i < 26; i++) if (probs[i] > probs[topIdx]) topIdx = i;

    const letter = LABELS[topIdx];
    const conf   = probs[topIdx];

    letterEl.textContent = letter;
    confEl.textContent   = `${(conf * 100).toFixed(0)}%`;
    dwellLabel.textContent = frameStability >= DWELL_FRAMES ? "✓ Accepted" : "Hold sign to type";

    updateDwellAndTyping(letter);
  } else {
    letterEl.textContent = "—";
    confEl.textContent   = "";
    dwellBar.style.width = "0%";
    dwellLabel.textContent = "Hold sign to type";
    frameStability = 0;
    lastPrediction = null;
  }

  requestAnimationFrame(processFrame);
}

// ── Initialization ────────────────────────────────────────────────────────────
async function init() {
  // 1. Load MediaPipe
  statusEl.textContent = "Loading MediaPipe…";
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );
  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numHands: 1,
    minHandDetectionConfidence: 0.5,
    minHandPresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });

  // 2. Load model weights
  statusEl.textContent = "Loading ASL model…";
  weights = await loadWeights();

  // 3. Start webcam
  statusEl.textContent = "Starting webcam…";
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
    audio: false,
  });
  video.srcObject = stream;
  await new Promise((resolve) => (video.onloadedmetadata = resolve));
  await video.play();

  canvas.width  = video.videoWidth;
  canvas.height = video.videoHeight;
  drawingUtils = new DrawingUtils(ctx);

  statusEl.textContent = "Running";
  statusEl.classList.add("ready");
  requestAnimationFrame(processFrame);
}

// ── Button controls ───────────────────────────────────────────────────────────
document.getElementById("btn-space").onclick = () => {
  typedText.push(" ");
  renderTypedText();
};
document.getElementById("btn-backspace").onclick = () => {
  typedText.pop();
  renderTypedText();
};
document.getElementById("btn-clear").onclick = () => {
  typedText = [];
  renderTypedText();
};

document.addEventListener("keydown", (e) => {
  const key = e.key.toUpperCase();
  if (key === "S") document.getElementById("btn-space").click();
  if (key === "B") document.getElementById("btn-backspace").click();
  if (key === "C") document.getElementById("btn-clear").click();
});

// ── Start ─────────────────────────────────────────────────────────────────────
init().catch((err) => {
  statusEl.textContent = "Error: " + err.message;
  statusEl.style.color = "#ff4444";
  console.error(err);
});
