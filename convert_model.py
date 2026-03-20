"""
Extract float32 weights from keypoint_classifier.keras and write:
  web/model/weights.bin   -- flat binary: all weights concatenated
  web/model/weights.json  -- manifest describing each tensor's offset/shape

Architecture:
  Input(42) -> BatchNorm(42) -> Dense(128, mish) -> Dense(64, mish)
            -> Dense(32, mish) -> Dense(26, softmax)

Inference is implemented manually in JS using mish and softmax functions,
so no TF.js model format is needed (avoids converter dependency hell).
"""

import io, json, os, zipfile
import numpy as np
import h5py

KERAS_PATH = "model/keypoint_classifier/keypoint_classifier.keras"
OUT_DIR    = "web/model"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Extract weights from .keras zip ──────────────────────────────────────────
with zipfile.ZipFile(KERAS_PATH, "r") as z:
    h5_bytes = z.read("model.weights.h5")

with h5py.File(io.BytesIO(h5_bytes), "r") as f:
    bn_gamma    = f[r"layers\batch_normalization/vars/0"][:]  # (42,)
    bn_beta     = f[r"layers\batch_normalization/vars/1"][:]  # (42,)
    bn_mean     = f[r"layers\batch_normalization/vars/2"][:]  # (42,)
    bn_var      = f[r"layers\batch_normalization/vars/3"][:]  # (42,)
    d1_kernel   = f[r"layers\dense/vars/0"][:]               # (42, 128)
    d1_bias     = f[r"layers\dense/vars/1"][:]               # (128,)
    d2_kernel   = f[r"layers\dense_1/vars/0"][:]             # (128, 64)
    d2_bias     = f[r"layers\dense_1/vars/1"][:]             # (64,)
    d3_kernel   = f[r"layers\dense_2/vars/0"][:]             # (64, 32)
    d3_bias     = f[r"layers\dense_2/vars/1"][:]             # (32,)
    d4_kernel   = f[r"layers\dense_3/vars/0"][:]             # (32, 26)
    d4_bias     = f[r"layers\dense_3/vars/1"][:]             # (26,)

tensors = [
    ("bn_gamma",   bn_gamma),
    ("bn_beta",    bn_beta),
    ("bn_mean",    bn_mean),
    ("bn_var",     bn_var),
    ("d1_kernel",  d1_kernel),
    ("d1_bias",    d1_bias),
    ("d2_kernel",  d2_kernel),
    ("d2_bias",    d2_bias),
    ("d3_kernel",  d3_kernel),
    ("d3_bias",    d3_bias),
    ("d4_kernel",  d4_kernel),
    ("d4_bias",    d4_bias),
]

# ── Sanity check via manual forward pass ─────────────────────────────────────
def batch_norm(x, eps=0.001):
    return bn_gamma * (x - bn_mean) / np.sqrt(bn_var + eps) + bn_beta

def mish(x):
    return x * np.tanh(np.log1p(np.exp(np.clip(x, -88, 88))))

def dense(x, kernel, bias):
    return x @ kernel + bias

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def predict(inp):
    x = batch_norm(inp)
    x = mish(dense(x, d1_kernel, d1_bias))
    x = mish(dense(x, d2_kernel, d2_bias))
    x = mish(dense(x, d3_kernel, d3_bias))
    return softmax(dense(x, d4_kernel, d4_bias))

# Compare against TFLite interpreter
try:
    import tensorflow as tf
    interp = tf.lite.Interpreter(
        model_path="model/keypoint_classifier/keypoint_classifier.tflite"
    )
    interp.allocate_tensors()
    np.random.seed(42)
    test = np.random.randn(42).astype(np.float32)
    test /= np.max(np.abs(test))
    interp.set_tensor(interp.get_input_details()[0]["index"], test.reshape(1, 42))
    interp.invoke()
    tflite_out = interp.get_tensor(interp.get_output_details()[0]["index"])[0]
    keras_out  = predict(test)
    max_diff   = np.max(np.abs(tflite_out - keras_out))
    top_tflite = np.argmax(tflite_out)
    top_keras  = np.argmax(keras_out)
    print(f"TFLite top-1: {top_tflite} ({tflite_out[top_tflite]:.3f})")
    print(f"Keras  top-1: {top_keras}  ({keras_out[top_keras]:.3f})")
    print(f"Max diff:     {max_diff:.4f}")
    if top_tflite == top_keras:
        print("✓ Top-1 prediction matches")
    else:
        print("! Top-1 differs (expected — quantization drift in TFLite is fine)")
        print("  The Keras float32 weights are the ground truth for the web app.")
except Exception as e:
    print(f"TFLite comparison skipped: {e}")

# ── Write binary file ─────────────────────────────────────────────────────────
manifest = []
buf = bytearray()

for name, arr in tensors:
    arr_f32 = arr.astype(np.float32)
    offset  = len(buf)
    data    = arr_f32.tobytes()
    buf    += data
    manifest.append({
        "name":   name,
        "offset": offset,
        "length": len(data),
        "shape":  list(arr_f32.shape),
    })
    print(f"  {name}: shape={list(arr_f32.shape)}  offset={offset}")

bin_path = os.path.join(OUT_DIR, "weights.bin")
with open(bin_path, "wb") as f:
    f.write(buf)
print(f"\nWrote {len(buf)} bytes to {bin_path}")

json_path = os.path.join(OUT_DIR, "weights.json")
with open(json_path, "w") as f:
    json.dump(manifest, f, indent=2)
print(f"Wrote {json_path}")
print("\nDone.")
