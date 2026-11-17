"""
Step 2 - quantization, by hand (numpy only, no download).

Quantization = storing weights with fewer bits (16 -> 8 -> 4) to cut memory and speed up
inference. It's why a 9B model fits on a laptop, and what the "Q4_K_M" in model names means.
It's also the Q in QLoRA (train LoRA adapters on top of a 4-bit frozen model).

Three parts:
  1. memory math: what fits where, for inference and for training
  2. quantize a weight matrix yourself and measure the error
  3. why outliers matter, and why block-wise quantization exists
"""
import numpy as np

rng = np.random.default_rng(0)

# ---------------------------------------------------------------------------
# 1. memory math
# ---------------------------------------------------------------------------

print("=" * 80 + "\n1. Memory for the WEIGHTS only (inference)\n")
formats = {"fp32": 4, "bf16/fp16": 2, "int8": 1, "4-bit": 0.5}
print(f"{'model':<14}" + "".join(f"{f:>12}" for f in formats))
for name, params in [("0.8B", 0.8e9), ("2B", 2e9), ("9B", 9e9), ("70B", 70e9), ("120B", 120e9)]:
    print(f"{name:<14}" + "".join(f"{params * b / 1e9:>10.1f}GB" for b in formats.values()))
print("\n+ the KV cache (grows with context length) and some overhead. A free Colab T4 has 15 GB,"
      "\na laptop GPU often 6-8 GB, 16-32 GB of RAM for CPU inference.")

print("\n" + "=" * 80 + "\n2. Memory for TRAINING (rough, without activations)\n")
# full fine-tuning with Adam in mixed precision: bf16 weights (2) + bf16 grads (2)
#   + fp32 master weights (4) + Adam m and v in fp32 (8) = ~16 bytes per parameter
# LoRA: frozen bf16 weights (2 bytes) + a small trainable adapter (~1% of params, 16 bytes each)
# QLoRA: same but the frozen weights are 4-bit (0.5 bytes)
print(f"{'model':<8}{'full FT':>12}{'LoRA bf16':>12}{'QLoRA 4bit':>12}")
for name, p in [("0.8B", 0.8e9), ("2B", 2e9), ("4B", 4e9), ("9B", 9e9)]:
    adapter = 0.01 * p * 16
    print(f"{name:<8}{p * 16 / 1e9:>10.1f}GB{(p * 2 + adapter) / 1e9:>10.1f}GB{(p * 0.5 + adapter) / 1e9:>10.1f}GB")
print("\n-> on a free T4 (15 GB): full fine-tuning is out of reach past ~0.8B, LoRA works up to ~4B,"
      "\n   QLoRA up to ~9-14B. That's why LoRA/QLoRA are the default for everyone without a cluster.")

# ---------------------------------------------------------------------------
# 3. quantizing a weight matrix
# ---------------------------------------------------------------------------

print("\n" + "=" * 80 + "\n3. Quantize a 1024x1024 weight matrix\n")
W = rng.normal(0, 0.02, (1024, 1024)).astype(np.float32)  # typical init scale of LLM weights
# real LLM weights have a few large outliers, add some
outliers = rng.choice(W.size, 40, replace=False)
W.flat[outliers] = rng.normal(0, 0.5, 40)
X = rng.normal(0, 1, (64, 1024)).astype(np.float32)  # a batch of activations
Y = X @ W.T  # reference output


def quantize(w, bits, block=None):
    """Symmetric absmax quantization: scale = max|w| / qmax, q = round(w / scale).
    block=None -> one scale for the whole matrix, block=k -> one scale per k consecutive weights."""
    qmax = 2 ** (bits - 1) - 1
    flat = w.reshape(-1, block) if block else w.reshape(1, -1)
    scale = np.abs(flat).max(axis=1, keepdims=True) / qmax
    q = np.clip(np.round(flat / scale), -qmax - 1, qmax)          # the stored integers
    return (q * scale).reshape(w.shape), scale.size                 # dequantized weights, #scales


print(f"{'method':<28}{'bits/weight':>12}{'weight err':>12}{'output err':>12}")
for label, bits, block in [("int8, one scale", 8, None), ("int8, blocks of 64", 8, 64),
                           ("int4, one scale", 4, None), ("int4, blocks of 32", 4, 32)]:
    Wq, n_scales = quantize(W, bits, block)
    bpw = bits + n_scales * 16 / W.size  # each scale stored in 16-bit
    w_err = np.abs(Wq - W).mean() / np.abs(W).mean()
    y_err = np.abs(X @ Wq.T - Y).mean() / np.abs(Y).mean()
    print(f"{label:<28}{bpw:>12.2f}{w_err:>11.1%}{y_err:>11.1%}")

# ---------------------------------------------------------------------------
print("""
What the numbers say:
  - with ONE scale, a single outlier sets max|w| for everybody, so all the normal small
    weights get crushed into a few integer levels. int4 with one scale is unusable.
  - with one scale per BLOCK, an outlier only hurts its own block. Costs a bit of extra storage
    (the scales: 4.5 bits/weight instead of 4) and fixes most of the damage.
  - that's what GGUF formats do (Q4_0, Q4_K_M, Q8_0...: block-wise schemes, K-quants mix
    precisions per layer). QLoRA uses NF4, a 4-bit format whose levels follow a normal
    distribution, plus block-wise scales.

Why Unsloth discourages QLoRA for Qwen3.5: some architectures have weights/activations
that quantize badly (more outliers, sensitive layers). The 4-bit base adds error that the
LoRA adapter has to compensate during training. Rule of thumb: use 16-bit LoRA when it fits,
QLoRA when it doesn't, and always compare with a quick eval.

Inference quantization (Ollama, GGUF) and training quantization (QLoRA) are separate choices:
you can train in 16-bit and still deploy a Q4_K_M GGUF, which is what notebook 06 does.""")
