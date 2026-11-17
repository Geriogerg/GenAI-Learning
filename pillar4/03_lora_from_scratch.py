"""
Step 3 - LoRA from scratch (PyTorch, CPU, ~1 minute).

LoRA (Low-Rank Adaptation): freeze the pretrained weight W (d_out x d_in) and learn a small
update with two thin matrices:
        W' = W + (alpha / r) * B @ A        A: r x d_in, B: d_out x r, r << d
Only A and B are trained. With d = 4096 and r = 16 that's 0.8% of the parameters of the layer.
B starts at zero, so at step 0 the model is exactly the pretrained one.

Scenario (you lived it in pillar 2's postmortem): a classifier is trained on images from
camera A. The camera gets replaced, the input distribution shifts, accuracy drops. We have
only a few hundred labeled samples from the new camera. Options:
    full fine-tuning     update all weights
    head only            retrain the last layer
    LoRA                 low-rank adapters on every linear layer, base frozen

The model is an MLP on synthetic features to keep it fast. It's the same math in a
transformer: LoRA goes on the attention and MLP projection layers.
"""
import copy
import math

import torch
import torch.nn as nn

torch.manual_seed(0)
D, CLASSES, HIDDEN = 64, 10, 256


# ---------------------------------------------------------------------------
# LoRA layer
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=4, alpha=8):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False                     # pretrained weights stay frozen
        self.scale = alpha / r
        self.A = nn.Parameter(torch.randn(r, base.in_features) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))  # zero -> starts as the base model
        self.enabled = True

    def forward(self, x):
        out = self.base(x)
        if self.enabled:
            out = out + (x @ self.A.T @ self.B.T) * self.scale
        return out

    def merged(self):
        """Fold the adapter into the weight: same outputs, zero extra cost at inference."""
        layer = copy.deepcopy(self.base)
        layer.weight.data += (self.B @ self.A) * self.scale
        return layer


def add_lora(model, r=4, alpha=8):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            setattr(model, name, LoRALinear(module, r, alpha))
        else:
            add_lora(module, r, alpha)
    return model


# ---------------------------------------------------------------------------
# data: camera A (lots of data) and camera B (shifted, few labels)
# ---------------------------------------------------------------------------

centers = torch.randn(CLASSES, D) * 0.5
mix = torch.eye(D) + 0.9 * torch.randn(D, D) / math.sqrt(D)   # the new camera's "distortion"
offset = 1.5 * torch.randn(D)


def make_data(n, camera):
    y = torch.randint(0, CLASSES, (n,))
    x = centers[y] + torch.randn(n, D)
    if camera == "B":
        x = x @ mix.T + offset
    return x, y


xa_train, ya_train = make_data(20000, "A")
xa_test, ya_test = make_data(4000, "A")
xb_train, yb_train = make_data(300, "B")      # only 300 labeled images from the new camera
xb_test, yb_test = make_data(4000, "B")


def build_model():
    return nn.Sequential(nn.Linear(D, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
                         nn.Linear(HIDDEN, CLASSES))


def train(model, x, y, epochs, lr):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    for _ in range(epochs):
        for idx in torch.randperm(len(x)).split(64):
            loss = nn.functional.cross_entropy(model(x[idx]), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


@torch.no_grad()
def accuracy(model, x, y):
    model.eval()
    acc = (model(x).argmax(1) == y).float().mean().item()
    model.train()
    return acc


def n_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# experiment
# ---------------------------------------------------------------------------

print("Pretraining on camera A ...")
base = train(build_model(), xa_train, ya_train, epochs=8, lr=1e-3)
total = sum(p.numel() for p in base.parameters())
print(f"  base model: {total:,} params | camera A acc {accuracy(base, xa_test, ya_test):.1%} | "
      f"camera B acc {accuracy(base, xb_test, yb_test):.1%}  <- the drop after the camera change\n")

results = []

full = train(copy.deepcopy(base), xb_train, yb_train, epochs=60, lr=1e-3)
results.append(("full fine-tuning", n_trainable(full), full))

head = copy.deepcopy(base)
for p in head.parameters():
    p.requires_grad = False
for p in head[-1].parameters():
    p.requires_grad = True
head = train(head, xb_train, yb_train, epochs=60, lr=1e-3)
results.append(("head only", n_trainable(head), head))

lora = add_lora(copy.deepcopy(base), r=4, alpha=8)
lora = train(lora, xb_train, yb_train, epochs=60, lr=3e-3)
results.append(("LoRA r=4", n_trainable(lora), lora))

print(f"{'method':<18}{'trainable':>12}{'% of model':>12}{'camera B':>10}{'camera A':>10}")
for name, n, model in results:
    print(f"{name:<18}{n:>12,}{n / total:>11.1%}{accuracy(model, xb_test, yb_test):>10.1%}"
          f"{accuracy(model, xa_test, ya_test):>10.1%}")

# --- LoRA superpowers -----------------------------------------------------------
lora_layers = [m for m in lora.modules() if isinstance(m, LoRALinear)]

for m in lora_layers:
    m.enabled = False
print(f"\nLoRA adapter switched OFF -> camera A acc {accuracy(lora, xa_test, ya_test):.1%} "
      f"(the original model is untouched, one base + one small adapter per camera)")
for m in lora_layers:
    m.enabled = True

merged = copy.deepcopy(lora)
for i, layer in enumerate(merged):
    if isinstance(layer, LoRALinear):
        merged[i] = layer.merged()
with torch.no_grad():
    diff = (merged(xb_test) - lora(xb_test)).abs().max().item()
print(f"Merged adapter into the weights -> max output difference {diff:.1e} (same model, no extra layers)")

adapter_kb = sum(m.A.numel() + m.B.numel() for m in lora_layers) * 4 / 1024
print(f"Adapter file size: {adapter_kb:.0f} KB vs {total * 4 / 1024:.0f} KB for the full model")

# Takeaways (typical run: camera B 65% -> ~88-89% after adaptation, check your own numbers):
#  - LoRA trains ~5% of the parameters here and lands within a point or two of full fine-tuning.
#    On LLMs (0.1-1% of the params) that's the difference between "needs 8 GPUs" and "free Colab".
#  - full fine-tuning drifts away from the old task (camera A accuracy goes down: forgetting).
#    LoRA keeps the base intact: switch the adapter off and camera A is exactly back.
#  - one base model + many small adapters (one per customer / task / camera) is a common
#    production pattern. Or merge it for zero inference overhead.
#  - r (rank) and alpha are the main knobs. Try r=1 and r=32, and fewer labeled samples (50).
#  - "head only" is the classic CV transfer-learning move: cheapest, but it can't fix a shift
#    that happens in the early features, like a camera change.
