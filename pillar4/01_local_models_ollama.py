"""
Step 1 - running an open model on your own machine with Ollama.

Until now every model ran on Groq's servers. Open-weight models (Qwen, Llama, Gemma,
Mistral...) can run on your laptop: no API key, no rate limit, no data leaving the machine,
no per-token cost. The price: small models are less capable, and speed depends on your hardware.

Ollama = a local model server. It downloads models already quantized (GGUF format, see
step 2) and serves them on http://localhost:11434.

Setup (once):
    1. install Ollama from https://ollama.com (Windows installer)
    2. ollama pull qwen3.5:0.8b        (~1 GB, runs fine on CPU)
    3. optional: ollama pull qwen3.5:2b  (better, slower)

This script compares the local model with the big Groq model on the same prompts:
speed, quality, and structured output.
"""
import json

from llm_clients import GROQ_MODEL, LOCAL_MODEL, check_ollama, groq_chat, ollama_chat

check_ollama()

PROMPTS = [
    "Explain in 3 sentences what LoRA is in deep learning.",
    "A model has 2 billion parameters stored in 16-bit. How many GB of memory do the weights take? "
    "Show the calculation in one line.",
    "Translate to French: 'The camera on line 3 was replaced and the images are darker.'",
]

for prompt in PROMPTS:
    messages = [{"role": "user", "content": prompt}]
    print("=" * 80)
    print(f"PROMPT: {prompt}\n")
    for name, fn in [(f"local {LOCAL_MODEL}", ollama_chat), (f"groq  {GROQ_MODEL}", groq_chat)]:
        text, secs = fn(messages)
        print(f"--- {name}  ({secs:.1f}s)\n{text.strip()}\n")

# --- structured output: free JSON vs schema-constrained ------------------------
print("=" * 80)
note = "L3 peinture nuit: 4 pcs rayées assez profond, cc chef"
ask = [{"role": "user", "content": f"Extract line, station, shift, defect and quantity from this note as JSON: {note}"}]

text, _ = ollama_chat(ask)
print(f"Local model, free JSON:\n{text.strip()}\n")

schema = {"type": "object", "required": ["line", "station", "shift", "defect", "quantity"],
          "properties": {"line": {"type": "string"}, "station": {"type": "string"},
                         "shift": {"type": "string", "enum": ["morning", "afternoon", "night"]},
                         "defect": {"type": "string"}, "quantity": {"type": "integer"}}}
text, _ = ollama_chat(ask, schema=schema)
try:
    print(f"Local model, constrained to a JSON schema:\n{json.dumps(json.loads(text), indent=2, ensure_ascii=False)}")
except json.JSONDecodeError:  # shouldn't happen with a schema, unless the output hit the length limit
    print(f"Local model, constrained to a JSON schema (unparseable!):\n{text}")

# How the constraint works: at each step, tokens that would break the schema get probability 0
# (constrained decoding). Valid JSON and valid enum values are GUARANTEED, which is very useful
# with small models. It doesn't make the content right though: wrong values stay possible.

# Things to note on your machine:
#  - first call is slow (model loading into RAM), the next ones are faster
#  - tokens/second on CPU: run `ollama run qwen3.5:0.8b --verbose` and ask something
#  - Ollama also exposes an OpenAI-compatible API (http://localhost:11434/v1), so code written
#    for OpenAI/Groq works with a local model by changing the base_url
