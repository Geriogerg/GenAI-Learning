"""
Two ways to call a model, same interface, so the scripts can compare them fairly:
    groq_chat()    big model in the cloud (the "teacher" / baseline)
    ollama_chat()  small open model running on your own machine

Both return (text, seconds).
"""
import os
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
LOCAL_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:0.8b")


def groq_chat(messages, model=None, json_mode=False, max_tokens=1000, temperature=0.0):
    from groq import Groq, RateLimitError

    model = model or GROQ_MODEL
    kwargs = dict(model=model, messages=messages, max_tokens=max_tokens, temperature=temperature)
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    if "gpt-oss" in model:
        kwargs["reasoning_effort"] = "low"
    if "qwen" in model:
        kwargs["reasoning_format"] = "parsed"

    client = Groq()
    for attempt in range(4):
        try:
            start = time.perf_counter()
            resp = client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content or "", time.perf_counter() - start
        except RateLimitError:
            if attempt == 3:
                raise
            time.sleep(15 * (attempt + 1))


def ollama_chat(messages, model=None, schema=None, temperature=0.0, images=None):
    """
    schema: a JSON schema -> Ollama constrains the generation to it ("structured outputs").
            The model literally cannot produce invalid JSON or an unknown enum value.
    images: list of image paths, attached to the last user message (vision models only).
    """
    import ollama

    if images:
        messages = [*messages[:-1], {**messages[-1], "images": [str(p) for p in images]}]
    start = time.perf_counter()
    resp = ollama.chat(model=model or LOCAL_MODEL, messages=messages, format=schema, think=False,
                       options={"temperature": temperature})
    return resp.message.content or "", time.perf_counter() - start


def check_ollama(model=None):
    """Friendly error instead of a stack trace when Ollama isn't running / the model isn't pulled."""
    import ollama

    model = model or LOCAL_MODEL
    try:
        names = [m.model for m in ollama.list().models]
    except Exception:
        raise SystemExit("Ollama isn't running. Install it from ollama.com, start it, and retry.")
    if not any(n == model or n.split(":")[0] == model for n in names):
        raise SystemExit(f"Model {model} not found locally. Run:  ollama pull {model}")
