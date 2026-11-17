"""
Shared helper for pillar 1: one Groq client, one chat() function with the model-specific
settings and a retry on rate limits. Script 01 calls the API directly on purpose, to see the
raw request; the others use this.
"""
import json
import os
import time

from dotenv import load_dotenv
from groq import Groq, RateLimitError

load_dotenv()  # finds the .env in d:\pythoncode

# llama-3.3-70b-versatile (used in the first version of these scripts) was retired by Groq
# in August 2026. gpt-oss-120b is the recommended replacement. Override with GROQ_MODEL in .env.
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def model_options(model):
    """Settings some models need. Kept in one place so the scripts stay readable."""
    if "gpt-oss" in model:
        return {"reasoning_effort": "low"}       # reasoning model, its thinking counts in max_tokens
    if "qwen" in model:
        return {"reasoning_format": "parsed"}    # otherwise <think>...</think> lands in the answer
    return {}


def chat(messages, tools=None, json_mode=False, max_tokens=1500, temperature=0.3, model=None, retries=3):
    """One chat completion. Returns the message (with .content and .tool_calls)."""
    model = model or MODEL
    kwargs = dict(model=model, messages=messages, max_tokens=max_tokens, temperature=temperature,
                  **model_options(model))
    if tools:
        kwargs.update(tools=tools, tool_choice="auto")
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    for attempt in range(retries):
        try:
            return client.chat.completions.create(**kwargs).choices[0].message
        except RateLimitError:
            if attempt == retries - 1:
                raise
            print(f"  (rate limited, retrying in {10 * (attempt + 1)}s)")
            time.sleep(10 * (attempt + 1))


def ask(system_prompt, user_prompt, **kwargs):
    msg = chat([{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], **kwargs)
    return msg.content or ""


def ask_json(system_prompt, user_prompt, **kwargs):
    return json.loads(ask(system_prompt, user_prompt, json_mode=True, **kwargs))


def assistant_message(msg):
    """SDK message -> plain dict we can send back in the next request (tool calling)."""
    out = {"role": "assistant", "content": msg.content or ""}
    if msg.tool_calls:
        out["tool_calls"] = [{"id": tc.id, "type": "function",
                              "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                             for tc in msg.tool_calls]
    return out
