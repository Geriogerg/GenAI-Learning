"""
LLM calls the way a production service needs them:
  - timeout (a hung API call must not hang your service)
  - retries with backoff on rate limits / transient errors
  - fallback to the other model if one keeps failing
  - token usage and cost on every call
  - streaming
  - a router that sends easy questions to the cheap model
  - a fake mode (FAKE_LLM=1) for tests, CI and load tests
"""
import re
import time
from dataclasses import dataclass

from app.config import settings


@dataclass
class LLMResult:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float

    @property
    def cost_usd(self):
        return cost(self.model, self.prompt_tokens, self.completion_tokens)


def cost(model, prompt_tokens, completion_tokens):
    p_in, p_out = settings.prices.get(model, (0.0, 0.0))
    return (prompt_tokens * p_in + completion_tokens * p_out) / 1e6


def estimate_tokens(text):
    return max(1, len(text) // 4)  # rough, good enough for budgets when the API gives no usage


# ---------------------------------------------------------------------------
# router
# ---------------------------------------------------------------------------

HARD_HINTS = re.compile(r"\b(why|compare|explain|difference|should|analy[sz]e|plan|pourquoi|comparer|expliquer)\b", re.I)


def route(question):
    """Cheap heuristic router. Real systems often use a small classifier or the small LLM itself.
    Measure it (lesson 03): a router that sends hard questions to the small model saves money
    and loses quality."""
    hard = len(question) > 160 or bool(HARD_HINTS.search(question)) or question.count("?") > 1
    return settings.model_big if hard else settings.model_small


# ---------------------------------------------------------------------------
# calls
# ---------------------------------------------------------------------------

_client = None


def _groq():
    global _client
    if _client is None:
        from groq import Groq
        _client = Groq(timeout=settings.llm_timeout_s, max_retries=0)  # we do our own retries
    return _client


def _kwargs(model, messages, max_tokens):
    kw = dict(model=model, messages=messages, max_tokens=max_tokens, temperature=0.2)
    if "gpt-oss" in model:
        kw["reasoning_effort"] = "low"
    if "qwen" in model:
        kw["reasoning_format"] = "parsed"
    return kw


def _fake_answer(messages):
    """Offline stand-in: quotes the first document of the context, cites it, takes ~0.3 s."""
    time.sleep(0.3)
    user = messages[-1]["content"]
    m = re.search(r'<document id="1"[^>]*>\s*(.+?)</document>', user, re.S)
    if not m:
        return "I couldn't find this in the documents."
    first = " ".join(m.group(1).split())[:200]
    return f"According to the documentation: {first} [1]"


def complete(messages, model, max_tokens=None, retries=3):
    """Non-streaming call with retries and fallback. Raises if everything failed."""
    max_tokens = max_tokens or settings.max_answer_tokens
    if settings.fake_llm:
        start = time.perf_counter()
        text = _fake_answer(messages)
        return LLMResult(text, "fake", sum(estimate_tokens(m["content"]) for m in messages),
                         estimate_tokens(text), time.perf_counter() - start)

    from groq import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

    candidates = [model] + [m for m in (settings.model_big, settings.model_small) if m != model]
    last_error = None
    for candidate in candidates:  # fallback: try the other model if this one keeps failing
        for attempt in range(retries):
            try:
                start = time.perf_counter()
                resp = _groq().chat.completions.create(**_kwargs(candidate, messages, max_tokens))
                usage = resp.usage
                return LLMResult(resp.choices[0].message.content or "", candidate,
                                 usage.prompt_tokens if usage else 0, usage.completion_tokens if usage else 0,
                                 time.perf_counter() - start)
            except (RateLimitError, APITimeoutError, APIConnectionError) as e:
                last_error = e
                time.sleep(min(2 ** attempt, 8))  # exponential backoff
            except APIStatusError as e:
                last_error = e
                if e.status_code < 500:  # 4xx (bad request, model gone...): retrying won't help
                    break
                time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"LLM unavailable after retries and fallback: {last_error}")


def stream(messages, model, max_tokens=None):
    """Yields text chunks as they're generated. First token arrives in ~0.2-1 s instead of
    waiting for the whole answer: that's what makes chat UIs feel fast."""
    max_tokens = max_tokens or settings.max_answer_tokens
    if settings.fake_llm:
        for word in _fake_answer(messages).split(" "):
            time.sleep(0.02)
            yield word + " "
        return
    resp = _groq().chat.completions.create(stream=True, **_kwargs(model, messages, max_tokens))
    for chunk in resp:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
