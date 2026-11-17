"""
All settings in one place, read from environment variables (12-factor style): the same image
runs on your laptop, in CI and in production, only the environment changes.
Locally they come from the .env file in d:\\pythoncode; in Docker from docker-compose's env_file.
"""
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent


def _bool(name, default="0"):
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes")


@dataclass(frozen=True)
class Settings:
    # models: a big one for hard questions, a cheap one for easy ones (see router in llm.py)
    model_big: str = os.getenv("MODEL_BIG", "openai/gpt-oss-120b")
    model_small: str = os.getenv("MODEL_SMALL", "openai/gpt-oss-20b")
    # USD per 1M tokens (input, output). Groq prices as of Oct 2026, check groq.com/pricing.
    prices: dict = field(default_factory=lambda: {
        "openai/gpt-oss-120b": (0.15, 0.60),
        "openai/gpt-oss-20b": (0.075, 0.30),
        "fake": (0.0, 0.0),
    })
    llm_timeout_s: float = float(os.getenv("LLM_TIMEOUT_S", "30"))
    max_answer_tokens: int = int(os.getenv("MAX_ANSWER_TOKENS", "800"))

    # FAKE_LLM=1: deterministic offline answers, for tests, CI and load tests without spending quota
    fake_llm: bool = _bool("FAKE_LLM")
    # FAKE_EMBED=1: hashing embeddings instead of the ONNX model (tests / CI, no download)
    fake_embed: bool = _bool("FAKE_EMBED")

    # security
    api_keys: frozenset = frozenset(k.strip() for k in os.getenv("API_KEYS", "dev-key-123").split(",") if k.strip())
    rate_limit_per_minute: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "30"))
    max_question_chars: int = int(os.getenv("MAX_QUESTION_CHARS", "1000"))
    guardrails: bool = _bool("GUARDRAILS", "1")

    # cache
    cache_ttl_s: int = int(os.getenv("CACHE_TTL_S", "3600"))
    semantic_cache_threshold: float = float(os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.95"))

    # data & logs
    docs_dir: Path = Path(os.getenv("DOCS_DIR", ROOT / "data" / "docs"))
    log_file: Path = Path(os.getenv("LOG_FILE", ROOT / "logs" / "requests.jsonl"))


settings = Settings()
