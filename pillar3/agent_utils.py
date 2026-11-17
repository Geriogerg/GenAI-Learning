"""
Shared agent machinery for pillar 3.

  Tool / @tool  -> turns a plain Python function into something an LLM can call:
                   JSON schema from the type hints, argument validation, errors as text
  chat()        -> one Groq call with retries
  Agent         -> the ReAct loop from 01_agent_loop.py, cleaned up so the other scripts
                   can reuse it (step limit, approvals, loop detection, trace, token count)
"""
import inspect
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from pydantic import ValidationError, create_model

load_dotenv()

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
PILLAR2_DIR = ROOT.parent / "pillar2"

MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
# gpt-oss on Groq doesn't do parallel tool calls, qwen does (used in 01 for the comparison)
PARALLEL_MODEL = os.getenv("GROQ_PARALLEL_MODEL", "qwen/qwen3.6-27b")

MAX_TOOL_OUTPUT = 4000  # chars. Tool outputs go back into the context at every step, keep them short


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

class Tool:
    """
    Wraps a function so that:
      - its JSON schema comes from the signature (type hints, defaults, Annotated descriptions)
      - its description comes from the docstring
      - bad arguments or exceptions come back as an error message the model can read and fix,
        instead of crashing the agent
    """

    def __init__(self, fn, requires_approval=False):
        self.fn = fn
        self.name = fn.__name__
        self.description = inspect.getdoc(fn) or ""
        self.requires_approval = requires_approval

        fields = {}
        for pname, p in inspect.signature(fn).parameters.items():
            annotation = p.annotation if p.annotation is not inspect.Parameter.empty else str
            default = p.default if p.default is not inspect.Parameter.empty else ...
            fields[pname] = (annotation, default)
        self.args_model = create_model(f"{self.name}_args", **fields)

    def schema(self):
        params = self.args_model.model_json_schema()
        params.pop("title", None)
        for prop in params.get("properties", {}).values():
            prop.pop("title", None)  # pydantic adds titles, they're just noise in the prompt
        return {"type": "function", "function": {"name": self.name, "description": self.description,
                                                 "parameters": params}}

    def __call__(self, raw_args):
        try:
            args = json.loads(raw_args or "{}") if isinstance(raw_args, str) else (raw_args or {})
        except json.JSONDecodeError:
            return f"Error: arguments are not valid JSON: {raw_args!r}"
        try:
            validated = self.args_model.model_validate(args)
        except ValidationError as e:
            problems = "; ".join(f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors())
            return f"Error: invalid arguments for {self.name}: {problems}"
        try:
            result = self.fn(**dict(validated))
        except Exception as e:
            return f"Error: {type(e).__name__}: {e}"
        return result if isinstance(result, str) else json.dumps(result, default=str)


def tool(fn=None, *, requires_approval=False):
    """Decorator: @tool or @tool(requires_approval=True)"""
    if fn is None:
        return lambda f: Tool(f, requires_approval=requires_approval)
    return Tool(fn)


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

_client = None
TOKENS = {"total": 0}


def get_client():
    global _client
    if _client is None:
        from groq import Groq
        if not os.getenv("GROQ_API_KEY"):
            raise RuntimeError("GROQ_API_KEY not found, check your .env file")
        _client = Groq()
    return _client


def chat(messages, tools=None, model=None, max_tokens=2000, temperature=0.2, json_mode=False,
         parallel_tool_calls=None, retries=3):
    """One chat completion. Returns the message object (with .content and .tool_calls)."""
    from groq import BadRequestError, RateLimitError

    model = model or MODEL
    kwargs = dict(model=model, messages=messages, max_tokens=max_tokens, temperature=temperature)
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
        if parallel_tool_calls is not None:
            kwargs["parallel_tool_calls"] = parallel_tool_calls
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    if "gpt-oss" in model:
        kwargs["reasoning_effort"] = "low"

    for attempt in range(retries):
        try:
            resp = get_client().chat.completions.create(**kwargs)
            if resp.usage:
                TOKENS["total"] += resp.usage.total_tokens
            return resp.choices[0].message
        except RateLimitError:
            if attempt == retries - 1:
                raise
            wait = 15 * (attempt + 1)
            print(f"    (rate limited, waiting {wait}s)")
            time.sleep(wait)
        except BadRequestError as e:
            # Groq returns 400 "tool_use_failed" when the model writes a malformed tool call.
            # It's random, a retry usually works.
            if "tool_use_failed" in str(e) and attempt < retries - 1:
                print("    (malformed tool call from the model, retrying)")
                continue
            raise


def assistant_message(msg):
    """Convert the SDK message to a plain dict we can send back (drops reasoning fields etc.)."""
    out = {"role": "assistant", "content": msg.content or ""}
    if msg.tool_calls:
        out["tool_calls"] = [
            {"id": tc.id, "type": "function",
             "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in msg.tool_calls
        ]
    return out


def ask_user_approval(name, args):
    print(f"\n  >>> The agent wants to run {name}({args})")
    return input("  >>> Approve? [y/N] ").strip().lower() in ("y", "yes")


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

@dataclass
class AgentResult:
    answer: str
    steps: int
    tokens: int
    trace: list = field(default_factory=list)      # one dict per tool call
    messages: list = field(default_factory=list)   # full conversation, useful for debugging
    hit_step_limit: bool = False

    @property
    def tool_errors(self):
        return sum(1 for t in self.trace if t["result"].startswith("Error"))


class Agent:
    def __init__(self, tools, system_prompt, model=None, max_steps=8, verbose=True,
                 approve=ask_user_approval, parallel_tool_calls=None):
        self.tools = {t.name: t for t in tools}
        self.schemas = [t.schema() for t in tools]
        self.system_prompt = system_prompt
        self.model = model
        self.max_steps = max_steps
        self.verbose = verbose
        self.approve = approve
        self.parallel_tool_calls = parallel_tool_calls

    def log(self, text):
        if self.verbose:
            print(text)

    def run_tool(self, call):
        name, raw_args = call.function.name, call.function.arguments
        tool_ = self.tools.get(name)
        if tool_ is None:
            return f"Error: unknown tool '{name}'. Available tools: {list(self.tools)}"
        if tool_.requires_approval and not self.approve(name, raw_args):
            return "The user refused this action. Don't retry it, explain what you would have done."
        return tool_(raw_args)

    def run(self, task, history=None):
        tokens_before = TOKENS["total"]
        messages = [{"role": "system", "content": self.system_prompt}, *(history or []),
                    {"role": "user", "content": task}]
        trace, seen_calls = [], {}
        answer, hit_limit, step = None, False, 0

        for step in range(1, self.max_steps + 1):
            msg = chat(messages, tools=self.schemas, model=self.model,
                       parallel_tool_calls=self.parallel_tool_calls)
            thinking = getattr(msg, "reasoning", None)
            if thinking:
                self.log(f"  [{step}] thinking: {' '.join(thinking.split())[:200]}")
            if msg.content and msg.tool_calls:
                self.log(f"  [{step}] says: {msg.content[:200]}")

            if not msg.tool_calls:
                answer = msg.content or ""
                break

            messages.append(assistant_message(msg))
            for call in msg.tool_calls:
                key = (call.function.name, call.function.arguments)
                seen_calls[key] = seen_calls.get(key, 0) + 1

                result = self.run_tool(call)
                if seen_calls[key] >= 3:  # same call with the same args again and again
                    result += "\nNote: you already made this exact call several times. Try something else or answer."

                self.log(f"  [{step}] {call.function.name}({call.function.arguments})")
                self.log(f"       -> {result[:300]}")
                trace.append({"step": step, "tool": call.function.name, "args": call.function.arguments,
                              "result": result})
                messages.append({"role": "tool", "tool_call_id": call.id, "content": result[:MAX_TOOL_OUTPUT]})
        else:
            # out of steps: one last call without tools so we still get an answer
            hit_limit = True
            self.log(f"  step limit ({self.max_steps}) reached, asking for a final answer")
            messages.append({"role": "user", "content": "Step limit reached. Give your best final answer "
                                                        "with the information you have, and say what's missing."})
            answer = chat(messages, model=self.model).content or ""

        messages.append({"role": "assistant", "content": answer})
        return AgentResult(answer=answer, steps=step, tokens=TOKENS["total"] - tokens_before,
                           trace=trace, messages=messages, hit_step_limit=hit_limit)
