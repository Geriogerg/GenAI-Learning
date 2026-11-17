"""
Guardrails: checks around the LLM, in code, that don't rely on the model behaving.

  input   length limit, prompt-injection patterns, PII redaction (before it reaches the LLM or the logs)
  context retrieved documents are UNTRUSTED: wrapped in tags, scanned for instructions
  output  system-prompt leak (canary), links to non-allowlisted domains (data exfiltration),
          citations that point to nothing (ungrounded answer)

Important: regex guardrails are a first layer, easy to bypass with a rephrasing
(05_red_team_eval.py shows it). Real systems add an ML classifier (e.g. Llama Prompt Guard,
Llama Guard) and above all STRUCTURAL limits: what the model can do (tools, permissions)
matters more than what it can be tricked into saying. See 04_prompt_injection.py.
"""
import re
import secrets
from dataclasses import dataclass, field

INJECTION_PATTERNS = [
    r"ignore (all |any |the )?(previous|prior|above|earlier) (instructions|rules|messages)",
    r"disregard (all |the )?(previous|prior|above|system)",
    r"forget (everything|all|your) (instructions|rules|above)",
    r"(reveal|show|print|repeat|output) (me )?(your|the) (system )?(prompt|instructions)",
    r"you are now (a|an|in) ",
    r"\b(developer|dan|jailbreak|god) mode\b",
    r"new (system )?instructions?:",
    r"<\s*/?\s*(system|instructions?)\s*>",
    r"oublie (toutes )?(tes|les) (instructions|consignes)",
    r"ignore (tes|les) (instructions|consignes)",
]
INJECTION_RE = re.compile("|".join(INJECTION_PATTERNS), re.I)

PII_PATTERNS = {
    "EMAIL": r"[\w.+-]+@[\w-]+\.[\w.-]+",
    "PHONE": r"(?<!\d)(?:\+33\s?|0)[1-9](?:[\s.-]?\d{2}){4}(?!\d)",       # French numbers
    "IBAN": r"\b[A-Z]{2}\d{2}(?:\s?[A-Z0-9]{4}){3,7}\b",
}

ALLOWED_LINK_DOMAINS = {"intranet.example.com"}  # the only place answers may link to
LINK_RE = re.compile(r"https?://([^/\s)\"'>]+)[^\s)\"'>]*", re.I)

# A random marker placed in the system prompt. If it ever shows up in an answer, the model
# is leaking its instructions. Cheap and very reliable.
CANARY = f"canary-{secrets.token_hex(6)}"


@dataclass
class Check:
    allowed: bool = True
    flags: list = field(default_factory=list)
    text: str = ""


def redact_pii(text):
    found = []
    for label, pattern in PII_PATTERNS.items():
        text, n = re.subn(pattern, f"[{label}]", text)
        if n:
            found.append(f"pii:{label.lower()}")
    return text, found


def check_input(question, max_chars=1000):
    if len(question) > max_chars:
        return Check(False, ["too_long"], question)
    if INJECTION_RE.search(question):
        return Check(False, ["prompt_injection"], question)
    text, pii = redact_pii(question)
    return Check(True, pii, text)


def scan_document(text):
    """Flags retrieved chunks that contain instructions aimed at the model (indirect injection).
    Documents written for humans rarely address "AI assistants" or ask to hide things from the user."""
    flat = " ".join(text.split())  # PDFs and markdown break sentences across lines
    return bool(INJECTION_RE.search(flat)) or any(re.search(p, flat, re.I) for p in DOC_INJECTION_PATTERNS)


DOC_INJECTION_PATTERNS = [
    r"\b(note|message|instructions?|attention)\s+(for|to)\s+(ai|llms?|(ai |language )?models?|assistants?|chatbots?|agents?)\b",
    r"\b(ai|llm|assistant|chatbot|agent)s?\b.{0,120}\byou (must|should|have to|need to)\b",
    r"\b(do not|don't|never) (mention|tell|reveal|show) (this|it|that)?\s?(step )?(to )?the user\b",
    r"\buse your (email|e-mail|browser|web|search|file|code)\s?(tool|function|access)\b",
]


def wrap_documents(hits):
    """Put retrieved text inside clearly delimited tags, numbered for citations.
    Delimiting helps the model separate data from instructions; it does NOT guarantee it."""
    return "\n\n".join(
        f'<document id="{i}" source="{h["source"]}">\n{h["text"]}\n</document>' for i, h in enumerate(hits, 1))


def check_output(answer, n_sources):
    flags = []
    if CANARY in answer:
        return Check(False, ["system_prompt_leak"], "")
    for domain in LINK_RE.findall(answer):
        if domain.lower() not in ALLOWED_LINK_DOMAINS:
            flags.append(f"external_link:{domain.lower()}")
    if flags:
        # remove the links instead of blocking: a markdown image to an attacker's server is how
        # data gets exfiltrated (the browser fetches it, with the data in the URL)
        answer = LINK_RE.sub("[link removed]", answer)
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
    if any(n < 1 or n > n_sources for n in cited):
        flags.append("invalid_citation")
    if n_sources and not cited and "couldn't find" not in answer.lower():
        flags.append("no_citation")
    return Check(True, flags, answer)
