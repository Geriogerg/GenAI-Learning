"""
The fine-tuning task of pillar 4: turn messy operator notes into a strict JSON defect record.

    "L3 peinture nuit: 4 pcs rayées assez profond, cc chef"
    -> {"line": "L3", "station": "paint", "shift": "night", "defect_code": "SCR",
        "severity": "high", "quantity": 4, "action": "scrap"}

Why this task: it's the typical case where fine-tuning beats prompting.
  - fixed output format, high volume (cheap small model wanted)
  - notes mix French and English, abbreviations, typos
  - the "action" follows HOUSE RULES that no general model knows (see decide_action):
    a fine-tuned model learns them from examples, a prompted model only if you write them down

This module is shared by the scripts AND the Colab notebook (upload it next to the data).
"""
import json
import random
import re

LINES = ["L1", "L2", "L3", "L4", "L5"]
STATIONS = ["press", "paint", "assembly", "inspection", "packaging"]
SHIFTS = ["morning", "afternoon", "night"]
DEFECTS = ["SCR", "DNT", "STN", "MSC", "MSA", "CRK", "NONE"]
SEVERITIES = ["low", "medium", "high"]
ACTIONS = ["rework", "scrap", "monitor", "stop_line"]
FIELDS = ["line", "station", "shift", "defect_code", "severity", "quantity", "action"]

SYSTEM_PROMPT = """Convert the operator note into a JSON defect record with exactly these fields:
line (L1-L5), station (press|paint|assembly|inspection|packaging), shift (morning|afternoon|night),
defect_code (SCR|DNT|STN|MSC|MSA|CRK|NONE), severity (low|medium|high), quantity (integer),
action (rework|scrap|monitor|stop_line). Return only the JSON."""

# What a human expert would add to the prompt (used by 05 to show what prompting alone can do)
RULES_PROMPT = SYSTEM_PROMPT + """

Codes: SCR scratch, DNT dent, STN stain, MSC missing screw, MSA misalignment, CRK crack, NONE nothing to report.
If nothing to report: severity low, quantity 0, action monitor.
Action rules, apply in this order:
1. crack -> scrap (always)
2. severity high and quantity >= 10 -> stop_line
3. missing screw -> rework
4. severity high -> scrap
5. severity medium -> rework
6. severity low -> monitor"""

JSON_SCHEMA = {  # for constrained decoding (Ollama "format")
    "type": "object",
    "properties": {
        "line": {"type": "string", "enum": LINES},
        "station": {"type": "string", "enum": STATIONS},
        "shift": {"type": "string", "enum": SHIFTS},
        "defect_code": {"type": "string", "enum": DEFECTS},
        "severity": {"type": "string", "enum": SEVERITIES},
        "quantity": {"type": "integer"},
        "action": {"type": "string", "enum": ACTIONS},
    },
    "required": FIELDS,
}


def decide_action(defect, severity, quantity):
    """The house rules. Nobody wrote them in the notes, the model has to learn them."""
    if defect == "NONE":
        return "monitor"
    if defect == "CRK":
        return "scrap"
    if severity == "high" and quantity >= 10:
        return "stop_line"
    if defect == "MSC":
        return "rework"
    return {"high": "scrap", "medium": "rework", "low": "monitor"}[severity]


# ---------------------------------------------------------------------------
# Note generator
# ---------------------------------------------------------------------------

WORDS = {
    "fr": {
        "SCR": ["rayure", "rayures", "rayées", "griffure", "rayé"],
        "DNT": ["bosse", "enfoncement", "choc", "bosselé"],
        "STN": ["tache", "taches", "salissure", "trace d'huile"],
        "MSC": ["vis manquante", "manque une vis", "vis absente"],
        "MSA": ["décalage", "mal aligné", "désaxé"],
        "CRK": ["fissure", "fêlure", "craquelure"],
        "NONE": ["RAS", "rien à signaler", "tout ok", "R.A.S"],
        "low": ["léger", "superficiel", "à peine visible", "pas grave"],
        "medium": ["moyen", "visible", "bien visible"],
        "high": ["profond", "grave", "important", "assez profond"],
        "press": ["presse", "emboutissage"], "paint": ["peinture", "cabine peinture"],
        "assembly": ["montage", "assemblage"], "inspection": ["contrôle", "poste contrôle"],
        "packaging": ["emballage", "conditionnement"],
        "morning": ["matin", "équipe du matin", "6h-14h"], "afternoon": ["aprem", "après-midi", "14h-22h"],
        "night": ["nuit", "équipe de nuit", "22h-6h"],
        "line": ["{n}", "ligne {n}", "L{n}", "l{n}"],
        "parts": ["pcs", "pièces", "p.", "pièce(s)"],
        "nums": ["zéro", "une", "deux", "trois", "quatre", "cinq", "six", "sept", "huit", "neuf", "dix"],
        "dozen": "une douzaine de",
        "rework": ["à retoucher", "retouche", "reprise"], "scrap": ["rebut", "au rebut", "à jeter"],
        "monitor": ["on surveille", "à suivre"], "stop_line": ["arrêt ligne", "ligne arrêtée"],
        "filler": ["cc chef", "voir photo", "machine café HS", "merci", "photo dans le dossier", "signalé au N+1"],
    },
    "en": {
        "SCR": ["scratch", "scratches", "scratched", "scuff"],
        "DNT": ["dent", "dents", "dented", "ding"],
        "STN": ["stain", "stains", "oil mark", "smudge"],
        "MSC": ["missing screw", "screw missing", "no screw"],
        "MSA": ["misaligned", "offset", "misalignment"],
        "CRK": ["crack", "cracked", "hairline crack"],
        "NONE": ["nothing to report", "all good", "no issue", "NTR"],
        "low": ["light", "minor", "barely visible", "superficial"],
        "medium": ["moderate", "visible", "noticeable"],
        "high": ["deep", "severe", "bad", "major"],
        "press": ["press", "stamping"], "paint": ["paint", "paint booth"],
        "assembly": ["assembly", "assy"], "inspection": ["inspection", "QC station"],
        "packaging": ["packaging", "packing"],
        "morning": ["morning", "AM shift", "6-14"], "afternoon": ["afternoon", "PM shift", "14-22"],
        "night": ["night", "night shift", "22-6"],
        "line": ["line {n}", "L{n}", "ln {n}"],
        "parts": ["pcs", "parts", "units"],
        "nums": ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"],
        "dozen": "a dozen",
        "rework": ["to rework", "rework"], "scrap": ["scrapped", "to scrap"],
        "monitor": ["keep an eye on it", "monitoring"], "stop_line": ["line stopped", "stopping the line"],
        "filler": ["fyi", "see pic", "pls check", "thx", "photo attached", "told the supervisor"],
    },
}


def _typo(word, rng, p=0.04):
    if len(word) > 3 and rng.random() < p:
        i = rng.randrange(len(word) - 1)
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    return word


def _quantity_text(q, w, rng):
    if q == 12 and rng.random() < 0.5:
        return w["dozen"]
    if q <= 10 and rng.random() < 0.3:
        return w["nums"][q]
    return str(q)


def random_record(rng):
    defect = rng.choices(DEFECTS, weights=[3, 2, 2, 2, 1, 1, 2])[0]
    if defect == "NONE":
        severity, quantity = "low", 0
    else:
        severity = rng.choice(SEVERITIES)
        quantity = rng.choice([1, 1, 2, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 40])
    return {"line": rng.choice(LINES), "station": rng.choice(STATIONS), "shift": rng.choice(SHIFTS),
            "defect_code": defect, "severity": severity, "quantity": quantity,
            "action": decide_action(defect, severity, quantity)}


def render_note(rec, rng):
    lang = rng.choice(["fr", "en"])
    other = "en" if lang == "fr" else "fr"
    w = WORDS[lang]
    pick = lambda key: rng.choice(WORDS[other if rng.random() < 0.15 else lang][key])  # 15% code-switching

    line = rng.choice(w["line"]).format(n=rec["line"][1])
    station, shift = pick(rec["station"]), pick(rec["shift"])

    if rec["defect_code"] == "NONE":
        defect_chunk = pick("NONE")
    else:
        qty = _quantity_text(rec["quantity"], w, rng)
        word, sev, parts = pick(rec["defect_code"]), pick(rec["severity"]), rng.choice(w["parts"])
        templates = {
            "fr": [f"{qty} {parts} avec {word} {sev}", f"{word} {sev} sur {qty} {parts}", f"{qty}x {word} ({sev})"],
            "en": [f"{qty} {parts} with {sev} {word}", f"{sev} {word} on {qty} {parts}", f"{qty}x {word}, {sev}"],
        }
        defect_chunk = rng.choice(templates[lang])

    chunks = [line, station, shift]
    rng.shuffle(chunks)
    chunks.insert(rng.randint(1, 3), defect_chunk)
    if rec["defect_code"] != "NONE" and rng.random() < 0.3:  # the action is only sometimes written down
        chunks.append(pick(rec["action"]))
    if rng.random() < 0.35:
        chunks.append(pick("filler"))

    seps = [" ", ", ", " - ", " / ", ". ", ": "]
    note = chunks[0] + "".join(rng.choice(seps) + c for c in chunks[1:])
    note = " ".join(_typo(t, rng) for t in note.split(" "))
    return note.lower() if rng.random() < 0.3 else note


def make_examples(n, seed):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        rec = random_record(rng)
        out.append({"note": render_note(rec, rng), "label": rec})
    return out


def to_chat(example, system_prompt=SYSTEM_PROMPT):
    """Chat format used for fine-tuning: system + user note + assistant JSON."""
    return {"messages": [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": example["note"]},
        {"role": "assistant", "content": json.dumps(example["label"], ensure_ascii=False)},
    ], "label": example["label"]}


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def parse_json(text):
    """Models wrap JSON in ``` fences or add text around it. Take the first {...} block."""
    if not text:
        return None
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def score(pred, label):
    """Per-field correctness + 'all' (every field right)."""
    if not isinstance(pred, dict):
        return {**{f: False for f in FIELDS}, "valid_json": False, "all": False}
    res = {}
    for f in FIELDS:
        p, t = pred.get(f), label[f]
        if f == "quantity":
            try:
                res[f] = int(p) == t
            except (TypeError, ValueError):
                res[f] = False
        else:
            res[f] = str(p).strip() == t
    res["valid_json"] = True
    res["all"] = all(res[f] for f in FIELDS)
    return res


def summarize(scores):
    n = len(scores)
    return {k: round(sum(s[k] for s in scores) / n, 3) for k in ["valid_json", "all"] + FIELDS}


if __name__ == "__main__":
    for ex in make_examples(8, seed=1):
        print(ex["note"])
        print("   ->", json.dumps(ex["label"], ensure_ascii=False))
