"""
Step 3 - structured outputs (JSON).

Free text is for humans. In a pipeline you want data your code can use directly:
    "The sentiment is positive, score 0.9"  ->  {"sentiment": "positive", "score": 0.9}

Two ingredients:
  - response_format={"type": "json_object"} forces the model to return valid JSON
  - the system prompt describes the exact fields you want
LLM as a structured data extractor is one of the most used patterns in industry.
"""
import json

from llm_utils import ask_json

# --- 1. sentiment analysis -------------------------------------------------------
print("=" * 70 + "\n1. Sentiment analysis\n")
result = ask_json(
    """You are a sentiment analysis expert. Respond with a JSON object with exactly these fields:
{"sentiment": "positive" | "negative" | "neutral", "score": float between 0 and 1,
 "keywords": list of 3 words that justify the sentiment}""",
    "I really love this new computer vision model, it works amazingly well on my dataset!",
)
print(result)
print(f"sentiment={result['sentiment']}  score={result['score']}  keywords={result['keywords']}")

# --- 2. information extraction ---------------------------------------------------
print("\n" + "=" * 70 + "\n2. Extraction from meeting notes\n")
notes = """
Meeting notes - June 10 2025
Attendees: Marie (Data Scientist), Thomas (ML Engineer), Sophie (Product Manager)
We discussed the new recommendation model. Deadline is July 15.
Main issues: lack of training data, latency too high in production (>200ms).
Next steps: Thomas will collect more data by June 20, Marie will optimize inference.
"""
result = ask_json(
    """You extract structured information from meeting notes. Respond with a JSON object:
{"date": "YYYY-MM-DD", "attendees": [{"name": str, "role": str}], "project": str, "deadline": "YYYY-MM-DD",
 "issues": [str], "next_steps": [{"person": str, "task": str, "due": "YYYY-MM-DD or null"}]}""",
    f"Extract all information from these meeting notes:\n{notes}",
)
print(json.dumps(result, indent=2))

# --- 3. quality report (CV flavor) ------------------------------------------------
print("\n" + "=" * 70 + "\n3. Quality report from an image description\n")
description = """
Industrial camera image from a production line.
Visible scratches on the metallic surface of the part.
Lighting is uneven, darker on the left side.
The part looks slightly misaligned compared to the reference position.
"""
result = ask_json(
    """You are a computer vision quality control expert. Return a JSON quality report:
{"quality_pass": boolean, "defects": [{"type": str, "severity": "low" | "medium" | "high"}],
 "imaging_issues": [str], "recommended_actions": [str], "confidence": float between 0 and 1}""",
    f"Produce a quality report for this image description:\n{description}",
)
print(json.dumps(result, indent=2))
print(f"\nquality_pass={result['quality_pass']}, {len(result['defects'])} defect(s):")
for d in result["defects"]:
    print(f"  - {d['type']} ({d['severity']})")

# Notes:
#  - JSON mode guarantees valid JSON, not the RIGHT fields. In real code, validate the result
#    (e.g. with a pydantic model) and retry or fail when a field is missing.
#  - Some APIs also offer strict schemas ("structured outputs" with a JSON schema), which
#    guarantee the fields too. Check the provider docs, support varies by model.
#  - In pillar 3bis you'll do example 3 with real images instead of a text description.
