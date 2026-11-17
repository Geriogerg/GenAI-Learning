"""
Pillar 1 mini project - ML model monitor (CLI chat).

Everything from pillar 1 together:
    system prompt         keeps the assistant focused on ML monitoring
    conversation history  it remembers earlier turns ("flag those models")
    function calling      it fetches data and takes actions through your functions
    structured data       every function returns clean JSON

Try, in this order:
    Give me a full health report
    Which models have drift?
    Flag all drifted models for retraining with appropriate priorities
    What's in the retraining queue now?
    Tell me about EfficientNet in detail
    What's the weather today?          <- should politely redirect
"""
import json
from datetime import datetime

from llm_utils import assistant_message, chat

MODELS_DB = {
    "resnet50": {"accuracy": 0.923, "latency_ms": 45, "params_M": 25, "dataset": "imagenet",
                 "last_trained": "2024-01-15", "production": True, "drift_detected": False},
    "yolov8": {"accuracy": 0.876, "latency_ms": 12, "params_M": 3, "dataset": "coco",
               "last_trained": "2024-03-02", "production": True, "drift_detected": True},
    "vit_large": {"accuracy": 0.951, "latency_ms": 120, "params_M": 307, "dataset": "imagenet",
                  "last_trained": "2023-11-20", "production": False, "drift_detected": False},
    "efficientnet": {"accuracy": 0.901, "latency_ms": 30, "params_M": 8, "dataset": "imagenet",
                     "last_trained": "2024-02-10", "production": True, "drift_detected": True},
}
RETRAINING_QUEUE = []


# --- functions -------------------------------------------------------------------

def list_models(production_only=False):
    names = [k for k, v in MODELS_DB.items() if v["production"] or not production_only]
    return {"models": names, "count": len(names)}


def get_model_details(model_name):
    model = MODELS_DB.get(model_name.lower())
    if not model:
        return {"error": f"model '{model_name}' not found. Available: {list(MODELS_DB)}"}
    return {"model": model_name.lower(), **model}


def get_models_with_drift():
    drifted = [k for k, v in MODELS_DB.items() if v["drift_detected"]]
    return {"models_with_drift": drifted, "count": len(drifted)}


def generate_health_report():
    prod = {k: v for k, v in MODELS_DB.items() if v["production"]}
    drifted = [k for k, v in prod.items() if v["drift_detected"]]
    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "production_models": len(prod),
        "models_with_drift": drifted,
        "average_accuracy": round(sum(v["accuracy"] for v in prod.values()) / len(prod), 3),
        "average_latency_ms": round(sum(v["latency_ms"] for v in prod.values()) / len(prod), 1),
        "overall_status": "CRITICAL" if len(drifted) > 1 else "WARNING" if drifted else "OK",
    }


def flag_for_retraining(model_name, reason, priority):
    if model_name.lower() not in MODELS_DB:
        return {"error": f"model '{model_name}' not found"}
    entry = {"model": model_name.lower(), "reason": reason, "priority": priority,
             "flagged_at": datetime.now().strftime("%Y-%m-%d %H:%M")}
    RETRAINING_QUEUE.append(entry)
    return {"status": "success", "queued": entry}


def get_retraining_queue():
    return {"queue": RETRAINING_QUEUE, "total": len(RETRAINING_QUEUE)}


available_functions = {f.__name__: f for f in [list_models, get_model_details, get_models_with_drift,
                                                generate_health_report, flag_for_retraining,
                                                get_retraining_queue]}


def tool(name, description, properties=None, required=None):
    """Small helper so the tool list below stays readable."""
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {
        "type": "object", "properties": properties or {}, "required": required or []}}}


tools = [
    tool("list_models", "List all models, optionally only the production ones",
         {"production_only": {"type": "boolean"}}),
    tool("get_model_details", "Detailed metrics and status of one model",
         {"model_name": {"type": "string"}}, ["model_name"]),
    tool("get_models_with_drift", "All models where data drift was detected"),
    tool("generate_health_report", "Health report of the production ML system"),
    tool("flag_for_retraining", "Add a model to the retraining queue with a reason and a priority",
         {"model_name": {"type": "string"}, "reason": {"type": "string"},
          "priority": {"type": "string", "enum": ["low", "medium", "high", "critical"]}},
         ["model_name", "reason", "priority"]),
    tool("get_retraining_queue", "Current retraining queue"),
]

SYSTEM_PROMPT = """You are an ML model monitoring assistant for a computer vision team.
You have access to a model registry with metrics, drift detection and a retraining queue.

Behavior:
- Always use tools to fetch real data before answering, never invent numbers
- Be concise and technical, you are talking to data scientists
- If drift is detected on a production model, suggest flagging it for retraining
- In reports, put critical issues first
- If asked something outside ML monitoring, politely redirect"""

history = []  # user / assistant / tool messages, kept across turns


def chat_turn(user_input):
    history.append({"role": "user", "content": user_input})
    message = chat([{"role": "system", "content": SYSTEM_PROMPT}] + history, tools=tools)

    while message.tool_calls:
        history.append(assistant_message(message))
        for call in message.tool_calls:
            args = json.loads(call.function.arguments or "{}")
            result = available_functions[call.function.name](**args)
            print(f"  [{call.function.name}({args})]")
            history.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)})
        message = chat([{"role": "system", "content": SYSTEM_PROMPT}] + history, tools=tools)

    history.append({"role": "assistant", "content": message.content or ""})
    return message.content


if __name__ == "__main__":
    print("ML Model Monitor - type 'quit' to exit\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if user_input:
            print(f"\nAssistant: {chat_turn(user_input)}\n" + "-" * 60)

# Note: the history grows with every tool result. Fine for a short session, but in a long one
# it gets slow and expensive. Pillar 3 step 5 shows how to summarize it.
