"""
Step 4 - function calling (tool use).

The model doesn't only answer: it can ask YOUR code to run a function, with arguments it
chooses. The loop:
    1. you describe your functions to the model (name, description, JSON schema of the params)
    2. the model reads the question and returns tool calls instead of text
    3. your code runs the functions
    4. you send the results back
    5. the model answers, or asks for more calls
The model never runs anything itself: your code stays in control. This is the base of agents (pillar 3).
"""
import json

from llm_utils import assistant_message, chat

# --- 1. the real Python functions --------------------------------------------------

FAKE_DB = {  # in real life: a database, MLflow...
    "resnet50": {"dataset": "imagenet", "accuracy": 0.923, "latency_ms": 45, "params_M": 25},
    "yolov8": {"dataset": "coco", "accuracy": 0.876, "latency_ms": 12, "params_M": 3},
    "vit_large": {"dataset": "imagenet", "accuracy": 0.951, "latency_ms": 120, "params_M": 307},
}


def get_model_performance(model_name, dataset):
    model = FAKE_DB.get(model_name.lower())
    if not model or model["dataset"] != dataset.lower():
        return {"error": f"no data for {model_name} on {dataset}. Known models: {list(FAKE_DB)}"}
    return {"model": model_name, **model}


def compare_models(model_a, model_b, metric):
    a, b = FAKE_DB.get(model_a.lower()), FAKE_DB.get(model_b.lower())
    if not a or not b:
        return {"error": f"unknown model. Known models: {list(FAKE_DB)}"}
    lower_is_better = metric in ("latency_ms", "params_M")
    winner = model_a if (a[metric] < b[metric]) == lower_is_better else model_b
    return {"metric": metric, model_a: a[metric], model_b: b[metric], "best": winner}


def flag_model_for_retraining(model_name, reason):
    print(f"    [SYSTEM] {model_name} flagged for retraining: {reason}")
    return {"status": "flagged", "model": model_name, "reason": reason}


available_functions = {
    "get_model_performance": get_model_performance,
    "compare_models": compare_models,
    "flag_model_for_retraining": flag_model_for_retraining,
}

# --- 2. what the model sees: descriptions + JSON schemas ----------------------------

tools = [
    {"type": "function", "function": {
        "name": "get_model_performance",
        "description": "Get performance metrics of an ML model on a dataset",
        "parameters": {"type": "object", "properties": {
            "model_name": {"type": "string", "description": "e.g. resnet50"},
            "dataset": {"type": "string", "description": "e.g. imagenet"}},
            "required": ["model_name", "dataset"]}}},
    {"type": "function", "function": {
        "name": "compare_models",
        "description": "Compare two models on one metric",
        "parameters": {"type": "object", "properties": {
            "model_a": {"type": "string"},
            "model_b": {"type": "string"},
            "metric": {"type": "string", "enum": ["accuracy", "latency_ms", "params_M"]}},
            "required": ["model_a", "model_b", "metric"]}}},
    {"type": "function", "function": {
        "name": "flag_model_for_retraining",
        "description": "Flag a model for retraining when its performance is not good enough",
        "parameters": {"type": "object", "properties": {
            "model_name": {"type": "string"},
            "reason": {"type": "string", "description": "why it needs retraining"}},
            "required": ["model_name", "reason"]}}},
]

SYSTEM_PROMPT = """You are an ML model management assistant. You can check model performance,
compare models and flag models for retraining. Always use the tools to get real data before answering."""


# --- 3. the loop ------------------------------------------------------------------

def run(question):
    print("=" * 70)
    print(f"User: {question}")
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}]

    message = chat(messages, tools=tools)
    while message.tool_calls:
        messages.append(assistant_message(message))
        for call in message.tool_calls:
            args = json.loads(call.function.arguments)
            result = available_functions[call.function.name](**args)
            print(f"  -> {call.function.name}({args})")
            print(f"     {result}")
            messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)})
        message = chat(messages, tools=tools)  # the model sees the results, answers or calls again

    print(f"\nAssistant: {message.content}\n")


run("What is the accuracy of ResNet50 on ImageNet?")
run("Which is faster, YOLOv8 or ViT Large?")
run("ViT Large has 120ms latency, which is too slow for production. What should we do?")

# Question 3 is the interesting one: nobody told the model to check the metrics AND flag the
# model, it chains the calls itself. That's the beginning of agentic behavior.
