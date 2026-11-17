"""
A small fake "ML platform" for the agents to work with: the CV model registry from pillars
1 and 2, drift numbers, a cost estimator and a retraining queue.

The functions are plain Python. agent_utils.Tool turns them into LLM tools (01-05, 08),
LangChain wraps them in 06, and the MCP server exposes them in 07. Same functions everywhere,
only the wrapper changes.
"""
import ast
import operator
from datetime import date
from typing import Annotated, Literal

from pydantic import Field

from agent_utils import Tool

MODELS = {
    "resnet50": {"task": "classification", "owner": "Marie", "status": "production", "accuracy": 0.923,
                 "latency_ms": 45, "params_M": 25, "last_trained": "2024-01-15", "psi": 0.04,
                 "retrainings_this_quarter": 0},
    "yolov8": {"task": "defect detection", "owner": "Thomas", "status": "production", "accuracy": 0.812,
               "latency_ms": 12, "params_M": 3, "last_trained": "2024-03-18", "psi": 0.31,
               "retrainings_this_quarter": 1},
    "efficientnet": {"task": "surface classification", "owner": "Julien", "status": "production",
                     "accuracy": 0.901, "latency_ms": 30, "params_M": 12, "last_trained": "2024-02-10",
                     "psi": 0.14, "retrainings_this_quarter": 1},
    "unet_coating": {"task": "coating segmentation", "owner": "Julien", "status": "production",
                     "accuracy": 0.874, "latency_ms": 38, "params_M": 31, "last_trained": "2024-01-29",
                     "psi": 0.28, "retrainings_this_quarter": 2},
    "vit_large": {"task": "classification", "owner": "Marie", "status": "not_deployed", "accuracy": 0.951,
                  "latency_ms": 120, "params_M": 307, "last_trained": "2023-11-20", "psi": None,
                  "retrainings_this_quarter": 0},
}

PSI_WARNING, PSI_ALERT = 0.10, 0.25
MAX_RETRAININGS_PER_QUARTER = 2
GPU_EUR_PER_HOUR = 2.5
LABELING_EUR_PER_SAMPLE = 0.04

RETRAINING_QUEUE = []


def _find(model_name):
    key = model_name.lower().strip().replace("-", "_").replace(" ", "_")
    if key not in MODELS:
        raise ValueError(f"unknown model '{model_name}'. Available models: {list(MODELS)}")
    return key


def _drift_level(psi):
    if psi is None:
        return "not monitored"
    return "drift" if psi > PSI_ALERT else "warning" if psi > PSI_WARNING else "stable"


# ---------------------------------------------------------------------------
# tool functions (docstrings and type hints ARE the tool descriptions)
# ---------------------------------------------------------------------------

def list_models(status: Annotated[Literal["all", "production", "not_deployed"],
                                  Field(description="Filter on deployment status")] = "all"):
    """List the computer vision models in the registry with their owner and status."""
    return [
        {"model": name, "task": m["task"], "owner": m["owner"], "status": m["status"]}
        for name, m in MODELS.items()
        if status == "all" or m["status"] == status
    ]


def get_model(model_name: Annotated[str, Field(description="Model name, e.g. 'yolov8'")]):
    """Get all registry details for one model: metrics, latency, size, owner, last training date,
    current drift (PSI) and how many times it was retrained this quarter."""
    key = _find(model_name)
    return {"model": key, **MODELS[key], "drift_level": _drift_level(MODELS[key]["psi"])}


def get_drift_report():
    """Latest weekly drift report for all production models: PSI and level
    (stable < 0.10 <= warning <= 0.25 < drift)."""
    return {
        "week": "2024-W14",
        "thresholds": {"warning": PSI_WARNING, "drift": PSI_ALERT},
        "models": [
            {"model": name, "psi": m["psi"], "level": _drift_level(m["psi"])}
            for name, m in MODELS.items() if m["status"] == "production"
        ],
    }


def estimate_retraining_cost(
    model_name: Annotated[str, Field(description="Model to retrain")],
    n_samples: Annotated[int, Field(gt=0, description="Number of new labeled samples to train on")],
):
    """Estimate the cost of retraining a model on n_samples new images:
    GPU hours, GPU cost, labeling cost and total, in euros."""
    key = _find(model_name)
    gpu_hours = round(1 + MODELS[key]["params_M"] * n_samples / 150_000, 1)
    gpu_cost = round(gpu_hours * GPU_EUR_PER_HOUR, 2)
    labeling = round(n_samples * LABELING_EUR_PER_SAMPLE, 2)
    return {"model": key, "n_samples": n_samples, "gpu_hours": gpu_hours, "gpu_cost_eur": gpu_cost,
            "labeling_cost_eur": labeling, "total_eur": round(gpu_cost + labeling, 2)}


def flag_for_retraining(
    model_name: Annotated[str, Field(description="Model to retrain")],
    reason: Annotated[str, Field(description="Why it needs retraining, one sentence")],
    priority: Literal["low", "medium", "high", "critical"],
):
    """Add a model to the retraining queue. This launches GPU jobs, so it requires human approval.
    Fails if the model already used its retraining budget for the quarter."""
    key = _find(model_name)
    used = MODELS[key]["retrainings_this_quarter"]
    if used >= MAX_RETRAININGS_PER_QUARTER:
        raise PermissionError(f"{key} was already retrained {used} times this quarter (max "
                              f"{MAX_RETRAININGS_PER_QUARTER}). The team lead must approve an exception.")
    entry = {"model": key, "reason": reason, "priority": priority, "flagged_on": str(date.today())}
    RETRAINING_QUEUE.append(entry)
    return {"status": "queued", **entry, "queue_length": len(RETRAINING_QUEUE)}


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.USub: operator.neg, ast.Mod: operator.mod}


def calculator(expression: Annotated[str, Field(description="Arithmetic expression, e.g. '(0.31-0.25)/0.25*100'")]):
    """Evaluate an arithmetic expression. Use it for any computation instead of doing math in your head."""
    def ev(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        raise ValueError("only numbers and + - * / ** % are allowed")
    # never eval() a string written by an LLM, parse it and only allow arithmetic
    return round(ev(ast.parse(expression, mode="eval").body), 6)


FUNCTIONS = [list_models, get_model, get_drift_report, estimate_retraining_cost, flag_for_retraining, calculator]
TOOLS = [Tool(f, requires_approval=(f is flag_for_retraining)) for f in FUNCTIONS]

SYSTEM_PROMPT = """You are an MLOps assistant for a computer vision team.
You can query the model registry, drift reports and costs, and queue retraining jobs.

Rules:
- Get facts from the tools, never invent numbers. Use the calculator for any arithmetic.
- Drift policy: PSI > 0.25 = drift, 0.10-0.25 = warning. A model can be retrained at most
  2 times per quarter without the team lead's approval.
- Only flag a model for retraining if the user asked for actions, not just information.
- Final answers: short, structured, with the numbers you used."""
