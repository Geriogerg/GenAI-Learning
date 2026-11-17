"""
Pillar 3 mini project - a data analysis agent.

Give it a CSV and a question. It explores the data by writing and running pandas code,
makes charts, checks its hypotheses, and writes a markdown report with the figures.
Basically a junior analyst: you'll see it's good at the mechanics and you still need to
check its conclusions.

Usage:
    python data/make_sales_data.py           # once, creates data/sales.csv
    python 09_project_data_agent.py
    python 09_project_data_agent.py --question "Why did North revenue drop in March?"
    python 09_project_data_agent.py --csv my_file.csv --confirm   # review each code before it runs

Output (report.md + charts) goes to outputs/run_<timestamp>/
"""
import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from agent_utils import DATA_DIR, OUTPUT_DIR, Agent, tool

parser = argparse.ArgumentParser()
parser.add_argument("--csv", default=str(DATA_DIR / "sales.csv"))
parser.add_argument("--question", default="Give me an overview of this sales data and anything "
                                           "unusual that the business should know about.")
parser.add_argument("--confirm", action="store_true", help="ask before running each piece of code")
parser.add_argument("--max-steps", type=int, default=20)
args = parser.parse_args()

CSV_PATH = Path(args.csv).resolve()
if not CSV_PATH.exists():
    raise SystemExit(f"{CSV_PATH} not found. For the demo data run: python data/make_sales_data.py")

RUN_DIR = OUTPUT_DIR / f"run_{datetime.now():%Y%m%d_%H%M%S}"
RUN_DIR.mkdir(parents=True)

# Every code run starts from this, so each snippet is self-contained (no hidden state
# between runs: if the agent needs a variable again, it recomputes it).
PREAMBLE = f"""
import warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
pd.set_option("display.width", 160); pd.set_option("display.max_columns", 30)
df = pd.read_csv(r"{CSV_PATH}")
"""

# Not a real sandbox! Just a guard against the obvious. For anything serious, run the code
# in a container (Docker) or a hosted sandbox (E2B, Modal...), never on your own machine
# with your files and credentials around.
BLOCKED = ["import os", "import sys", "subprocess", "shutil", "socket", "requests", "urllib",
           "open(", "__import__", "eval(", "exec(", "os.", "pathlib", ".to_csv", "remove", "rmdir"]


@tool
def describe_dataset():
    """First thing to call: columns, types, size, missing values, a few rows."""
    df = pd.read_csv(CSV_PATH)
    return (f"shape: {df.shape}\n\ndtypes:\n{df.dtypes.to_string()}\n\n"
            f"missing values:\n{df.isna().sum().to_string()}\n\nfirst rows:\n{df.head(5).to_string()}")


@tool
def run_python(code: str):
    """Run Python code on the dataset and return what it prints. `df` (pandas DataFrame of the CSV),
    pd, np and plt are already loaded. Use print() to see results, keep outputs short (aggregates,
    not full tables). Save charts with plt.savefig("name.png") then plt.close(); they go in the
    report folder. No file access, network or system calls."""
    found = [b for b in BLOCKED if b in code]
    if found:
        return f"Error: blocked for safety: {found}. Only analysis code on df is allowed."
    if args.confirm:
        print("\n" + "-" * 60 + f"\n{code}\n" + "-" * 60)
        if input("Run this code? [Y/n] ").strip().lower() == "n":
            return "The user refused to run this code."

    script = RUN_DIR / "_last_run.py"
    script.write_text(PREAMBLE + "\n" + code, encoding="utf-8")
    try:
        proc = subprocess.run([sys.executable, "-I", str(script)], cwd=RUN_DIR, capture_output=True,
                              text=True, timeout=60)
    except subprocess.TimeoutExpired:
        return "Error: the code took more than 60 s and was stopped."

    out = proc.stdout.strip()
    if proc.returncode != 0:
        # only the end of the traceback is useful to the model
        err = "\n".join(proc.stderr.strip().splitlines()[-6:])
        return f"Error:\n{err}" + (f"\n\nOutput before the error:\n{out[:1000]}" if out else "")
    if len(out) > 3000:
        out = out[:3000] + "\n... (output truncated, print less)"
    return out or "(no output, use print())"


@tool
def write_report(markdown: str):
    """Write the final report as markdown. Call it once at the end. Include charts with ![title](name.png)."""
    path = RUN_DIR / "report.md"
    path.write_text(markdown, encoding="utf-8")
    charts = sorted(p.name for p in RUN_DIR.glob("*.png"))
    missing = [c for c in charts if c not in markdown]
    note = f" Charts not referenced in the report: {missing}" if missing else ""
    return f"Report written to {path}.{note}"


SYSTEM_PROMPT = """You are a senior data analyst. You answer business questions about a dataset
by running code on it, like a careful analyst would.

Method:
1. describe_dataset first. Note data quality issues (missing values, duplicates, impossible values)
   and say how you handle them.
2. Look at the big picture (totals, trends over time, by main dimensions).
3. When something looks unusual, form a hypothesis and test it by drilling down
   (which region? which product? which dates?). Don't stop at the first level.
4. Make 2 to 4 charts that support the main findings (clear titles and axis labels).
5. Call write_report with: a 3-line executive summary, findings with the numbers that back them,
   data quality notes, and open questions / recommended next checks.

Rules:
- Every number in the report must come from code you ran. Never estimate.
- Short code snippets, one question each. Print aggregates, not raw tables.
- If code fails, read the error and fix it."""

agent = Agent([describe_dataset, run_python, write_report], SYSTEM_PROMPT, max_steps=args.max_steps)

print(f"Dataset: {CSV_PATH}\nQuestion: {args.question}\nOutputs: {RUN_DIR}\n")
result = agent.run(args.question)

print("\n" + "=" * 80)
print(result.answer)
print(f"\n{result.steps} steps, {result.tokens} tokens, {result.tool_errors} tool errors")
report = RUN_DIR / "report.md"
print(f"Report: {report}" if report.exists() else "No report written (step limit? see the trace above)")

# Open report.md in VS Code (Ctrl+Shift+V for the preview with the charts).
#
# Questions to try on the demo data:
#   --question "Why did North revenue drop in March?"
#   --question "Is there a product quality problem?"
#   --question "Which region is growing fastest, and is it real growth or noise?"
#
# Then compare with the spoiler at the bottom of the README. Things to watch:
#  - does it clean the data (duplicates, negative units, missing prices) before computing?
#  - does it drill down to the product level, or stop at "North dropped in March"?
#  - does it confuse revenue noise from expensive products with real trends?
