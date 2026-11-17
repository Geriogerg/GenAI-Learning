"""
Step 2 - prompt engineering.

Same model, same question, 4 ways of asking. The model doesn't change, the prompt changes
everything:
    1. vague system prompt
    2. precise system prompt with an output structure
    3. chain-of-thought: ask it to reason step by step
    4. few-shot: show examples of the expected input -> output
"""
from llm_utils import ask

QUESTION = "Should I use RAG or fine-tuning for my project?"


def run(title, system_prompt, user_prompt):
    print("=" * 70)
    print(title)
    print("=" * 70)
    print(ask(system_prompt, user_prompt))
    print()


run("1. Vague system prompt",
    "You are a helpful assistant.",
    QUESTION)

run("2. Precise system prompt",
    """You are an expert ML engineer working on production systems.
When asked a technical question, always structure your answer as:
1. One line answer
2. When to use each option (2 bullet points max)
3. A concrete example""",
    QUESTION)

run("3. Chain of thought",
    "You are an expert ML engineer.",
    QUESTION + " Think step by step before answering, then give your recommendation.")

run("4. Few-shot",
    "You are an expert ML engineer.",
    """I give you project descriptions, you answer RAG or fine-tuning and why, in one sentence.

Project: chat with our internal documentation that changes every week
Answer: RAG - the knowledge changes often, retraining every week would be too costly.

Project: classify support tickets into 10 fixed categories, we have 20k labeled tickets
Answer: Fine-tuning - fixed task, plenty of labeled data.

Project: answer customer questions about our product catalog of 5000 items, updated daily
Answer:""")

# What to notice:
#  1 -> a generic wall of text, "it depends"
#  2 -> same knowledge, usable format. The system prompt is the cheapest quality lever you have.
#  3 -> more thorough on ambiguous questions. Reasoning models (like gpt-oss) already do this
#       internally, so the gain is smaller than with older models.
#  4 -> the model copies the format AND the level of detail of the examples. Few-shot is the
#       most reliable way to get a precise output style.
