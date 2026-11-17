# Pillar 1 - LLMs and APIs

Calling an LLM from Python, prompt engineering, structured JSON outputs, function calling,
and a small model-monitoring assistant that combines them. $0 with the Groq free tier.

## Setup

```
cd d:\pythoncode\pillar1
conda activate first-env
pip install -r requirements.txt
```

The `.env` with `GROQ_API_KEY` in `d:\pythoncode` is picked up automatically.

**Model note:** the first version of these scripts used `llama-3.3-70b-versatile`, which Groq
retired in August 2026. They now default to `openai/gpt-oss-120b`. To try another model, add
`GROQ_MODEL=...` to `.env` (list on console.groq.com/docs/models). `llm_utils.py` already handles
the settings gpt-oss and Qwen models need.

## Steps

| # | Script | What you learn |
|---|---|---|
| 1 | `01_first_call.py` | The anatomy of an API call: messages, tokens, finish reason, reasoning |
| 2 | `02_prompting.py` | Vague vs precise system prompt, chain-of-thought, few-shot |
| 3 | `03_structured_outputs.py` | JSON mode: sentiment, information extraction, quality report |
| 4 | `04_function_calling.py` | The tool-calling loop, the model chaining calls by itself |
| 5 | `05_project_model_monitor.py` | Mini project: CLI assistant with memory and 6 tools |

## What comes next

- Pillar 2: RAG, giving the model your own documents
- Pillar 3: agents, the function-calling loop over many steps
- Pillar 3bis: the same with images (vision-language models + CV tools)
