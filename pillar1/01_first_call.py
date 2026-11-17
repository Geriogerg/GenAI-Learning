"""
Step 1 - first LLM API call.

Your code sends a request to the model hosted by Groq, gets the answer back. Every LLM API
(Groq, OpenAI, Anthropic, Mistral, Gemini...) has the same shape:
    model     -> which model
    messages  -> the conversation: a system message (rules / role) + user messages
    max_tokens, temperature -> length and randomness of the answer

The API key comes from the .env file, never from the code.
"""
import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": "You are an expert data scientist. Answer concisely."},
        {"role": "user", "content": "What is the difference between RAG and fine-tuning? Give me 3 bullet points."},
    ],
    max_tokens=1000,
    temperature=0.3,
    # gpt-oss is a reasoning model (see below), "low" keeps its thinking short
    **({"reasoning_effort": "low"} if "gpt-oss" in MODEL else {}),
)

message = response.choices[0].message
print(message.content)

# The response has more than the text, worth looking at once:
print("\n" + "-" * 60)
print(f"model:          {response.model}")
print(f"finish reason:  {response.choices[0].finish_reason}   ('length' = cut by max_tokens)")
print(f"tokens:         {response.usage.prompt_tokens} in + {response.usage.completion_tokens} out")
if getattr(message, "reasoning", None):
    # reasoning models "think" before answering. These tokens are billed and count in max_tokens.
    print(f"reasoning:      {message.reasoning[:200]}...")

# Try: temperature=1.2 and run it 3 times (answers vary), or max_tokens=50 (finish reason 'length').
