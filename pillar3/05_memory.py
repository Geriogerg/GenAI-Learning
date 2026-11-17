"""
Step 5 - memory.

Two different problems people both call "memory":

1. Short-term (the conversation): every turn re-sends the whole history, so it grows until
   it costs too much or doesn't fit. Fix: keep the last turns as-is, summarize the older ones.

2. Long-term (across sessions): the agent forgets everything when the script stops.
   Fix: store facts in a vector DB (pillar 2 again) and give the agent tools to save and
   recall them, plus an automatic recall at the start of each turn.

Commands: /new (new session: short-term memory cleared, long-term kept), /memories, /forget, quit
"""
import uuid
from datetime import datetime

import chromadb
from chromadb.utils import embedding_functions

from agent_utils import ROOT, Agent, chat, tool
from registry_tools import SYSTEM_PROMPT as REGISTRY_PROMPT
from registry_tools import TOOLS as REGISTRY_TOOLS

# ---------------------------------------------------------------------------
# Long-term memory: a Chroma collection on disk
# ---------------------------------------------------------------------------

client = chromadb.PersistentClient(path=str(ROOT / "memory_db"))
memory = client.get_or_create_collection(
    "agent_memory",
    embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2"),
    metadata={"hnsw:space": "cosine"},
)


@tool
def remember(fact: str, category: str = "general"):
    """Save a durable fact about the user or the team for future sessions: preferences,
    decisions, constraints. One fact per call, written as a standalone sentence."""
    # skip near-duplicates, otherwise the memory fills up with the same fact 10 times
    if memory.count():
        res = memory.query(query_texts=[fact], n_results=1)
        if res["distances"][0] and res["distances"][0][0] < 0.1:
            return f"Already known: {res['documents'][0][0]}"
    memory.add(ids=[uuid.uuid4().hex], documents=[fact],
               metadatas=[{"category": category, "saved_at": datetime.now().isoformat(timespec="seconds")}])
    return "Saved."


@tool
def recall(query: str):
    """Search the long-term memory for facts saved in previous sessions."""
    if memory.count() == 0:
        return "Memory is empty."
    res = memory.query(query_texts=[query], n_results=min(5, memory.count()))
    return [d for d, dist in zip(res["documents"][0], res["distances"][0]) if dist < 0.7] or "Nothing relevant."


def auto_recall(user_input, k=3):
    """Memories injected before every turn, so the agent doesn't have to think of calling recall."""
    if memory.count() == 0:
        return []
    res = memory.query(query_texts=[user_input], n_results=min(k, memory.count()))
    return [d for d, dist in zip(res["documents"][0], res["distances"][0]) if dist < 0.6]


# ---------------------------------------------------------------------------
# Short-term memory: recent turns verbatim + a running summary of the rest
# ---------------------------------------------------------------------------

class ConversationMemory:
    def __init__(self, max_chars=3000, keep_last=4):
        self.messages = []   # only user/assistant turns, not the tool calls
        self.summary = ""
        self.max_chars = max_chars
        self.keep_last = keep_last

    def add(self, role, content):
        self.messages.append({"role": role, "content": content})
        if sum(len(m["content"]) for m in self.messages) > self.max_chars:
            self._compress()

    def _compress(self):
        old, self.messages = self.messages[:-self.keep_last], self.messages[-self.keep_last:]
        transcript = "\n".join(f"{m['role']}: {m['content']}" for m in old)
        msg = chat([
            {"role": "system", "content": "Update the running summary of a conversation with the new turns. "
                                          "Keep facts, numbers, decisions and open questions. Max 120 words."},
            {"role": "user", "content": f"Current summary:\n{self.summary or '(empty)'}\n\nNew turns:\n{transcript}"},
        ], max_tokens=800)
        self.summary = msg.content.strip()
        print(f"  (memory: compressed {len(old)} old messages into the summary)")

    def as_history(self):
        history = []
        if self.summary:
            history.append({"role": "system", "content": f"Summary of the earlier conversation: {self.summary}"})
        return history + self.messages

    def clear(self):
        self.messages, self.summary = [], ""


SYSTEM_PROMPT = REGISTRY_PROMPT + """

Memory:
- You have a long-term memory. When the user states a preference, a decision or a durable fact
  ("I'm the owner of...", "we decided to...", "always give costs in..."), save it with remember.
- Relevant memories are shown to you before each message. Use recall if you need more.
- Don't save one-off questions or numbers that are already in the registry."""

agent = Agent(REGISTRY_TOOLS + [remember, recall], SYSTEM_PROMPT, max_steps=6)
conversation = ConversationMemory()

print(f"Agent with memory. {memory.count()} long-term memories stored.")
print("Commands: /new, /memories, /forget, quit\n")

while True:
    user_input = input("You: ").strip()
    if not user_input:
        continue
    if user_input.lower() in ("quit", "exit", "q"):
        break
    if user_input == "/new":
        conversation.clear()
        print("  (new session: conversation forgotten, long-term memory kept)\n")
        continue
    if user_input == "/memories":
        data = memory.get(include=["documents", "metadatas"])
        for doc, meta in zip(data["documents"], data["metadatas"]):
            print(f"  [{meta['category']}] {doc}")
        print()
        continue
    if user_input == "/forget":
        client.delete_collection("agent_memory")
        print("  (long-term memory deleted, restart the script)\n")
        break

    memories = auto_recall(user_input)
    task = user_input
    if memories:
        task = "Relevant memories:\n" + "\n".join(f"- {m}" for m in memories) + f"\n\nUser: {user_input}"

    result = agent.run(task, history=conversation.as_history())
    print(f"\nAgent: {result.answer}\n")

    conversation.add("user", user_input)
    conversation.add("assistant", result.answer)

# Scenario to try:
#   I'm Julien, I own efficientnet and unet_coating.
#   For costs, always round to the nearest 100 euros.
#   how are my models doing?                    <- uses "my" from memory
#   /new
#   what would it cost to retrain my models on 5000 images?   <- new session, still knows
#   /memories
#
# In real products memory is a whole topic: what to save (the agent decides here, sometimes
# badly), when to update or delete a fact that became wrong, privacy (users must be able
# to see and delete what's stored). Keep that in mind before putting this in front of users.
