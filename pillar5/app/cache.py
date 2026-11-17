"""
Two caches in front of the LLM:

  ExactCache     same question (after normalization) -> same answer. Safe, but only hits on
                 literal repeats.
  SemanticCache  a question whose embedding is very close to a cached one -> reuse its answer.
                 Hits much more often, and can be WRONG: "who owns yolov8?" and "who owns
                 resnet50?" are close in embedding space but have different answers.
                 Lesson 03 measures where the threshold should be.

In-memory = per process, lost on restart. With several workers/servers you'd use Redis.
"""
import re
import time
from collections import OrderedDict

import numpy as np


def normalize(q):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", q.lower())).strip()


class ExactCache:
    def __init__(self, ttl_s=3600, max_items=1000):
        self.ttl_s, self.max_items = ttl_s, max_items
        self.items = OrderedDict()

    def get(self, question):
        key = normalize(question)
        item = self.items.get(key)
        if item and time.time() - item[0] < self.ttl_s:
            self.items.move_to_end(key)
            return item[1]
        self.items.pop(key, None)
        return None

    def put(self, question, value):
        self.items[normalize(question)] = (time.time(), value)
        self.items.move_to_end(normalize(question))
        while len(self.items) > self.max_items:
            self.items.popitem(last=False)  # evict least recently used


class SemanticCache:
    def __init__(self, embed, threshold=0.95, ttl_s=3600, max_items=1000):
        self.embed, self.threshold, self.ttl_s, self.max_items = embed, threshold, ttl_s, max_items
        self.vectors, self.entries = [], []  # parallel lists: (timestamp, question, value)

    def _vec(self, text):
        v = np.asarray(self.embed([text])[0], dtype=np.float32)
        return v / (np.linalg.norm(v) + 1e-9)

    def lookup(self, question):
        """Returns (value, similarity, cached_question) or (None, best_similarity, None)."""
        if not self.vectors:
            return None, 0.0, None
        sims = np.stack(self.vectors) @ self._vec(question)
        best = int(sims.argmax())
        ts, cached_q, value = self.entries[best]
        if sims[best] >= self.threshold and time.time() - ts < self.ttl_s:
            return value, float(sims[best]), cached_q
        return None, float(sims[best]), None

    def put(self, question, value):
        self.vectors.append(self._vec(question))
        self.entries.append((time.time(), question, value))
        if len(self.entries) > self.max_items:
            self.vectors.pop(0)
            self.entries.pop(0)
