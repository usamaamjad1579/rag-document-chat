"""Test helpers that keep the suite fast and offline.

`HashEmbedding` replaces the real embedding model with a bag-of-words hash,
and `FakeLLM` returns scripted replies instead of calling Groq.
"""

import hashlib
import math
import re
import uuid

import pytest
from chromadb import Documents, EmbeddingFunction, Embeddings

from rag import VectorStore


class HashEmbedding(EmbeddingFunction):
    DIM = 256

    def __init__(self):
        pass

    def __call__(self, input: Documents) -> Embeddings:
        vectors = []
        for text in input:
            vec = [0.0] * self.DIM
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.DIM] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            vectors.append([v / norm for v in vec])
        return vectors

    @staticmethod
    def name() -> str:
        return "test-hash-embedding"

    def get_config(self) -> dict:
        return {}

    @staticmethod
    def build_from_config(config: dict) -> "HashEmbedding":
        return HashEmbedding()


class FakeLLM:
    def __init__(self, replies=None):
        self.replies = list(replies or [])
        self.calls: list[list[dict]] = []

    def chat(self, messages):
        self.calls.append(messages)
        return self.replies.pop(0) if self.replies else "Answer [1]"


@pytest.fixture
def store():
    return VectorStore(collection=f"test-{uuid.uuid4().hex[:8]}", embedding_function=HashEmbedding())


@pytest.fixture
def fake_llm():
    return FakeLLM()
