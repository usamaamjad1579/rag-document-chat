import re
from dataclasses import dataclass

from .chunker import chunk_pages
from .llm import ChatModel
from .loader import Page
from .store import Hit, VectorStore

NOT_FOUND = "I couldn't find that in the uploaded documents."

SYSTEM_PROMPT = f"""You answer questions about the user's documents.

Rules:
- Use only the numbered context passages. Do not use outside knowledge.
- Cite every claim with its passage number in square brackets, like [1] or [2][3].
- If the passages do not contain the answer, reply exactly: "{NOT_FOUND}"
- Keep answers short and direct."""

CONDENSE_PROMPT = """Rewrite the follow-up question so it can be understood without the chat history.
Return only the rewritten question.

Chat history:
{history}

Follow-up question: {question}"""


@dataclass
class Answer:
    text: str
    sources: list[Hit]
    query: str  # the standalone question that was actually searched


def format_context(hits: list[Hit]) -> str:
    return "\n\n".join(f"[{i}] ({hit.label})\n{hit.text}" for i, hit in enumerate(hits, start=1))


def cited_sources(text: str, hits: list[Hit]) -> list[Hit]:
    """Return the passages the model actually cited, in citation order."""
    numbers = dict.fromkeys(int(n) for n in re.findall(r"\[(\d+)\]", text))
    return [hits[n - 1] for n in numbers if 1 <= n <= len(hits)]


class RAGPipeline:
    def __init__(self, store: VectorStore, llm: ChatModel, top_k: int = 4, min_score: float = 0.25):
        self.store = store
        self.llm = llm
        self.top_k = top_k
        self.min_score = min_score

    def ingest(self, pages: list[Page], chunk_size: int = 800, overlap: int = 150) -> int:
        return self.store.add(chunk_pages(pages, chunk_size, overlap))

    def condense(self, question: str, history: list[dict] | None) -> str:
        """Turn a follow-up like "and for interns?" into a full search query."""
        if not history:
            return question
        transcript = "\n".join(f"{m['role']}: {m['content']}" for m in history[-4:])
        prompt = CONDENSE_PROMPT.format(history=transcript, question=question)
        rewritten = self.llm.chat([{"role": "user", "content": prompt}]).strip()
        return rewritten or question

    def retrieve(self, query: str) -> list[Hit]:
        return [hit for hit in self.store.search(query, self.top_k) if hit.score >= self.min_score]

    def answer(self, question: str, history: list[dict] | None = None) -> Answer:
        query = self.condense(question, history)
        hits = self.retrieve(query)
        if not hits:
            return Answer(text=NOT_FOUND, sources=[], query=query)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{format_context(hits)}\n\nQuestion: {query}"},
        ]
        text = self.llm.chat(messages).strip()
        if text == NOT_FOUND:
            return Answer(text=text, sources=[], query=query)
        return Answer(text=text, sources=cited_sources(text, hits) or hits, query=query)
