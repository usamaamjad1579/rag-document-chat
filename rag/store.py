from dataclasses import dataclass

import chromadb

from .chunker import Chunk


@dataclass
class Hit:
    text: str
    source: str
    page: int | None
    score: float  # cosine similarity, higher is better

    @property
    def label(self) -> str:
        return f"{self.source}, p. {self.page}" if self.page else self.source


class VectorStore:
    """Thin wrapper around a ChromaDB collection.

    By default Chroma embeds text locally with the all-MiniLM-L6-v2 model, so
    indexing documents does not need an API key.
    """

    def __init__(self, persist_dir: str | None = None, collection: str = "documents", embedding_function=None):
        self.client = chromadb.PersistentClient(path=persist_dir) if persist_dir else chromadb.EphemeralClient()
        kwargs = {"embedding_function": embedding_function} if embedding_function else {}
        self.collection = self.client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"}, **kwargs
        )

    def add(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0
        metadatas = []
        for chunk in chunks:
            meta = {"source": chunk.source}
            if chunk.page is not None:
                meta["page"] = chunk.page
            metadatas.append(meta)
        self.collection.upsert(
            ids=[c.id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=metadatas,
        )
        return len(chunks)

    def search(self, query: str, k: int = 4) -> list[Hit]:
        count = self.collection.count()
        if count == 0:
            return []
        result = self.collection.query(query_texts=[query], n_results=min(k, count))
        hits = []
        for text, meta, distance in zip(result["documents"][0], result["metadatas"][0], result["distances"][0]):
            hits.append(Hit(text=text, source=meta["source"], page=meta.get("page"), score=1 - distance))
        return hits

    def sources(self) -> list[str]:
        metas = self.collection.get(include=["metadatas"])["metadatas"]
        return sorted({m["source"] for m in metas})

    def delete_source(self, source: str) -> None:
        self.collection.delete(where={"source": source})

    def count(self) -> int:
        return self.collection.count()
