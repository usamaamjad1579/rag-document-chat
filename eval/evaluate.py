"""Measure retrieval quality on the sample documents.

For each question we know which passage holds the answer. A question is a
"hit" when that passage appears in the top-k retrieved chunks. We report
Hit@k and MRR (mean reciprocal rank) for several chunk sizes.

Run from the project root:
    python -m eval.evaluate
    python -m eval.evaluate --answers   # also generate answers with the LLM (needs GROQ_API_KEY)
"""

import argparse
import json
import uuid
from pathlib import Path

from rag import RAGPipeline, VectorStore, chunk_pages, load_folder
from rag.config import load_settings

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS = json.loads((ROOT / "eval" / "questions.json").read_text(encoding="utf-8"))
SAMPLE_DIR = ROOT / "data" / "sample"


def rank_of_answer(hits, item) -> int | None:
    for rank, hit in enumerate(hits, start=1):
        if hit.source == item["source"] and item["expected"].lower() in hit.text.lower():
            return rank
    return None


def evaluate(chunk_size: int, overlap: int, k: int) -> dict:
    store = VectorStore(collection=f"eval-{uuid.uuid4().hex[:8]}")
    store.add(chunk_pages(load_folder(SAMPLE_DIR), chunk_size, overlap))

    ranks = [rank_of_answer(store.search(item["question"], k), item) for item in QUESTIONS]
    hits = [r for r in ranks if r is not None]
    return {
        "chunk_size": chunk_size,
        "overlap": overlap,
        "chunks": store.count(),
        f"hit@{k}": len(hits) / len(QUESTIONS),
        "mrr": sum(1 / r for r in hits) / len(QUESTIONS),
        "misses": [q["question"] for q, r in zip(QUESTIONS, ranks) if r is None],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--answers", action="store_true", help="also print LLM answers for each question")
    args = parser.parse_args()

    print(f"{len(QUESTIONS)} questions, top-k = {args.k}\n")
    print(f"{'chunk':>6} {'overlap':>8} {'chunks':>7} {'hit@k':>7} {'MRR':>6}")
    for chunk_size, overlap in [(300, 50), (500, 100), (800, 150), (1200, 200)]:
        result = evaluate(chunk_size, overlap, args.k)
        print(
            f"{result['chunk_size']:>6} {result['overlap']:>8} {result['chunks']:>7} "
            f"{result[f'hit@{args.k}']:>7.2f} {result['mrr']:>6.2f}"
        )
        for miss in result["misses"]:
            print(f"{'':>8}miss: {miss}")

    if args.answers:
        from rag.llm import GroqChat

        settings = load_settings()
        store = VectorStore(collection=f"eval-{uuid.uuid4().hex[:8]}")
        pipeline = RAGPipeline(store, GroqChat(settings.groq_api_key, settings.model), top_k=args.k)
        pipeline.ingest(load_folder(SAMPLE_DIR), settings.chunk_size, settings.chunk_overlap)
        for item in QUESTIONS:
            answer = pipeline.answer(item["question"])
            print(f"\nQ: {item['question']}\nA: {answer.text}")


if __name__ == "__main__":
    main()
