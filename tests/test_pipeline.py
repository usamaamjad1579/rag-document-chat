from conftest import FakeLLM

from rag import NOT_FOUND, Page, RAGPipeline
from rag.pipeline import cited_sources, format_context
from rag.store import Hit

DOCS = [
    Page(text="Employees receive 20 days of paid annual leave per year.", source="handbook.md", page=None),
    Page(text="The robot battery takes 2 hours and 30 minutes to charge.", source="manual.pdf", page=4),
]


def make_pipeline(store, llm, min_score=0.0):
    pipeline = RAGPipeline(store, llm, top_k=2, min_score=min_score)
    pipeline.ingest(DOCS, chunk_size=200, overlap=20)
    return pipeline


def test_ingest_and_search_finds_the_right_document(store, fake_llm):
    make_pipeline(store, fake_llm)
    best = store.search("how many days of annual leave", k=1)[0]
    assert best.source == "handbook.md"


def test_store_lists_and_deletes_sources(store, fake_llm):
    make_pipeline(store, fake_llm)
    assert store.sources() == ["handbook.md", "manual.pdf"]
    store.delete_source("manual.pdf")
    assert store.sources() == ["handbook.md"]


def test_answer_sends_numbered_context_and_returns_cited_source(store):
    llm = FakeLLM(["You get 20 days [1]."])
    pipeline = make_pipeline(store, llm)
    answer = pipeline.answer("How many days of annual leave do I get?")

    prompt = llm.calls[0][-1]["content"]
    assert "[1] (handbook.md)" in prompt
    assert answer.text == "You get 20 days [1]."
    assert [hit.source for hit in answer.sources] == ["handbook.md"]


def test_empty_store_returns_not_found_without_calling_llm(store, fake_llm):
    pipeline = RAGPipeline(store, fake_llm)
    answer = pipeline.answer("Anything?")
    assert answer.text == NOT_FOUND
    assert fake_llm.calls == []


def test_low_similarity_hits_are_filtered_out(store, fake_llm):
    pipeline = make_pipeline(store, fake_llm, min_score=0.99)
    answer = pipeline.answer("completely unrelated question about volcanoes")
    assert answer.text == NOT_FOUND
    assert fake_llm.calls == []


def test_follow_up_question_is_condensed_before_search(store):
    llm = FakeLLM(["How long does the robot battery take to charge?", "2 hours 30 minutes [1]."])
    pipeline = make_pipeline(store, llm)
    history = [
        {"role": "user", "content": "Tell me about the robot battery."},
        {"role": "assistant", "content": "It is a 48 V battery."},
    ]
    answer = pipeline.answer("how long to charge it?", history=history)

    assert answer.query == "How long does the robot battery take to charge?"
    assert answer.sources[0].source == "manual.pdf"
    assert len(llm.calls) == 2


def test_cited_sources_parses_and_deduplicates():
    hits = [Hit("a", "a.md", None, 0.9), Hit("b", "b.pdf", 2, 0.8)]
    assert cited_sources("x [2] y [1][2] z [7]", hits) == [hits[1], hits[0]]


def test_format_context_labels_pages():
    hits = [Hit("text", "b.pdf", 2, 0.8)]
    assert format_context(hits) == "[1] (b.pdf, p. 2)\ntext"
