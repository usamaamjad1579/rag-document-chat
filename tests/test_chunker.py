import pytest

from rag import Page, chunk_pages, split_text

LONG_TEXT = "\n\n".join(
    f"Paragraph {i}. " + " ".join(f"word{i}_{j}" for j in range(40)) for i in range(12)
)


def test_short_text_is_one_chunk():
    assert split_text("Hello world.", chunk_size=100, overlap=10) == ["Hello world."]


def test_empty_text_gives_no_chunks():
    assert split_text("   \n\n  ", chunk_size=100, overlap=10) == []


def test_chunks_respect_size_limit():
    chunks = split_text(LONG_TEXT, chunk_size=300, overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 300 for c in chunks)


def test_neighbouring_chunks_overlap():
    chunks = split_text(LONG_TEXT, chunk_size=300, overlap=80)
    shared = [set(a.split()) & set(b.split()) for a, b in zip(chunks, chunks[1:])]
    assert all(shared), "every pair of neighbouring chunks should share some words"


def test_no_text_is_lost():
    chunks = split_text(LONG_TEXT, chunk_size=300, overlap=50)
    joined = " ".join(chunks)
    for word in LONG_TEXT.split():
        assert word in joined


def test_very_long_word_is_hard_cut():
    chunks = split_text("x" * 250, chunk_size=100, overlap=10)
    assert all(len(c) <= 100 for c in chunks)
    assert "".join(chunks) == "x" * 250


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        split_text("abc", chunk_size=10, overlap=10)


def test_chunk_pages_keeps_source_and_page():
    pages = [Page(text=LONG_TEXT, source="report.pdf", page=3)]
    chunks = chunk_pages(pages, chunk_size=300, overlap=50)
    assert {c.source for c in chunks} == {"report.pdf"}
    assert {c.page for c in chunks} == {3}
    assert len({c.id for c in chunks}) == len(chunks)
