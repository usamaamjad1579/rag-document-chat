"""Recursive text splitter.

Splits on the largest natural boundary first (paragraphs, then lines, then
sentences, then words) and packs the pieces into chunks of at most
`chunk_size` characters. Neighbouring chunks share up to `overlap` characters
so that an answer sitting on a chunk boundary is not lost.
"""

import re
from dataclasses import dataclass

from .loader import Page

SEPARATORS = ["\n\n", "\n", ". ", " "]


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    page: int | None


def _clean(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _split_recursive(text: str, size: int, separators: list[str]) -> list[str]:
    if len(text) <= size:
        return [text]
    if not separators:
        return [text[i : i + size] for i in range(0, len(text), size)]

    sep, rest = separators[0], separators[1:]
    parts = text.split(sep)
    pieces: list[str] = []
    for i, part in enumerate(parts):
        if i < len(parts) - 1:
            part += sep  # keep the separator so chunks read naturally
        if not part:
            continue
        if len(part) > size:
            pieces.extend(_split_recursive(part, size, rest))
        else:
            pieces.append(part)
    return pieces


def _merge(pieces: list[str], size: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    length = 0
    for piece in pieces:
        if current and length + len(piece) > size:
            chunks.append("".join(current).strip())
            # Drop pieces from the front until only the overlap tail is left
            # and the next piece fits.
            while current and (length > overlap or length + len(piece) > size):
                length -= len(current.pop(0))
        current.append(piece)
        length += len(piece)
    if current:
        chunks.append("".join(current).strip())
    return [c for c in chunks if c]


def split_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    text = _clean(text)
    if not text:
        return []
    return _merge(_split_recursive(text, chunk_size, SEPARATORS), chunk_size, overlap)


def chunk_pages(pages: list[Page], chunk_size: int = 800, overlap: int = 150) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page in pages:
        for i, text in enumerate(split_text(page.text, chunk_size, overlap)):
            chunk_id = f"{page.source}:p{page.page or 0}:c{i}"
            chunks.append(Chunk(id=chunk_id, text=text, source=page.source, page=page.page))
    return chunks
