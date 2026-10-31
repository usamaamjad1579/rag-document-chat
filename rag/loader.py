from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from pypdf import PdfReader

SUPPORTED_TYPES = (".pdf", ".txt", ".md")


@dataclass
class Page:
    text: str
    source: str
    page: int | None  # 1-based page number for PDFs, None for plain text


def load_document(name: str, data: bytes) -> list[Page]:
    """Turn an uploaded file into a list of pages with their source name."""
    suffix = Path(name).suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(BytesIO(data))
        return [
            Page(text=page.extract_text() or "", source=name, page=number)
            for number, page in enumerate(reader.pages, start=1)
        ]
    if suffix in (".txt", ".md"):
        return [Page(text=data.decode("utf-8", errors="replace"), source=name, page=None)]
    raise ValueError(f"Unsupported file type '{suffix}'. Use one of: {', '.join(SUPPORTED_TYPES)}")


def load_file(path: str | Path) -> list[Page]:
    path = Path(path)
    return load_document(path.name, path.read_bytes())


def load_folder(folder: str | Path) -> list[Page]:
    pages: list[Page] = []
    for path in sorted(Path(folder).iterdir()):
        if path.suffix.lower() in SUPPORTED_TYPES:
            pages.extend(load_file(path))
    return pages
