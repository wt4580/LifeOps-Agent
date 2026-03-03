from __future__ import annotations

import os
from dataclasses import dataclass

from .pdf_text import extract_pdf
from .txt_text import extract_txt
from .ocr_png import extract_png


@dataclass
class TextChunk:
    text: str
    page: int | None = None


@dataclass
class Document:
    path: str
    chunks: list[TextChunk]


def _chunk_text(text: str, size: int = 800) -> list[str]:
    text = text.strip()
    if not text:
        return []
    parts = []
    start = 0
    while start < len(text):
        parts.append(text[start : start + size])
        start += size
    return parts


def scan_documents(root_dir: str) -> list[Document]:
    documents: list[Document] = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            path = os.path.join(dirpath, filename)
            ext = os.path.splitext(filename)[1].lower()
            if ext == ".pdf":
                pages = extract_pdf(path)
                chunks = [TextChunk(text=p.text, page=p.page) for p in pages if p.text]
            elif ext == ".txt":
                text = extract_txt(path)
                chunks = [TextChunk(text=chunk) for chunk in _chunk_text(text)]
            elif ext == ".png":
                text = extract_png(path)
                chunks = [TextChunk(text=chunk) for chunk in _chunk_text(text)]
            else:
                continue
            documents.append(Document(path=path, chunks=chunks))
    return documents

