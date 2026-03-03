from __future__ import annotations

from dataclasses import dataclass
from pypdf import PdfReader


@dataclass
class PageText:
    page: int
    text: str


def extract_pdf(path: str) -> list[PageText]:
    reader = PdfReader(path)
    pages: list[PageText] = []
    for idx, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append(PageText(page=idx, text=text))
    return pages

