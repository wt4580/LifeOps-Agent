from __future__ import annotations

import os
import re
from dataclasses import dataclass

from .pdf_text import extract_pdf
from .txt_text import extract_txt
from .ocr_png import extract_png


@dataclass
class TextChunk:
    text: str
    page: int | None = None
    section_title: str | None = None
    heading_level: int | None = None
    is_table: bool = False


@dataclass
class Document:
    path: str
    chunks: list[TextChunk]


def _is_heading_line(line: str) -> tuple[str, int] | None:
    t = line.strip()
    if not t:
        return None
    if re.match(r"^第[一二三四五六七八九十百千]+[章节篇条部分]", t):
        return (t, 1)
    if re.match(r"^[一二三四五六七八九十百千]+[、．\.\s]", t):
        return (t, 2)
    if re.match(r"^\d+(?:\.\d+)*[\.\s]", t):
        return (t, 2)
    if re.match(r"^[（(][一二三四五六七八九十\d]+[)）]", t):
        return (t, 3)
    if t.isupper() and len(t) > 2:
        return (t, 1)
    if t == t.upper() and any(c in t for c in "（）()"):
        return (t, 1)
    if len(t) <= 30 and any(kw in t for kw in ["章", "节", "篇", "条", "目", "录", "前言", "引言", "概述", "附录", "参考", "结论", "总结"]):
        return (t, 1)
    return None


def _is_table_block(text: str) -> bool:
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if len(lines) < 2:
        return False
    pipe_count = sum(1 for l in lines if l.startswith("|") or "|" in l)
    return pipe_count >= len(lines) * 0.6


def _smart_split(text: str, max_size: int = 800, overlap: int = 200) -> list[str]:
    if not text.strip():
        return []
    text = text.strip()
    if len(text) <= max_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_size, len(text))
        if end < len(text):
            boundary = text.rfind("\n\n", start + int(overlap * 0.5), end)
            if boundary == -1 or boundary <= start:
                boundary = text.rfind("\n", start + int(overlap * 0.5), end)
            if boundary == -1 or boundary <= start:
                boundary = text.rfind("。", start + int(overlap * 0.5), end)
            if boundary == -1 or boundary <= start:
                boundary = end
            else:
                boundary += 1
            chunks.append(text[start:boundary])
            start = max(boundary - overlap, start + 1)
        else:
            chunks.append(text[start:])
            break
    if not chunks:
        chunks = [text]
    return chunks


def _build_pdf_chunks(path: str) -> list[TextChunk]:
    pages = extract_pdf(path)
    if not pages:
        return []

    raw_chunks: list[TextChunk] = []
    current_section: tuple[str, int] | None = None

    for p in pages:
        content = (p.text or "").strip()
        if not content:
            continue

        lines = content.split("\n")
        page_heading = None
        body_start = 0
        for i, line in enumerate(lines):
            heading = _is_heading_line(line)
            if heading:
                page_heading = heading
                body_start = i + 1
                current_section = heading
                break

        body = "\n".join(lines[body_start:]).strip()
        if not body:
            if page_heading:
                raw_chunks.append(TextChunk(
                    text=page_heading[0], page=p.page,
                    section_title=page_heading[0], heading_level=page_heading[1],
                ))
            continue

        parts = _smart_split(body) if len(body) > 800 else [body]
        for part in parts:
            is_table = _is_table_block(part)
            title = current_section[0] if current_section else None
            level = current_section[1] if current_section else None
            raw_chunks.append(TextChunk(
                text=part, page=p.page,
                section_title=title, heading_level=level,
                is_table=is_table,
            ))

    merged: list[TextChunk] = []
    for c in raw_chunks:
        if merged and merged[-1].page == c.page and c.is_table:
            merged[-1].text += "\n" + c.text
            merged[-1].is_table = True
        else:
            merged.append(c)

    return merged


def _build_txt_chunks(path: str) -> list[TextChunk]:
    text = extract_txt(path)
    if not text.strip():
        return []

    lines = text.split("\n")
    chunks: list[TextChunk] = []
    current_section: tuple[str, int] | None = None
    buffer_lines: list[str] = []
    buffer_len = 0

    def flush_buffer():
        nonlocal buffer_lines, buffer_len
        if not buffer_lines:
            return
        block = "\n".join(buffer_lines).strip()
        if not block:
            return
        title = current_section[0] if current_section else None
        level = current_section[1] if current_section else None
        is_table = _is_table_block(block)
        if len(block) <= 800:
            chunks.append(TextChunk(text=block, section_title=title, heading_level=level, is_table=is_table))
        else:
            for part in _smart_split(block):
                chunks.append(TextChunk(text=part, section_title=title, heading_level=level, is_table=is_table))
        buffer_lines = []
        buffer_len = 0

    for line in lines:
        heading = _is_heading_line(line)
        if heading:
            flush_buffer()
            current_section = heading
            continue
        stripped = line.strip()
        if not stripped and buffer_len > 600:
            flush_buffer()
            continue
        buffer_lines.append(line)
        buffer_len += len(line) + 1

    flush_buffer()
    return chunks or [TextChunk(text=text)]


def _build_png_chunks(path: str) -> list[TextChunk]:
    text = extract_png(path)
    if not text.strip():
        return []
    parts = _smart_split(text) if len(text) > 800 else [text]
    return [TextChunk(text=p) for p in parts]


def scan_documents(root_dir: str) -> list[Document]:
    documents: list[Document] = []

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            path = os.path.join(dirpath, filename)
            ext = os.path.splitext(filename)[1].lower()

            if ext == ".pdf":
                chunks = _build_pdf_chunks(path)
            elif ext == ".txt":
                chunks = _build_txt_chunks(path)
            elif ext == ".png":
                chunks = _build_png_chunks(path)
            else:
                continue

            documents.append(Document(path=path, chunks=chunks))

    return documents
