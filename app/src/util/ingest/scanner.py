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
    section_path: str | None = None
    section_rank: int | None = None
    is_table: bool = False


@dataclass
class HeadingNode:
    title: str
    level: int


class HeadingStack:
    def __init__(self):
        self._stack: list[HeadingNode] = []
        self._section_count: int = 0

    def update(self, title: str, level: int) -> None:
        while self._stack and self._stack[-1].level >= level:
            self._stack.pop()
        self._stack.append(HeadingNode(title=title, level=level))
        self._section_count = 0

    @property
    def path(self) -> str:
        return " > ".join(n.title for n in self._stack) if self._stack else ""

    @property
    def current_title(self) -> str | None:
        return self._stack[-1].title if self._stack else None

    @property
    def current_level(self) -> int | None:
        return self._stack[-1].level if self._stack else None

    def bump_section_rank(self) -> int:
        self._section_count += 1
        return self._section_count


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
    if re.match(r"^\d{1,2}(?:\.\d+)*[\.\s](?!\d)", t):
        return (t, 2)
    if re.match(r"^[（(][一二三四五六七八九十\d]+[)）]", t):
        return (t, 3)
    if t.isupper() and len(t) > 2 and not re.search('[\u4e00-\u9fff]', t) and t[0].isalpha():
        return (t, 1)
    if len(t) <= 30 and any(t.startswith(kw) for kw in ["章", "节", "篇", "条", "前言", "引言", "概述", "附录", "参考", "结论", "总结"]):
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
                boundary = text.rfind("。", start + int(overlap * 0.5), end)
            if boundary == -1 or boundary <= start:
                boundary = text.rfind("\n", start + int(overlap * 0.5), end)
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

    lines_with_pages: list[tuple[str, int]] = []
    for p in pages:
        text = (p.text or "").strip()
        if not text:
            continue
        for line in text.split("\n"):
            lines_with_pages.append((line, p.page))
    if not lines_with_pages:
        return []

    chunks: list[TextChunk] = []
    heading_stack = HeadingStack()
    buffer_lines: list[tuple[str, int]] = []
    buffer_len = 0

    def flush_buffer():
        nonlocal buffer_lines, buffer_len
        if not buffer_lines:
            return
        block = "\n".join(l for l, _ in buffer_lines).strip()
        if not block:
            buffer_lines = []
            buffer_len = 0
            return
        section_rank = heading_stack.bump_section_rank()
        is_table = _is_table_block(block)
        page = buffer_lines[0][1]
        if len(block) <= 800:
            chunks.append(TextChunk(
                text=block, page=page,
                section_title=heading_stack.current_title,
                heading_level=heading_stack.current_level,
                section_path=heading_stack.path,
                section_rank=section_rank,
                is_table=is_table,
            ))
        else:
            for part in _smart_split(block):
                sr = heading_stack.bump_section_rank()
                chunks.append(TextChunk(
                    text=part, page=page,
                    section_title=heading_stack.current_title,
                    heading_level=heading_stack.current_level,
                    section_path=heading_stack.path,
                    section_rank=sr,
                    is_table=is_table,
                ))
        buffer_lines = []
        buffer_len = 0

    for line, page in lines_with_pages:
        heading = _is_heading_line(line)
        if heading:
            if re.search(r'\.{4,}', line):
                stripped = line.strip()
                if not stripped and buffer_len > 600:
                    flush_buffer()
                    continue
                buffer_lines.append((line, page))
                buffer_len += len(line) + 1
                continue
            flush_buffer()
            heading_stack.update(title=heading[0], level=heading[1])
            continue
        stripped = line.strip()
        if not stripped and buffer_len > 600:
            flush_buffer()
            continue
        buffer_lines.append((line, page))
        buffer_len += len(line) + 1

    flush_buffer()
    return chunks or [TextChunk(text="\n".join(l for l, _ in lines_with_pages))]


def _build_txt_chunks(path: str) -> list[TextChunk]:
    text = extract_txt(path)
    if not text.strip():
        return []

    lines = text.split("\n")
    chunks: list[TextChunk] = []
    heading_stack = HeadingStack()
    buffer_lines: list[str] = []
    buffer_len = 0

    def flush_buffer():
        nonlocal buffer_lines, buffer_len
        if not buffer_lines:
            return
        block = "\n".join(buffer_lines).strip()
        if not block:
            return
        section_rank = heading_stack.bump_section_rank()
        is_table = _is_table_block(block)
        if len(block) <= 800:
            chunks.append(TextChunk(
                text=block,
                section_title=heading_stack.current_title,
                heading_level=heading_stack.current_level,
                section_path=heading_stack.path,
                section_rank=section_rank,
                is_table=is_table,
            ))
        else:
            for part in _smart_split(block):
                sr = heading_stack.bump_section_rank()
                chunks.append(TextChunk(
                    text=part,
                    section_title=heading_stack.current_title,
                    heading_level=heading_stack.current_level,
                    section_path=heading_stack.path,
                    section_rank=sr,
                    is_table=is_table,
                ))
        buffer_lines = []
        buffer_len = 0

    for line in lines:
        heading = _is_heading_line(line)
        if heading:
            flush_buffer()
            heading_stack.update(title=heading[0], level=heading[1])
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
