from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from sqlalchemy import select

from .db import SessionLocal
from .models import DocumentChunk
from .llm_qwen import chat_completion
from .settings import settings
from .ingest.scanner import scan_documents


logger = logging.getLogger(__name__)


@dataclass
class Citation:
    path: str
    page: int | None
    snippet: str
    score: float


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def index_documents() -> dict:
    start = datetime.utcnow()
    docs = scan_documents(settings.docs_dir)
    inserted = 0
    skipped = 0
    with SessionLocal() as db:
        for doc in docs:
            for chunk in doc.chunks:
                chunk_hash = _hash_text(chunk.text)
                exists = db.execute(
                    select(DocumentChunk).where(
                        DocumentChunk.file_path == doc.path,
                        DocumentChunk.page == chunk.page,
                        DocumentChunk.chunk_hash == chunk_hash,
                    )
                ).scalar_one_or_none()
                if exists:
                    skipped += 1
                    continue
                db.add(
                    DocumentChunk(
                        file_path=doc.path,
                        page=chunk.page,
                        chunk_text=chunk.text,
                        chunk_hash=chunk_hash,
                    )
                )
                inserted += 1
        db.commit()
    elapsed = (datetime.utcnow() - start).total_seconds()
    logger.info("Index complete. inserted=%s skipped=%s elapsed=%.2fs", inserted, skipped, elapsed)
    return {"inserted": inserted, "skipped": skipped}


def _tokenize(text: str) -> list[str]:
    return [w.lower() for w in text.split() if w.strip()]


def search_chunks(query: str, top_k: int = 5) -> list[Citation]:
    tokens = _tokenize(query)
    if not tokens:
        return []
    with SessionLocal() as db:
        rows = db.execute(select(DocumentChunk)).scalars().all()
    scored: list[Citation] = []
    for row in rows:
        text = row.chunk_text.lower()
        score = sum(text.count(t) for t in tokens)
        if score <= 0:
            continue
        snippet = row.chunk_text[:240]
        scored.append(Citation(path=row.file_path, page=row.page, snippet=snippet, score=float(score)))
    scored.sort(key=lambda x: x.score, reverse=True)
    logger.info("Search hits: %s", len(scored))
    return scored[:top_k]


def rag_answer(question: str, citations: list[Citation]) -> str:
    context_lines = []
    for idx, cite in enumerate(citations, start=1):
        label = f"[{idx}] {os.path.basename(cite.path)} p{cite.page or '-'}"
        context_lines.append(f"{label}: {cite.snippet}")
    context = "\n".join(context_lines)
    system_prompt = (
        "You answer based on provided context with citations. "
        "Cite sources inline like [1], [2]. If not in context, say you don't know."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]
    return chat_completion(messages, temperature=0.2)

