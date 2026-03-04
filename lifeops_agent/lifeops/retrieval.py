from __future__ import annotations

import hashlib
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import select
from rank_bm25 import BM25Okapi

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
    reason: str = ""  # 命中原因（bm25 / path / rewrite-bm25 等），便于可审计


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def index_documents(*, rebuild: bool = False) -> dict:
    start = datetime.utcnow()
    docs = scan_documents(settings.docs_dir)
    inserted = 0
    skipped = 0

    with SessionLocal() as db:
        if rebuild:
            deleted = db.query(DocumentChunk).delete()
            db.commit()
            logger.info("Rebuild index: cleared document_chunks deleted=%s", deleted)

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
    logger.info("Index complete. inserted=%s skipped=%s elapsed=%.2fs rebuild=%s", inserted, skipped, elapsed, rebuild)
    return {"inserted": inserted, "skipped": skipped, "rebuild": rebuild}


_WORD_RE = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    """简易分词：英文按单词、中文按连续汉字片段（MVP）。"""

    return [m.group(0).lower() for m in _WORD_RE.finditer(text) if m.group(0).strip()]


def _extract_filename_hints(query: str) -> list[str]:
    """从问题中抽取可能的文件名线索。

    支持：xxx.png / xxx.pdf / xxx.txt 或者带路径片段。
    """

    hints: set[str] = set()
    for m in re.finditer(r"[A-Za-z0-9_\-\.]+\.(?:png|pdf|txt)", query, flags=re.IGNORECASE):
        hints.add(m.group(0))
    return list(hints)


def _rewrite_query(query: str) -> str:
    """LLM Query Rewrite：把用户问法改写为适合检索的关键词串。"""

    prompt = (
        "你是检索查询改写器（query rewrite）。\n"
        "目标：把用户问题改写成更适合从本地文档中检索的关键词/短语。\n"
        "规则：\n"
        "- 输出必须只是一行关键词，用空格分隔；不要解释，不要换行。\n"
        "- 保留专有名词、缩写、阶段名（如 stage2）、人名、术语。\n"
        "- 如果问题包含文件名（如 work.png），请保留 work 这样的主体词，同时也保留 work.png 作为线索。\n"
        "- 删除无意义词：如 里面、什么、一下、帮我、请问 等。"
    )
    try:
        text = chat_completion(
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": query},
            ],
            temperature=0.0,
        )
        return " ".join(text.split())
    except Exception as exc:
        logger.warning("query rewrite failed: %s", exc)
        return query


def _collect_chunks() -> list[DocumentChunk]:
    with SessionLocal() as db:
        return db.execute(select(DocumentChunk)).scalars().all()


def _path_recall(rows: list[DocumentChunk], query: str, top_k: int) -> list[Citation]:
    """路径/文件名召回：当用户明确提到文件名时，优先取该文件的 chunks。"""

    hints = _extract_filename_hints(query)
    if not hints:
        return []

    scored: list[Citation] = []
    q_lower = query.lower()

    for r in rows:
        path_lower = (r.file_path or "").lower()
        base_lower = os.path.basename(r.file_path or "").lower()

        hit = False
        score = 0.0
        for h in hints:
            hl = h.lower()
            if hl in path_lower or hl == base_lower:
                hit = True
                score += 10.0

        # 额外：如果 query 里包含去掉扩展名的主体词，也加一点分
        for h in hints:
            stem = os.path.splitext(h)[0].lower()
            if stem and stem in q_lower and stem in path_lower:
                hit = True
                score += 2.0

        if not hit:
            continue

        snippet = (r.chunk_text or "")[:240]
        scored.append(Citation(path=r.file_path, page=r.page, snippet=snippet, score=score, reason="path"))

    scored.sort(key=lambda x: x.score, reverse=True)
    return scored[:top_k]


def _bm25_recall(rows: list[DocumentChunk], query: str, top_k: int, reason: str) -> list[Citation]:
    tokens = _tokenize(query)
    if not tokens:
        return []

    corpus_tokens = [_tokenize(r.chunk_text or "") for r in rows]
    bm25 = BM25Okapi(corpus_tokens)
    scores = bm25.get_scores(tokens)

    citations: list[Citation] = []
    for r, s in zip(rows, scores):
        if s <= 0:
            continue
        citations.append(
            Citation(
                path=r.file_path,
                page=r.page,
                snippet=(r.chunk_text or "")[:240],
                score=float(s),
                reason=reason,
            )
        )

    citations.sort(key=lambda c: c.score, reverse=True)
    return citations[:top_k]


def _merge_rank(*lists: list[Citation], top_k: int) -> list[Citation]:
    """合并多路召回结果：同一 (path,page,snippet) 视为同一条，score 取 max，并保留最高优先级的 reason。"""

    merged: dict[tuple[str, int | None, str], Citation] = {}
    for l in lists:
        for c in l:
            key = (c.path, c.page, c.snippet)
            if key not in merged:
                merged[key] = c
            else:
                if c.score > merged[key].score:
                    merged[key].score = c.score
                # reason 优先保留 path，其次 rewrite-bm25，其次 bm25
                if merged[key].reason != "path" and c.reason == "path":
                    merged[key].reason = "path"
                elif merged[key].reason == "bm25" and c.reason == "rewrite-bm25":
                    merged[key].reason = "rewrite-bm25"

    out = list(merged.values())
    out.sort(key=lambda x: x.score, reverse=True)
    return out[:top_k]


def search_chunks(query: str, top_k: int = 5) -> list[Citation]:
    """BM25 + 三路召回：

    1) 路径/文件名召回（解决“work.png 里有什么”这种问法）
    2) BM25 直接用原问题
    3) LLM query rewrite 后再 BM25

    最后合并排序，返回 TopK。
    """

    rows = _collect_chunks()
    if not rows:
        return []

    rewritten = _rewrite_query(query)

    path_hits = _path_recall(rows, query, top_k=max(top_k, 8))
    bm25_hits = _bm25_recall(rows, query, top_k=max(top_k, 8), reason="bm25")
    rewrite_hits = _bm25_recall(rows, rewritten, top_k=max(top_k, 8), reason="rewrite-bm25")

    merged = _merge_rank(path_hits, rewrite_hits, bm25_hits, top_k=top_k)

    logger.info(
        "Search hits merged=%s (path=%s bm25=%s rewrite=%s rewritten=%r)",
        len(merged),
        len(path_hits),
        len(bm25_hits),
        len(rewrite_hits),
        rewritten,
    )

    return merged


def rag_answer(question: str, citations: list[Citation]) -> str:
    context_lines: list[str] = []
    for idx, cite in enumerate(citations, start=1):
        label = f"[{idx}] {os.path.basename(cite.path)} p{cite.page or '-'} ({cite.reason})"
        context_lines.append(f"{label}: {cite.snippet}")
    context = "\n".join(context_lines)

    system_prompt = (
        "你是一个严谨的知识库问答助手。你只能根据提供的 Context 回答，并在句子中内联标注引用 [1][2]。\n"
        "如果 Context 不足以回答，请明确说不知道，不要编造。"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]
    return chat_completion(messages, temperature=0.2)
