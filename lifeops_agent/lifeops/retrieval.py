from __future__ import annotations

"""lifeops.retrieval

这个模块是知识库检索层的主入口，负责两件大事：
1) `index_documents`：把本地文件内容切块并写入索引（SQLite + 可选向量索引）。
2) `search_chunks`：根据问题召回候选文本块，供 RAG 回答使用。

支持的检索后端：
- bm25：传统关键词检索（默认、稳定）
- langchain：向量检索 + 可选重排
- hybrid：多路召回融合（path + bm25 + rewrite + vector）
"""

import hashlib
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone

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
    """返回给上层的可引用证据片段。"""

    path: str
    page: int | None
    snippet: str
    score: float
    reason: str = ""  # 命中原因（bm25 / path / rewrite-bm25 / vector 等），便于可审计


def _hash_text(text: str) -> str:
    """对 chunk 文本做稳定哈希，用于去重判断。"""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _index_documents_sqlite(*, rebuild: bool) -> tuple[int, int]:
    """执行 SQLite 侧的文本索引构建（路径+页码+文本哈希去重）。"""

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

    return inserted, skipped


def index_documents(*, rebuild: bool = False) -> dict:
    """统一索引入口：先建 SQLite 索引，再按配置可选构建向量索引。"""

    start = datetime.now(timezone.utc)
    inserted, skipped = _index_documents_sqlite(rebuild=rebuild)

    backend = settings.rag_backend.lower().strip()
    result: dict = {
        "inserted": inserted,
        "skipped": skipped,
        "rebuild": rebuild,
        "backend": backend,
        "vector_attempted": backend in {"langchain", "hybrid"},
    }

    if backend in {"langchain", "hybrid"}:
        try:
            from .rag_langchain import build_vector_index

            vec = build_vector_index(root_dir=settings.docs_dir, rebuild=rebuild)
            result.update(vec)
        except Exception as exc:
            logger.warning("Vector index build failed, keep sqlite index only: %s", exc)
            result["vector_error"] = str(exc)

    elapsed = (datetime.now(timezone.utc) - start).total_seconds()
    logger.info(
        "Index complete. inserted=%s skipped=%s elapsed=%.2fs rebuild=%s backend=%s",
        inserted,
        skipped,
        elapsed,
        rebuild,
        backend,
    )
    return result


_WORD_RE = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    """简易分词：英文按单词、中文按连续汉字片段（MVP）。"""

    return [m.group(0).lower() for m in _WORD_RE.finditer(text) if m.group(0).strip()]


def _extract_filename_hints(query: str) -> list[str]:
    """从问题中抽取文件名线索（如 work.png / report.pdf）。"""

    hints: set[str] = set()
    for m in re.finditer(r"[A-Za-z0-9_.-]+\.(?:png|pdf|txt|md|docx)", query, flags=re.IGNORECASE):
        hints.add(m.group(0))
    return list(hints)


def _rewrite_query(query: str) -> str:
    """用 LLM 改写 query，提升召回对“口语问法”的鲁棒性。"""

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
    """从数据库取出全部 chunk（MVP 实现，后续可替换为更高效索引）。"""

    with SessionLocal() as db:
        return db.execute(select(DocumentChunk)).scalars().all()


def _path_recall(rows: list[DocumentChunk], query: str, top_k: int) -> list[Citation]:
    """路径/文件名召回：解决“某文件里有什么”这类问题。"""

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
    """BM25 召回：基于词项统计的传统相关性检索。"""

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
    """简单融合：按 key 去重，保留高分与更优 reason。"""

    merged: dict[tuple[str, int | None, str], Citation] = {}
    for l in lists:
        for c in l:
            key = (c.path, c.page, c.snippet)
            if key not in merged:
                merged[key] = c
            else:
                if c.score > merged[key].score:
                    merged[key].score = c.score
                if merged[key].reason != "path" and c.reason == "path":
                    merged[key].reason = "path"
                elif merged[key].reason == "bm25" and c.reason == "rewrite-bm25":
                    merged[key].reason = "rewrite-bm25"

    out = list(merged.values())
    out.sort(key=lambda x: x.score, reverse=True)
    return out[:top_k]


def _rrf_fuse(*lists: list[Citation], top_k: int, k: int = 60) -> list[Citation]:
    """RRF 融合：对不同召回通道做稳健排序融合。"""

    fused: dict[tuple[str, int | None, str], Citation] = {}
    rrf_score: dict[tuple[str, int | None, str], float] = {}

    for l in lists:
        for rank, c in enumerate(l, start=1):
            key = (c.path, c.page, c.snippet)
            if key not in fused:
                fused[key] = Citation(path=c.path, page=c.page, snippet=c.snippet, score=0.0, reason=c.reason)
            else:
                if c.reason not in fused[key].reason:
                    fused[key].reason = f"{fused[key].reason}+{c.reason}"
            rrf_score[key] = rrf_score.get(key, 0.0) + (1.0 / (k + rank))

    for key, score in rrf_score.items():
        fused[key].score = score

    out = list(fused.values())
    out.sort(key=lambda x: x.score, reverse=True)
    return out[:top_k]


def _search_langchain(query: str, top_k: int) -> list[Citation]:
    """调用向量后端检索（失败时返回空列表，不抛异常中断）。"""

    try:
        from .rag_langchain import search_with_rerank
    except Exception as exc:
        logger.warning("langchain backend import failed: %s", exc)
        return []

    try:
        hits = search_with_rerank(query, top_k=top_k, candidate_k=settings.rag_candidate_k)
        return [
            Citation(path=h.path, page=h.page, snippet=h.snippet, score=h.score, reason=h.reason)
            for h in hits
        ]
    except Exception as exc:
        logger.warning("langchain retrieval failed: %s", exc)
        return []


def search_chunks(query: str, top_k: int = 5) -> list[Citation]:
    """统一检索入口：按 backend 分流到 bm25 / langchain / hybrid。"""

    backend = settings.rag_backend.lower().strip()
    top_k = top_k or settings.rag_top_k

    rows = _collect_chunks()
    if not rows and backend in {"bm25", "hybrid"}:
        return []

    rewritten = _rewrite_query(query)

    path_hits = _path_recall(rows, query, top_k=max(top_k, 8)) if rows else []
    bm25_hits = _bm25_recall(rows, query, top_k=max(top_k, 8), reason="bm25") if rows else []
    rewrite_hits = _bm25_recall(rows, rewritten, top_k=max(top_k, 8), reason="rewrite-bm25") if rows else []

    if backend == "langchain":
        vector_hits = _search_langchain(query, top_k=max(top_k, settings.rag_candidate_k))
        if vector_hits:
            logger.info("Search backend=langchain hits=%s", len(vector_hits))
            return vector_hits[:top_k]
        return _merge_rank(path_hits, rewrite_hits, bm25_hits, top_k=top_k)

    if backend == "hybrid":
        vector_hits = _search_langchain(query, top_k=max(top_k, settings.rag_candidate_k))
        merged = _rrf_fuse(path_hits, rewrite_hits, bm25_hits, vector_hits, top_k=top_k)
        logger.info(
            "Search backend=hybrid merged=%s (path=%s bm25=%s rewrite=%s vector=%s rewritten=%r)",
            len(merged),
            len(path_hits),
            len(bm25_hits),
            len(rewrite_hits),
            len(vector_hits),
            rewritten,
        )
        return merged

    merged = _merge_rank(path_hits, rewrite_hits, bm25_hits, top_k=top_k)
    logger.info(
        "Search backend=bm25 merged=%s (path=%s bm25=%s rewrite=%s rewritten=%r)",
        len(merged),
        len(path_hits),
        len(bm25_hits),
        len(rewrite_hits),
        rewritten,
    )
    return merged


def rag_answer(question: str, citations: list[Citation]) -> str:
    """把候选证据拼成 Context，交给 LLM 生成带引用回答。"""

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
