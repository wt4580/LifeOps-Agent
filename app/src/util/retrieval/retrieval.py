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
import json
import os
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import func, select
from rank_bm25 import BM25Okapi

from ...common.config.log_config import logger
from ...common.config.knowledge_db_config import KnowledgeSessionLocal
from ...domain.entity.knowledge_entity import Document, DocumentChunk
from ...common.config.llm_config import chat_completion
from ...common.config.base_config import settings
from ..ingest.scanner import scan_documents


@dataclass
class Citation:
    """返回给上层的可引用证据片段。"""

    path: str
    page: int | None
    snippet: str
    score: float
    reason: str = ""
    source_type: str | None = None
    doc_topic: str | None = None
    section_title: str | None = None
    section_path: str | None = None
    summary: str | None = None
    chunk_id: int | None = None
    relatedness: str = "low"


# ---------------------------------------------------------------------------
# 相关度得分约定（统一 0-100 尺度）
# - 各召回通道的原始分规范化为 0-100，语义 = "该证据与问题的相关度得分"。
# - 强/弱/低参考阈值见 base_config（rag_score_strong=70 / rag_score_weak=30）。
# - 向量/重排通道最可信（×0.85）；BM25 词面匹配次之；文件名命中最弱（封顶低分）。
# ---------------------------------------------------------------------------
_VECTOR_SCORE_MAX = 85.0   # 向量 1/(1+d)∈(0,1] 映射上限
_BM25_SCORE_MAX = 60.0     # BM25 词面得分映射上限（弱于向量）
_PATH_HIT_SCORE = 25.0     # 文件名完整命中（低于 weak 阈值，属低参考）
_PATH_STEM_SCORE = 15.0    # 文件名词干命中
_PATH_SCORE_CAP = 28.0     # 文件名通道单条封顶（永不超过 weak 阈值，杜绝假强相关）


def _classes_score(value: float) -> str:
    """由 0-100 得分给出强/弱/低相关标签。"""
    strong = getattr(settings, "rag_score_strong", 70)
    weak = getattr(settings, "rag_score_weak", 30)
    if value >= strong:
        return "strong"
    if value >= weak:
        return "weak"
    return "low"


def _norm_vector(score: float) -> float:
    """向量/重排通道映射到 0-100：原始分 clamp 到 [0,1] 后 ×85（封顶）。"""
    if score is None:
        return 0.0
    try:
        clamped = min(max(float(score), 0.0), 1.0)
    except (TypeError, ValueError):
        clamped = 0.0
    return clamped * _VECTOR_SCORE_MAX


def _hash_text(text: str) -> str:
    """对 chunk 文本做稳定哈希，用于去重判断。"""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _infer_source_type(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return "pdf"
    if ext == ".txt":
        return "txt"
    if ext == ".png":
        return "png"
    return "other"


def _infer_doc_topic(path: str) -> str:
    lower = path.lower()
    if any(k in lower for k in ["指南", "guide", "manual", "手册", "规范", "标准"]):
        return "guideline"
    if any(k in lower for k in ["菜谱", "食谱", "recipe", "meal"]):
        return "recipe"
    if any(k in lower for k in ["简历", "resume", "cv"]):
        return "resume"
    if any(k in lower for k in ["日记", "日志", "log", "memo", "记录"]):
        return "personal_log"
    if any(k in lower for k in ["论文", "paper", "study", "研究"]):
        return "paper"
    return "other"


def _index_documents_sqlite(*, rebuild: bool) -> tuple[int, int]:
    """执行 SQLite 侧的文本索引构建（路径+页码+文本哈希去重）。"""

    docs = scan_documents(settings.docs_dir)
    inserted = 0
    skipped = 0

    with KnowledgeSessionLocal() as db:
        if rebuild:
            db.query(Document).delete()
            deleted = db.query(DocumentChunk).delete()
            db.commit()
            logger.info("Rebuild index: cleared documents + document_chunks deleted=%s", deleted)

        for doc in docs:
            source_type = _infer_source_type(doc.path)
            doc_topic = _infer_doc_topic(doc.path)
            for chunk in doc.chunks:
                chunk_hash = _hash_text(chunk.text)
                exists = db.execute(
                    select(DocumentChunk).where(
                        DocumentChunk.file_path == doc.path,
                        DocumentChunk.page == chunk.page,
                        DocumentChunk.chunk_hash == chunk_hash,
                    ).limit(1)
                ).scalars().first()
                if exists:
                    skipped += 1
                    continue
                db.add(
                    DocumentChunk(
                        file_path=doc.path,
                        page=chunk.page,
                        chunk_text=chunk.text,
                        chunk_hash=chunk_hash,
                        source_type=source_type,
                        doc_topic=doc_topic,
                        section_title=chunk.section_title,
                        heading_level=chunk.heading_level,
                        section_path=chunk.section_path,
                        section_rank=chunk.section_rank,
                        is_table=chunk.is_table,
                    )
                )
                inserted += 1
        db.commit()

    return inserted, skipped


def _migrate_section_paths() -> None:
    """给已有 chunks 回填 section_path 和 section_rank（增量迁移用）。"""
    from ..ingest.scanner import HeadingStack

    with KnowledgeSessionLocal() as db:
        file_paths = db.execute(
            select(DocumentChunk.file_path).where(DocumentChunk.section_path == None).distinct()
        ).scalars().all()
        for fp in file_paths:
            chunks = db.execute(
                select(DocumentChunk).where(
                    DocumentChunk.file_path == fp,
                    DocumentChunk.section_path == None,
                ).order_by(DocumentChunk.id)
            ).scalars().all()
            if not chunks:
                continue
            heading_stack = HeadingStack()
            for c in chunks:
                if c.section_title and c.heading_level is not None:
                    heading_stack.update(title=c.section_title, level=c.heading_level)
                c.section_path = heading_stack.path
                c.section_rank = heading_stack.bump_section_rank()
        db.commit()
        if file_paths:
            logger.info("Migrated section_path for %s files", len(file_paths))


def _sync_document_records() -> None:
    """从 document_chunks 聚合数据，创建或更新 documents 表记录。"""
    with KnowledgeSessionLocal() as db:
        file_rows = db.execute(
            select(
                DocumentChunk.file_path,
                DocumentChunk.source_type,
                DocumentChunk.doc_topic,
                DocumentChunk.section_title,
                DocumentChunk.heading_level,
                DocumentChunk.section_path,
                func.count(DocumentChunk.id),
            ).group_by(DocumentChunk.file_path)
        ).all()

        for (fp, src_type, topic, _sec_title, _heading_lv, _sec_path, cnt) in file_rows:
            existing = db.execute(
                select(Document).where(Document.file_path == fp).limit(1)
            ).scalars().first()

            heading_rows = db.execute(
                select(
                    DocumentChunk.section_title,
                    DocumentChunk.heading_level,
                ).where(
                    DocumentChunk.file_path == fp,
                    DocumentChunk.section_title != None,
                ).distinct().order_by(DocumentChunk.id)
            ).all()
            outline = [{"title": t, "level": lv} for t, lv in heading_rows if t] if heading_rows else None

            if existing:
                existing.chunk_count = cnt
                existing.outline = json.dumps(outline, ensure_ascii=False) if outline else None
                existing.updated_at = datetime.now(timezone.utc)
            else:
                db.add(Document(
                    file_path=fp,
                    source_type=src_type or "other",
                    topics=topic or "other",
                    outline=json.dumps(outline, ensure_ascii=False) if outline else None,
                    chunk_count=cnt,
                ))
        db.commit()


def _generate_document_summaries() -> None:
    """对缺少摘要的 Document 记录逐批生成文件级摘要（LLM 生成），不限量"""
    with KnowledgeSessionLocal() as db:
        docs = db.execute(
            select(Document).where(Document.summary == None)
            .order_by(Document.id)
        ).scalars().all()
        if not docs:
            return

        batch_size = 10
        total_updated = 0
        for i in range(0, len(docs), batch_size):
            for doc in docs[i:i + batch_size]:
                chunks = db.execute(
                    select(DocumentChunk.chunk_text, DocumentChunk.summary)
                    .where(DocumentChunk.file_path == doc.file_path)
                    .order_by(DocumentChunk.id)
                    .limit(50)
                ).all()
                if not chunks:
                    continue
                snippet = " | ".join(
                    (s or t)[:200] for t, s in chunks[:5]
                )
                try:
                    answer = chat_completion(
                        [{"role": "user", "content": (
                            f"为以下知识文档生成一句话摘要（30字以内）：\n"
                            f"文件名：{os.path.basename(doc.file_path)}\n"
                            f"内容片段：{snippet}"
                        )}],
                        temperature=0.0,
                    ).strip()[:120]
                    if answer:
                        doc.summary = answer
                        total_updated += 1
                except Exception as exc:
                    logger.warning("Document summary failed for %s: %s", doc.file_path, exc)
        db.commit()
        logger.info("Generated document summaries total=%s", total_updated)


def get_available_knowledge_topics() -> list[dict]:
    with KnowledgeSessionLocal() as db:
        docs = db.execute(
            select(Document).order_by(Document.file_path).limit(100)
        ).scalars().all()
        result = []
        for d in docs:
            entry = {"file": os.path.basename(d.file_path), "topic": d.topics or "other"}
            if d.summary:
                entry["summary"] = d.summary
            if d.outline:
                try:
                    parsed = json.loads(d.outline)
                    entry["sections"] = [h["title"] for h in parsed if h.get("title")][:10]
                except Exception:
                    pass
            result.append(entry)
        return result


def get_document_outline() -> list[dict]:
    with KnowledgeSessionLocal() as db:
        docs = db.execute(
            select(Document).order_by(Document.file_path).limit(100)
        ).scalars().all()
        result = []
        for d in docs:
            headings = []
            if d.outline:
                try:
                    parsed = json.loads(d.outline)
                    headings = parsed[:20]
                except Exception:
                    pass
            result.append({
                "file": os.path.basename(d.file_path),
                "headings": headings,
            })
        return result


def index_documents(*, rebuild: bool = False, reset: bool = False) -> dict:
    """统一索引入口：先建 SQLite 索引，再按配置可选构建向量索引。
    
    Args:
        rebuild: 重建索引（清空再重录，但保留去重判断）
        reset:  重置索引（跳过所有去重，清空存量后全量重录，隐含 rebuild=True）
    """
    if reset:
        rebuild = True

    start = datetime.now(timezone.utc)
    inserted, skipped = _index_documents_sqlite(rebuild=rebuild)

    backend = settings.rag_backend.lower().strip()
    result: dict = {
        "inserted": inserted,
        "skipped": skipped,
        "rebuild": rebuild,
        "reset": reset,
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

    if inserted > 0 or rebuild:
        _sync_document_records()
        threading.Thread(target=_generate_document_summaries, daemon=True).start()

    elapsed = (datetime.now(timezone.utc) - start).total_seconds()
    logger.info(
        "Index complete. inserted=%s skipped=%s elapsed=%.2fs rebuild=%s reset=%s backend=%s",
        inserted,
        skipped,
        elapsed,
        rebuild,
        reset,
        backend,
    )
    result["summary_generation"] = "background"
    return result


def _tokenize(text: str) -> list[str]:
    """简易分词：拉丁词按整词；连续汉字用字二元组，避免整句成一个 token 导致 BM25 零命中。"""

    tokens: list[str] = []
    if not (text or "").strip():
        return tokens
    t = text.lower()
    i = 0
    n = len(t)
    while i < n:
        c = t[i]
        if "\u4e00" <= c <= "\u9fff":
            j = i + 1
            while j < n and "\u4e00" <= t[j] <= "\u9fff":
                j += 1
            run = t[i:j]
            if len(run) == 1:
                tokens.append(run)
            else:
                tokens.extend(run[k : k + 2] for k in range(len(run) - 1))
            i = j
            continue
        if c.isascii() and (c.isalnum() or c in "._-"):
            j = i + 1
            while j < n:
                cj = t[j]
                if cj.isascii() and (cj.isalnum() or cj in "._-"):
                    j += 1
                else:
                    break
            piece = t[i:j]
            if piece:
                tokens.append(piece)
            i = j
            continue
        i += 1
    return tokens


def _extract_filename_hints(query: str) -> list[str]:
    """从问题中抽取文件名线索（如 work.png / report.pdf）。"""

    hints: set[str] = set()
    for m in re.finditer(r"[A-Za-z0-9_.-]+\.(?:png|pdf|txt|md|docx)", query, flags=re.IGNORECASE):
        hints.add(m.group(0))
    return list(hints)


def _rewrite_query(query: str, session_id: str = "") -> str:
    """用 LLM 改写 query，提升召回对"口语问法"的鲁棒性。"""

    prompt = (
        "你是检索查询改写器（query rewrite）。\n\n"
        "## 目标\n"
        "把用户问题改写成更适合从本地文档中检索的关键词/短语。\n"
        "## 规则\n"
        "- 输出必须只是一行关键词，用空格分隔；不要解释，不要换行，不要列表。\n"
        "- 严格输出 3-10 个关键词，不要多也不要少。\n"
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
            max_tokens=64,
            runtime_context={"scenario": "rewrite_query", "session_id": session_id},
        )
        cleaned = " ".join(text.split())
        if len(cleaned) > 200:
            logger.warning("rewrite query too long (%d chars), fallback to original: %s", len(cleaned), cleaned[:80])
            return query
        words = cleaned.split()
        if len(words) > 20:
            logger.warning("rewrite query too many words (%d), fallback to original: %s", len(words), cleaned[:80])
            return query
        return cleaned
    except Exception as exc:
        logger.warning("query rewrite failed: %s", exc)
        return query


def _collect_chunks() -> list[DocumentChunk]:
    """从数据库取出全部 chunk（MVP 实现，后续可替换为更高效索引）。"""

    with KnowledgeSessionLocal() as db:
        return db.execute(select(DocumentChunk)).scalars().all()


def _is_personal_state_query(query: str) -> bool:
    text = (query or "").strip().lower()
    if not text:
        return False
    has_self_ref = any(k in text for k in ["我", "我的", "最近", "这段时间"])
    has_state = any(k in text for k in ["饮食怎么样", "吃得怎么样", "运动怎么样", "状态怎么样", "我最近", "我今天", "我这周"])
    return has_self_ref and has_state


def _apply_source_filter(rows: list[DocumentChunk], query: str) -> list[DocumentChunk]:
    """按问题类型做来源过滤，减少“指南内容误当个人事实”。"""

    if not rows:
        return rows

    if _is_personal_state_query(query):
        blocked = {"guideline", "recipe", "manual", "paper"}
        filtered = [r for r in rows if (r.doc_topic or "other") not in blocked]
        # 若过滤后为空，回退原集合，避免极端情况下完全无候选可检索。
        if filtered:
            return filtered
    return rows


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
                if score < _PATH_HIT_SCORE:
                    score = _PATH_HIT_SCORE

        for h in hints:
            stem = os.path.splitext(h)[0].lower()
            if stem and stem in q_lower and stem in path_lower:
                hit = True
                if score < _PATH_STEM_SCORE:
                    score = _PATH_STEM_SCORE

        if not hit:
            continue
        score = min(score, _PATH_SCORE_CAP)

        scored.append(
            Citation(
                path=r.file_path,
                page=r.page,
                snippet=r.chunk_text or "",
                score=score,
                relatedness=_classes_score(score),
                reason="path",
                source_type=getattr(r, "source_type", None),
                doc_topic=getattr(r, "doc_topic", None),
                section_title=getattr(r, "section_title", None),
                section_path=getattr(r, "section_path", None),
                summary=getattr(r, "summary", None),
                chunk_id=r.id,
            )
        )

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

    min_score = getattr(settings, "rag_evidence_min_score", 0.02)

    def _norm_bm25(s: float) -> float:
        # BM25 原始分无量纲，饱和度映射到 0-60：s>=1 即达上限，弱于向量通道。
        return float(min(s, 1.0)) * _BM25_SCORE_MAX

    citations: list[Citation] = []
    for r, s in zip(rows, scores):
        if s < min_score:
            continue
        norm = _norm_bm25(s)
        citations.append(
            Citation(
                path=r.file_path,
                page=r.page,
                snippet=r.chunk_text or "",
                score=norm,
                relatedness=_classes_score(norm),
                reason=reason,
                source_type=getattr(r, "source_type", None),
                doc_topic=getattr(r, "doc_topic", None),
                section_title=getattr(r, "section_title", None),
                section_path=getattr(r, "section_path", None),
                summary=getattr(r, "summary", None),
                chunk_id=r.id,
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
    for o in out:
        o.relatedness = _classes_score(o.score)
    out.sort(key=lambda x: x.score, reverse=True)
    return out[:top_k]


def _rrf_fuse(*lists: list[Citation], top_k: int, k: int = 60) -> list[Citation]:
    """RRF 融合：对不同召回通道做稳健排序融合。

    RRF 分数仅用于排序；展示的 score 字段保留规范化的相关度得分（0-100）。
    """

    fused: dict[tuple[str, int | None, str], Citation] = {}
    rrf_score: dict[tuple[str, int | None, str], float] = {}
    display_score: dict[tuple[str, int | None, str], float] = {}

    for l in lists:
        for rank, c in enumerate(l, start=1):
            key = (c.path, c.page, c.snippet)
            is_vector = "vector" in (c.reason or "")
            if key not in fused:
                fused[key] = Citation(
                    path=c.path, page=c.page, snippet=c.snippet, score=0.0, reason=c.reason,
                    source_type=c.source_type, doc_topic=c.doc_topic,
                    section_title=c.section_title, section_path=c.section_path, summary=c.summary,
                )
            else:
                if c.reason not in fused[key].reason:
                    fused[key].reason = f"{fused[key].reason}+{c.reason}"
            # 展示分数：向量分覆盖 BM25/路径分；非向量通道取最大值
            if is_vector:
                display_score[key] = c.score
            else:
                display_score[key] = max(display_score.get(key, 0.0), c.score)
            rrf_score[key] = rrf_score.get(key, 0.0) + (1.0 / (k + rank))

    for key, score in rrf_score.items():
        fused[key].score = score

    out = list(fused.values())
    out.sort(key=lambda x: x.score, reverse=True)

    for o in out:
        key = (o.path, o.page, o.snippet)
        if key in display_score:
            o.score = display_score[key]
        o.relatedness = _classes_score(o.score)

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
            Citation(
                path=h.path,
                page=h.page,
                snippet=h.snippet,
                score=_norm_vector(h.score),
                relatedness=_classes_score(_norm_vector(h.score)),
                reason=h.reason,
                source_type=h.source_type,
                doc_topic=h.doc_topic,
                section_title=h.section_title,
                section_path=h.section_path,
                chunk_id=h.chunk_id,
            )
            for h in hits
        ]
    except Exception as exc:
        logger.warning("langchain retrieval failed: %s", exc)
        return []


def _apply_document_summary_filter(rows: list) -> list:
    """用文档摘要做轻量预过滤：查询词匹配文档摘要则只保留该文档的 chunks。"""
    with KnowledgeSessionLocal() as db:
        docs = db.execute(
            select(Document.file_path, Document.summary)
            .where(Document.summary != None)
        ).all()
    if not docs:
        return rows

    corpus = [_tokenize(d.summary) for d in docs if d.summary]
    if not corpus:
        return rows
    tokens = _tokenize(" ".join(_tokenize(rows[0].chunk_text or "") if hasattr(rows[0], 'chunk_text') else ""))
    # 用查询词匹配文档摘要
    query_tokens = _tokenize(" ".join(_tokenize(d.summary[:200]) for d in docs if d.summary))
    # 简化：用文档摘要 BM25 匹配查询
    doc_map = {d.file_path: d.summary for d in docs if d.summary}
    doc_paths = list(doc_map.keys())
    doc_texts = [doc_map[p] for p in doc_paths]
    doc_corpus_tokens = [_tokenize(t) for t in doc_texts]
    bm25 = BM25Okapi(doc_corpus_tokens)
    query_tokens = _tokenize(query)
    if not query_tokens:
        return rows
    scores = bm25.get_scores(query_tokens)
    matched = [(p, s) for p, s in zip(doc_paths, scores) if s > 0]
    if not matched:
        return rows
    # 保留命中文档的 file_path 集合
    keep_paths = set(p for p, s in matched)
    filtered = [r for r in rows if r.file_path in keep_paths]
    if filtered:
        logger.info("Document summary filter: %d/%d rows kept for %d matched docs",
                     len(filtered), len(rows), len(keep_paths))
        return filtered
    return rows


def search_chunks(query: str, top_k: int = 5, session_id: str = "") -> list[Citation]:
    """统一检索入口：按 backend 分流到 bm25 / langchain / hybrid。"""

    backend = settings.rag_backend.lower().strip()
    top_k = top_k or settings.rag_top_k

    rows = _collect_chunks()
    rows = _apply_source_filter(rows, query)
    if not rows and backend in {"bm25", "hybrid"}:
        return []

    rows = _apply_document_summary_filter(rows)

    rewritten = _rewrite_query(query, session_id=session_id)

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


def rag_answer(
    question: str,
    citations: list[Citation],
    *,
    profile_context: str | None = None,
    pre_advice: str | None = None,
    on_token: Callable[[str], None] | None = None,
) -> str:
    """把候选证据拼成 Context，交给 LLM 生成带引用回答。"""

    if not citations:
        return settings.rag_no_evidence_template

    # 证据强度门槛基于统一的 0-100 相关度得分；weak 以上才算有效证据。
    best_score = max(c.score for c in citations)
    weak_threshold = getattr(settings, "rag_score_weak", 30)
    strong_count = sum(1 for c in citations if c.score >= weak_threshold)
    semantic_channel = any("vector" in (c.reason or "") for c in citations)
    if semantic_channel:
        if not any((c.snippet or "").strip() for c in citations):
            return settings.rag_no_evidence_template
    else:
        rich_snippet_count = sum(1 for c in citations if len((c.snippet or "").strip()) >= 30)
        if strong_count == 0 and best_score < weak_threshold and rich_snippet_count < 2:
            return settings.rag_no_evidence_template

    window_size = 2
    context_lines: list[str] = []
    for idx, cite in enumerate(citations, start=1):
        meta = f"{cite.reason}|{cite.source_type or 'unknown'}|{cite.doc_topic or 'other'}"
        if cite.section_path:
            meta += f"|{cite.section_path}"
        elif cite.section_title:
            meta += f"|section:{cite.section_title}"
        label = f"[{idx}] {os.path.basename(cite.path)} p{cite.page or '-'} ({meta})"
        if cite.chunk_id:
            try:
                with KnowledgeSessionLocal() as db:
                    rows = db.execute(
                        select(DocumentChunk.id, DocumentChunk.chunk_text)
                        .where(DocumentChunk.file_path == cite.path)
                        .where(DocumentChunk.id.between(cite.chunk_id - window_size, cite.chunk_id + window_size))
                        .order_by(DocumentChunk.id)
                    ).all()
                if rows:
                    parts = []
                    for cid, text in rows:
                        text = text or ""
                        if cid == cite.chunk_id:
                            parts.append(f">>> {text}")
                        else:
                            parts.append(text)
                    context_lines.append(f"{label}:\n" + "\n---\n".join(parts))
                else:
                    context_lines.append(f"{label}: {cite.snippet}")
            except Exception:
                context_lines.append(f"{label}: {cite.snippet}")
        else:
            context_lines.append(f"{label}: {cite.snippet}")
    context = "\n\n".join(context_lines)

    doc_paths = {c.path for c in citations}
    if doc_paths:
        with KnowledgeSessionLocal() as db:
            doc_rows = db.execute(
                select(Document.file_path, Document.summary)
                .where(Document.file_path.in_(list(doc_paths)))
                .where(Document.summary != None)
            ).all()
        if doc_rows:
            bg_lines = [f"  [{os.path.basename(fp)}] {summary}" for fp, summary in doc_rows]
            context += "\n\nDocument Background:\n" + "\n".join(bg_lines)

    profile_lines: list[str] = []
    if profile_context:
        profile_lines.append(f"PROFILE: {profile_context}")
    if pre_advice:
        profile_lines.append(f"PROFILE_HINT: {pre_advice}")
    profile_block = "\n".join(profile_lines)

    system_prompt = (
        "你是一个严谨的知识库问答助手。你只能根据提供的 Context 回答，并在句子中内联标注引用 [1][2]。\n"
        "如果 Context 不足以回答，请明确说不知道，不要编造。\n\n"
        "## 关键约束\n"
        "Context 默认是外部知识文档，不代表用户真实发生过的行为。\n"
        "例如菜谱/指南中的食物不能被表述成‘用户已经吃过’。\n"
        "只有当 Context 明确出现‘用户本人记录/我今天吃了/个人日志’等证据时，才能描述为用户已发生事实。\n"
        "## 个性化约束\n"
        "若提供了 PROFILE/PROFILE_HINT，可用于个性化表达，但不得与 Context 冲突，也不得把文档建议误写成用户历史。"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"{profile_block}\n\nContext:\n{context}\n\nQuestion: {question}",
        },
    ]
    return chat_completion(
        messages,
        temperature=0.2,
        runtime_context={
            "scenario": "rag_answer",
            "evidence_best_score": round(best_score, 6),
            "evidence_strong_count": strong_count,
        },
        on_token=on_token,
    )
