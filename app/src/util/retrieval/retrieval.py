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
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import select
from rank_bm25 import BM25Okapi

from ...common.config.log_config import logger
from ...common.config.knowledge_db_config import KnowledgeSessionLocal
from ...domain.entity.knowledge_entity import DocumentChunk
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
    summary: str | None = None


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
            deleted = db.query(DocumentChunk).delete()
            db.commit()
            logger.info("Rebuild index: cleared document_chunks deleted=%s", deleted)

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
                        source_type=source_type,
                        doc_topic=doc_topic,
                        section_title=chunk.section_title,
                        heading_level=chunk.heading_level,
                        is_table=chunk.is_table,
                    )
                )
                inserted += 1
        db.commit()

    return inserted, skipped


def _generate_chunk_summaries() -> None:
    with KnowledgeSessionLocal() as db:
        rows = db.execute(
            select(DocumentChunk).where(DocumentChunk.summary == None).limit(200)
        ).scalars().all()
        if not rows:
            return

        batch_size = 10
        updated = 0
        for i in range(0, len(rows), batch_size):
            group = rows[i:i + batch_size]
            texts = []
            for r in group:
                t = (r.chunk_text or "").strip()[:300]
                texts.append(t)

            prompt_lines = []
            for idx, t in enumerate(texts, start=1):
                prompt_lines.append(f"[{idx}] {t}")
            try:
                answer = chat_completion(
                    [{"role": "user", "content": (
                        "为以下每段文本生成一句话摘要（20字以内），"
                        "按格式输出：\n[idx] 摘要\n\n文本：\n" + "\n".join(prompt_lines)
                    )}],
                    temperature=0.0,
                )
                for r, line in zip(group, answer.strip().split("\n")):
                    line = line.strip()
                    if not line:
                        continue
                    m = re.match(r"^\[(\d+)\]\s*(.*)", line)
                    if m:
                        idx = int(m.group(1)) - 1
                        if 0 <= idx < len(group):
                            group[idx].summary = m.group(2).strip()[:80]
                            updated += 1
            except Exception as exc:
                logger.warning("Chunk summary batch failed: %s", exc)
                # fallback: individual calls
                for r in group:
                    try:
                        s = chat_completion(
                            [{"role": "user", "content": f"用一句话概括以下内容（20字以内）：\n{(r.chunk_text or '')[:300]}"}],
                            temperature=0.0,
                        ).strip()[:80]
                        if s:
                            r.summary = s
                            updated += 1
                    except Exception:
                        pass
        db.commit()
        logger.info("Generated chunk summaries count=%s", updated)


def get_available_knowledge_topics() -> list[dict]:
    with KnowledgeSessionLocal() as db:
        rows = db.execute(
            select(
                DocumentChunk.file_path,
                DocumentChunk.section_title,
                DocumentChunk.summary,
            ).distinct().limit(500)
        ).all()
        doc_map: dict[str, dict] = {}
        for fp, section, summary in rows:
            if fp not in doc_map:
                topic = _infer_doc_topic(fp)
                doc_map[fp] = {"file": os.path.basename(fp), "topic": topic, "sections": set(), "summaries": []}
            if section:
                doc_map[fp]["sections"].add(section)
            if summary:
                doc_map[fp]["summaries"].append(summary)
        result = []
        for info in doc_map.values():
            entry = {"file": info["file"], "topic": info["topic"]}
            if info["sections"]:
                entry["sections"] = sorted(info["sections"])[:10]
            if info["summaries"]:
                entry["summary_samples"] = info["summaries"][:3]
            result.append(entry)
        return result


def get_document_outline() -> list[dict]:
    with KnowledgeSessionLocal() as db:
        rows = db.execute(
            select(
                DocumentChunk.file_path,
                DocumentChunk.section_title,
                DocumentChunk.heading_level,
            ).distinct().order_by(DocumentChunk.file_path, DocumentChunk.heading_level, DocumentChunk.id)
        ).all()
        doc_outline: dict[str, list[dict]] = {}
        for fp, title, level in rows:
            if not title:
                continue
            if fp not in doc_outline:
                doc_outline[fp] = []
            doc_outline[fp].append({"title": title, "level": level or 1})
        result = []
        for fp, headings in doc_outline.items():
            result.append({
                "file": os.path.basename(fp),
                "headings": headings[:20],
            })
        return result


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

    if inserted > 0:
        _generate_chunk_summaries()

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


def _rewrite_query(query: str) -> str:
    """用 LLM 改写 query，提升召回对“口语问法”的鲁棒性。"""

    prompt = (
        "你是检索查询改写器（query rewrite）。\n\n"
        "## 目标\n"
        "把用户问题改写成更适合从本地文档中检索的关键词/短语。\n"
        "## 规则\n"
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
                score += 10.0

        for h in hints:
            stem = os.path.splitext(h)[0].lower()
            if stem and stem in q_lower and stem in path_lower:
                hit = True
                score += 2.0

        if not hit:
            continue

        snippet = (r.chunk_text or "")[:240]
        scored.append(
            Citation(
                path=r.file_path,
                page=r.page,
                snippet=snippet,
                score=score,
                reason="path",
                source_type=getattr(r, "source_type", None),
                doc_topic=getattr(r, "doc_topic", None),
                section_title=getattr(r, "section_title", None),
                summary=getattr(r, "summary", None),
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

    citations: list[Citation] = []
    for r, s in zip(rows, scores):
        if s < min_score:
            continue
        citations.append(
            Citation(
                path=r.file_path,
                page=r.page,
                snippet=(r.chunk_text or "")[:240],
                score=float(s),
                reason=reason,
                source_type=getattr(r, "source_type", None),
                doc_topic=getattr(r, "doc_topic", None),
                section_title=getattr(r, "section_title", None),
                summary=getattr(r, "summary", None),
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
                fused[key] = Citation(
                    path=c.path, page=c.page, snippet=c.snippet, score=0.0, reason=c.reason,
                    source_type=c.source_type, doc_topic=c.doc_topic,
                    section_title=c.section_title, summary=c.summary,
                )
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
            Citation(
                path=h.path,
                page=h.page,
                snippet=h.snippet,
                score=h.score,
                reason=h.reason,
                source_type=getattr(h, "source_type", None),
                doc_topic=getattr(h, "doc_topic", None),
                section_title=getattr(h, "section_title", None),
                summary=getattr(h, "summary", None),
            )
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
    rows = _apply_source_filter(rows, query)
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


def rag_answer(
    question: str,
    citations: list[Citation],
    *,
    profile_context: str | None = None,
    pre_advice: str | None = None,
) -> str:
    """把候选证据拼成 Context，交给 LLM 生成带引用回答。"""

    if not citations:
        return settings.rag_no_evidence_template

    # 证据强度门槛原本针对 BM25 分数尺度；向量距离分 1/(1+d) 与 CrossEncoder logits 与 0.02 不可比，
    # 否则会误杀「已有命中」的语义检索结果。
    best_score = max(c.score for c in citations)
    strong_count = sum(1 for c in citations if c.score >= settings.rag_evidence_min_score)
    semantic_channel = any("vector" in (c.reason or "") for c in citations)
    if semantic_channel:
        if not any((c.snippet or "").strip() for c in citations):
            return settings.rag_no_evidence_template
    else:
        rich_snippet_count = sum(1 for c in citations if len((c.snippet or "").strip()) >= 30)
        if strong_count == 0 and best_score < settings.rag_evidence_min_score and rich_snippet_count < 2:
            return settings.rag_no_evidence_template

    context_lines: list[str] = []
    for idx, cite in enumerate(citations, start=1):
        meta = f"{cite.reason}|{cite.source_type or 'unknown'}|{cite.doc_topic or 'other'}"
        if cite.section_title:
            meta += f"|section:{cite.section_title}"
        if cite.summary:
            meta += f"|summary:{cite.summary}"
        label = (
            f"[{idx}] {os.path.basename(cite.path)} p{cite.page or '-'} ({meta})"
        )
        context_lines.append(f"{label}: {cite.snippet}")
    context = "\n".join(context_lines)

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
    )
