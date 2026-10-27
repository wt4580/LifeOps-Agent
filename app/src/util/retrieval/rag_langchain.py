from __future__ import annotations

"""lifeops.rag_langchain

这个模块提供"向量检索链路"的实现，作为 BM25 的可选增强后端。

核心能力：
1) 从 SQLite document_chunks 表加载已切分文本，构建本地 FAISS 向量索引。
2) 查询时先向量召回（vector_recall），再可选 cross-encoder 重排（rerank_hits）。

设计原则：
- 不破坏原有 `retrieval.py` 接口；
- 依赖缺失时给出明确错误（重排模型不回退为静默关闭或联网拉取）。
- 每个候选都保留 source/page/section_path，保证引用可追溯。
"""

import hashlib
import logging
import pickle
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from sqlalchemy import select

from ...common.config.base_config import settings
from ...common.config.knowledge_db_config import KnowledgeSessionLocal
from ...domain.entity.knowledge_entity import DocumentChunk


logger = logging.getLogger(__name__)


@dataclass
class LCRetrievalHit:
    """LangChain 向量检索命中结果的统一结构。"""

    path: str
    page: int | None
    snippet: str
    score: float
    reason: str
    section_path: str | None = None
    source_type: str | None = None
    doc_topic: str | None = None
    section_title: str | None = None
    chunk_id: int | None = None


def _require_langchain() -> None:
    """运行时依赖检查：未安装 LangChain 相关包时给出清晰提示。"""
    try:
        import langchain  # noqa: F401
    except Exception as exc:  # pragma: no cover - runtime dependency guard
        raise RuntimeError(
            "LangChain dependencies are not installed. Run `pip install -r requirements.txt` first."
        ) from exc


def _pick_model_source(*, local_path: str | None, remote_model: str, model_label: str) -> tuple[str, str]:
    """选择模型加载来源：优先指定本地路径，其次 HuggingFace 默认缓存，最后回退远程模型名。"""

    if local_path:
        p = Path(local_path)
        if p.exists():
            logger.info("Model source [%s]=local path=%s", model_label, str(p))
            return "local", str(p)
        logger.warning(
            "Local model path not found for %s path=%s; fallback to remote model=%s",
            model_label,
            local_path,
            remote_model,
        )

    cache_path = _find_in_hf_cache(remote_model)
    if cache_path:
        logger.info("Model source [%s]=hf-cache path=%s", model_label, cache_path)
        return "local", cache_path

    logger.info("Model source [%s]=remote model=%s", model_label, remote_model)
    return "remote", remote_model


def _find_in_hf_cache(model_name: str) -> str | None:
    """在 HuggingFace 默认缓存目录中查找已下载的模型快照。"""
    repo_id = model_name.replace("/", "--")
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    model_dir = cache_dir / f"models--{repo_id}"
    if not model_dir.is_dir():
        return None
    refs_dir = model_dir / "refs"
    snapshots_dir = model_dir / "snapshots"
    if not refs_dir.is_dir() or not snapshots_dir.is_dir():
        return None
    main_ref = refs_dir / "main"
    if main_ref.is_file():
        commit_hash = main_ref.read_text().strip()
        snapshot = snapshots_dir / commit_hash
        if snapshot.is_dir():
            return str(snapshot)
    # fallback: pick first snapshot
    for entry in sorted(snapshots_dir.iterdir()):
        if entry.is_dir():
            return str(entry)
    return None


@lru_cache(maxsize=1)
def _embeddings():
    """懒加载 embedding 模型，避免每次查询重复初始化。"""
    _require_langchain()
    from langchain_huggingface import HuggingFaceEmbeddings

    source, model_name = _pick_model_source(
        local_path=settings.rag_embed_model_local,
        remote_model=settings.rag_embed_model,
        model_label="embedding",
    )

    try:
        return HuggingFaceEmbeddings(model_name=model_name)
    except Exception as exc:
        if source == "local":
            logger.warning(
                "Failed to load local embedding model path=%s err=%s; fallback to remote model=%s",
                model_name,
                exc,
                settings.rag_embed_model,
            )
            logger.info("Model source [embedding]=remote model=%s", settings.rag_embed_model)
            return HuggingFaceEmbeddings(model_name=settings.rag_embed_model)
        raise


@lru_cache(maxsize=1)
def _reranker():
    """懒加载 CrossEncoder 重排模型（失败直接抛错，不回退联网下载）。"""
    if not settings.rag_use_rerank:
        return None
    try:
        from sentence_transformers import CrossEncoder
    except Exception as exc:
        raise RuntimeError(
            "未安装 sentence-transformers，无法加载重排模型。请执行: pip install -r app/requirements.txt"
        ) from exc

    source, model_name = _pick_model_source(
        local_path=settings.rag_rerank_model_local,
        remote_model=settings.rag_rerank_model,
        model_label="reranker",
    )
    # 本地目录常缺少 tokenizer.json；走 fast 会在 convert_slow_tokenizer 里触发
    # vocab_file is None -> endswith 报错。慢速 tokenizer 依赖 Python 包 sentencepiece。
    tokenizer_args = {"use_fast": False} if source == "local" else {}
    return CrossEncoder(model_name, tokenizer_args=tokenizer_args)


def _load_documents_from_db():
    """从 document_chunks 表加载文本，转成 LangChain Document 列表供向量化。

    与 BM25 侧使用相同的扫描分块结果，确保两侧 chunk 一一对应。
    """
    _require_langchain()
    from langchain_core.documents import Document

    with KnowledgeSessionLocal() as db:
        rows = db.execute(
            select(DocumentChunk).order_by(DocumentChunk.id)
        ).scalars().all()

        docs: list[Document] = []
        for row in rows:
            if not row.chunk_text:
                continue
            docs.append(Document(
                page_content=row.chunk_text,
                metadata={
                    "source": row.file_path,
                    "page": row.page,
                    "chunk_pos": row.id,
                    "section_path": row.section_path,
                    "section_title": row.section_title,
                    "source_type": row.source_type,
                    "doc_topic": row.doc_topic,
                },
            ))
    return docs


def _doc_signature_from_values(source: str, page: int | None, chunk_pos: int | None, content: str) -> tuple[tuple[str, int | None, int | None], str]:
    """生成文档分块签名：位置键 + 内容哈希。"""

    key = (source, page, chunk_pos)
    content_hash = hashlib.sha256((content or "").encode("utf-8")).hexdigest()
    return key, content_hash


def _collect_new_doc_signatures(docs: list) -> list[tuple[tuple[str, int | None, int | None], str]]:
    out: list[tuple[tuple[str, int | None, int | None], str]] = []
    for d in docs:
        source = str(d.metadata.get("source") or "")
        page = d.metadata.get("page")
        chunk_pos = d.metadata.get("chunk_pos")
        try:
            page_num = int(page) if page is not None else None
        except Exception:
            page_num = None
        try:
            pos_num = int(chunk_pos) if chunk_pos is not None else None
        except Exception:
            pos_num = None
        out.append(_doc_signature_from_values(source, page_num, pos_num, d.page_content or ""))
    return out


def _collect_old_doc_signatures(vec_dir: Path) -> list[tuple[tuple[str, int | None, int | None], str]]:
    """从已持久化的 index.pkl 读取旧索引签名（无需加载 embedding）。"""

    pkl_path = vec_dir / "index.pkl"
    if not pkl_path.exists():
        return []

    try:
        with pkl_path.open("rb") as f:
            obj = pickle.load(f)
    except Exception as exc:
        logger.warning("Failed to load previous vector index metadata path=%s err=%s", pkl_path, exc)
        return []

    docstore = None
    index_to_docstore_id = None
    if isinstance(obj, tuple) and len(obj) == 2:
        docstore, index_to_docstore_id = obj

    if docstore is None or index_to_docstore_id is None:
        return []

    doc_dict = getattr(docstore, "_dict", {}) or {}
    out: list[tuple[tuple[str, int | None, int | None], str]] = []

    for doc_id in index_to_docstore_id.values():
        doc = doc_dict.get(doc_id)
        if doc is None:
            continue
        source = str(getattr(doc, "metadata", {}).get("source") or "")
        page = getattr(doc, "metadata", {}).get("page")
        chunk_pos = getattr(doc, "metadata", {}).get("chunk_pos")
        try:
            page_num = int(page) if page is not None else None
        except Exception:
            page_num = None
        try:
            pos_num = int(chunk_pos) if chunk_pos is not None else None
        except Exception:
            pos_num = None
        out.append(_doc_signature_from_values(source, page_num, pos_num, getattr(doc, "page_content", "") or ""))

    return out


def _calc_vector_diff(
    old_sigs: list[tuple[tuple[str, int | None, int | None], str]],
    new_sigs: list[tuple[tuple[str, int | None, int | None], str]],
) -> dict:
    """计算向量索引差异：新增/改写/删除/不变。"""

    old_map = {k: h for k, h in old_sigs}
    new_map = {k: h for k, h in new_sigs}

    old_keys = set(old_map.keys())
    new_keys = set(new_map.keys())

    added_keys = new_keys - old_keys
    deleted_keys = old_keys - new_keys
    common_keys = old_keys & new_keys

    rewritten_keys = {k for k in common_keys if old_map.get(k) != new_map.get(k)}
    unchanged_keys = {k for k in common_keys if old_map.get(k) == new_map.get(k)}

    return {
        "vector_prev_chunks": len(old_map),
        "vector_chunks": len(new_map),
        "vector_added_chunks": len(added_keys),
        "vector_rewritten_chunks": len(rewritten_keys),
        "vector_deleted_chunks": len(deleted_keys),
        "vector_unchanged_chunks": len(unchanged_keys),
        "vector_changed": bool(added_keys or rewritten_keys or deleted_keys),
    }


def build_vector_index(*, root_dir: str | None = None, rebuild: bool = False) -> dict:
    """构建/重建 FAISS 向量索引并持久化到本地目录。
    
    Args:
        root_dir: 不再使用，保留兼容签名。实际从 document_chunks 表加载。
        rebuild: 是否删除旧索引从头重建。
    """
    docs = _load_documents_from_db()
    vec_dir = Path(settings.rag_vector_dir)

    old_sigs = _collect_old_doc_signatures(vec_dir)
    new_sigs = _collect_new_doc_signatures(docs)
    diff = _calc_vector_diff(old_sigs, new_sigs)

    vec_dir.mkdir(parents=True, exist_ok=True)

    if docs:
        _require_langchain()
        from langchain_community.vectorstores import FAISS

        vs = FAISS.from_documents(docs, _embeddings())
        vs.save_local(str(vec_dir))
    else:
        for filename in ("index.faiss", "index.pkl"):
            (vec_dir / filename).unlink(missing_ok=True)

    # 清理缓存，防止同进程中旧索引驻留
    _load_vector_store.cache_clear()

    result = {"vector_dir": str(vec_dir), "rebuild": rebuild}
    result.update(diff)
    return result


@lru_cache(maxsize=1)
def _load_vector_store():
    """加载本地 FAISS 索引（带缓存）。"""
    _require_langchain()
    from langchain_community.vectorstores import FAISS

    vec_dir = Path(settings.rag_vector_dir)
    if not vec_dir.exists() or not any(vec_dir.iterdir()):
        return None
    previous = _collect_old_doc_signatures(vec_dir)
    current = _collect_new_doc_signatures(_load_documents_from_db())
    if _calc_vector_diff(previous, current)["vector_changed"]:
        logger.warning("Vector index differs from current documents; use SQLite retrieval until rebuilt")
        return None
    return FAISS.load_local(
        str(vec_dir),
        _embeddings(),
        allow_dangerous_deserialization=True,
    )


def _log_rerank_results(channel: str, query: str, hits: list[LCRetrievalHit]) -> None:
    """记录检索/重排后的候选结果，便于排查为何某些 chunk 排前。"""
    for i, h in enumerate(hits, start=1):
        snippet = (h.snippet or "").strip().replace("\n", " ")[:80]
        logger.info(
            "rerank[%s] rank=%d reason=%s score=%.4f path=%s page=%s chunk=%s snippet=%s",
            channel, i, h.reason, h.score, h.path, h.page, h.chunk_id, snippet,
        )


def vector_recall(query: str, *, candidate_k: int) -> list[LCRetrievalHit]:
    """第一阶段召回：向量检索 candidate_k 条候选（不做重排）。"""
    vs = _load_vector_store()
    if vs is None:
        return []

    pairs = vs.similarity_search_with_score(query, k=candidate_k)
    if not pairs:
        return []

    candidates: list[LCRetrievalHit] = []
    for doc, distance in pairs:
        src = str(doc.metadata.get("source") or "")
        page = doc.metadata.get("page")
        snippet = (doc.page_content or "").strip()
        chunk_id = doc.metadata.get("chunk_pos")
        score = 1.0 / (1.0 + float(distance))
        candidates.append(LCRetrievalHit(
            path=src, page=page, snippet=snippet, score=score, reason="vector",
            section_path=doc.metadata.get("section_path"),
            source_type=doc.metadata.get("source_type"),
            doc_topic=doc.metadata.get("doc_topic"),
            section_title=doc.metadata.get("section_title"),
            chunk_id=chunk_id,
        ))

    candidates.sort(key=lambda x: x.score, reverse=True)
    return candidates


def rerank_hits(query: str, hits: list[LCRetrievalHit], *, top_k: int) -> list[LCRetrievalHit]:
    """第三阶段重排：对候选打相关性分并重排，返回 top_k 条。

    未启用 reranker 时，仅按召回分排序返回 top_k。
    """
    if not hits:
        return []

    reranker = _reranker()
    if reranker is None:
        hits.sort(key=lambda x: x.score, reverse=True)
        hits = hits[:top_k]
        _log_rerank_results("vector", query, hits)
        return hits

    inputs = [[query, c.snippet] for c in hits]
    rerank_scores = reranker.predict(inputs)
    for c, rs in zip(hits, rerank_scores):
        c.score = float(rs)
        c.reason = "vector-rerank"

    hits.sort(key=lambda x: x.score, reverse=True)
    hits = hits[:top_k]
    _log_rerank_results("vector-rerank", query, hits)
    return hits


def search_with_rerank(query: str, *, top_k: int, candidate_k: int) -> list[LCRetrievalHit]:
    """向量召回 + 重排，兼容旧入口（用于纯向量基线对比）。"""
    hits = vector_recall(query, candidate_k=candidate_k)
    return rerank_hits(query, hits, top_k=top_k)
