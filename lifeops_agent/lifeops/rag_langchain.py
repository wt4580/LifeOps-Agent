from __future__ import annotations

"""lifeops.rag_langchain

这个模块提供“向量检索链路”的实现，作为 BM25 的可选增强后端。

核心能力：
1) 文档加载与分块（PDF/TXT/MD/DOCX/PNG-OCR）。
2) 构建并持久化本地 FAISS 索引。
3) 查询时先向量召回，再可选 cross-encoder 重排。

设计原则：
- 不破坏原有 `retrieval.py` 接口；
- 依赖缺失时可回退，不阻断主流程；
- 每个候选都保留 source/page，保证引用可追溯。
"""

import hashlib
import logging
import os
import pickle
import shutil
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .ingest.ocr_png import extract_png
from .settings import settings


logger = logging.getLogger(__name__)


@dataclass
class LCRetrievalHit:
    """LangChain 向量检索命中结果的统一结构。"""

    path: str
    page: int | None
    snippet: str
    score: float
    reason: str


def _require_langchain() -> None:
    """运行时依赖检查：未安装 LangChain 相关包时给出清晰提示。"""
    try:
        import langchain  # noqa: F401
    except Exception as exc:  # pragma: no cover - runtime dependency guard
        raise RuntimeError(
            "LangChain dependencies are not installed. Run `pip install -r requirements.txt` first."
        ) from exc


def _pick_model_source(*, local_path: str | None, remote_model: str, model_label: str) -> tuple[str, str]:
    """选择模型加载来源：优先本地路径，不可用则回退远程模型名。"""

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

    logger.info("Model source [%s]=remote model=%s", model_label, remote_model)
    return "remote", remote_model


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
    """懒加载重排模型；若关闭或依赖不可用则返回 None。"""
    if not settings.rag_use_rerank:
        return None
    try:
        from sentence_transformers import CrossEncoder
    except Exception:
        return None

    source, model_name = _pick_model_source(
        local_path=settings.rag_rerank_model_local,
        remote_model=settings.rag_rerank_model,
        model_label="reranker",
    )

    try:
        return CrossEncoder(model_name)
    except Exception as exc:
        if source == "local":
            logger.warning(
                "Failed to load local reranker model path=%s err=%s; fallback to remote model=%s",
                model_name,
                exc,
                settings.rag_rerank_model,
            )
            logger.info("Model source [reranker]=remote model=%s", settings.rag_rerank_model)
            return CrossEncoder(settings.rag_rerank_model)
        return None


def _make_splitter():
    """创建递归文本切分器。"""
    _require_langchain()
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    return RecursiveCharacterTextSplitter(
        chunk_size=settings.rag_chunk_size,
        chunk_overlap=settings.rag_chunk_overlap,
        separators=["\n\n", "\n", "。", "；", "，", " ", ""],
    )


def _load_documents(root_dir: str):
    """递归加载文档并切分为可向量化的 Document 列表。"""
    _require_langchain()
    from langchain_core.documents import Document
    from langchain_community.document_loaders import PyPDFLoader, TextLoader
    from langchain_text_splitters import MarkdownHeaderTextSplitter

    splitter = _make_splitter()
    docs: list[Document] = []

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            path = os.path.join(dirpath, filename)
            ext = os.path.splitext(filename)[1].lower()
            if ext == ".pdf":
                loaded = PyPDFLoader(path).load()
                docs.extend(splitter.split_documents(loaded))
            elif ext == ".docx":
                try:
                    # 按需导入，避免缺少 docx2txt 时整个索引失败。
                    from langchain_community.document_loaders import Docx2txtLoader

                    loaded = Docx2txtLoader(path).load()
                    docs.extend(splitter.split_documents(loaded))
                except Exception as exc:
                    logger.warning("Skip DOCX file due to loader/dependency error path=%s err=%s", path, exc)
                    continue
            elif ext == ".md":
                raw = TextLoader(path, encoding="utf-8", autodetect_encoding=True).load()[0]
                md_splitter = MarkdownHeaderTextSplitter(
                    headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")]
                )
                header_docs = md_splitter.split_text(raw.page_content)
                for d in header_docs:
                    d.metadata["source"] = path
                docs.extend(splitter.split_documents(header_docs))
            elif ext == ".txt":
                loaded = TextLoader(path, encoding="utf-8", autodetect_encoding=True).load()
                docs.extend(splitter.split_documents(loaded))
            elif ext == ".png":
                text = extract_png(path)
                if text.strip():
                    png_docs = [Document(page_content=text, metadata={"source": path})]
                    docs.extend(splitter.split_documents(png_docs))
            else:
                continue

    # 规范 metadata，确保 source/page 可追溯
    normalized: list[Document] = []
    chunk_pos_counter: dict[tuple[str, int | None], int] = {}
    for d in docs:
        src = d.metadata.get("source") or d.metadata.get("file_path") or ""
        page = d.metadata.get("page")
        try:
            page_num = int(page) + 1 if page is not None else None
        except Exception:
            page_num = None

        key = (str(src), page_num)
        chunk_pos_counter[key] = chunk_pos_counter.get(key, 0) + 1

        d.metadata["source"] = str(src)
        d.metadata["page"] = page_num
        # 每个 source+page 内的序号，用于后续判断“同位置是否改写”。
        d.metadata["chunk_pos"] = chunk_pos_counter[key]
        normalized.append(d)

    return normalized


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


def build_vector_index(*, root_dir: str, rebuild: bool = False) -> dict:
    """构建/重建 FAISS 向量索引并持久化到本地目录。"""
    _require_langchain()
    from langchain_community.vectorstores import FAISS

    docs = _load_documents(root_dir)
    vec_dir = Path(settings.rag_vector_dir)

    old_sigs = _collect_old_doc_signatures(vec_dir)
    new_sigs = _collect_new_doc_signatures(docs)
    diff = _calc_vector_diff(old_sigs, new_sigs)

    if rebuild and vec_dir.exists():
        shutil.rmtree(vec_dir)
    vec_dir.mkdir(parents=True, exist_ok=True)

    if docs:
        vs = FAISS.from_documents(docs, _embeddings())
        vs.save_local(str(vec_dir))

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
    return FAISS.load_local(
        str(vec_dir),
        _embeddings(),
        allow_dangerous_deserialization=True,
    )


def search_with_rerank(query: str, *, top_k: int, candidate_k: int) -> list[LCRetrievalHit]:
    """执行向量检索 + 可选重排。

    - 第一阶段：向量召回 candidate_k 条候选。
    - 第二阶段：若启用 reranker，则对候选打相关性分并重排。
    - 最终返回 top_k 条。
    """
    vs = _load_vector_store()
    if vs is None:
        return []

    pairs = vs.similarity_search_with_score(query, k=max(top_k, candidate_k))
    if not pairs:
        return []

    candidates: list[LCRetrievalHit] = []
    for doc, distance in pairs:
        src = str(doc.metadata.get("source") or "")
        page = doc.metadata.get("page")
        snippet = (doc.page_content or "").strip()[:280]
        # FAISS L2 距离越小越好，这里转成可排序分数（越大越好）
        score = 1.0 / (1.0 + float(distance))
        candidates.append(LCRetrievalHit(path=src, page=page, snippet=snippet, score=score, reason="vector"))

    reranker = _reranker()
    if reranker is None:
        candidates.sort(key=lambda x: x.score, reverse=True)
        return candidates[:top_k]

    inputs = [[query, c.snippet] for c in candidates]
    rerank_scores = reranker.predict(inputs)
    for c, rs in zip(candidates, rerank_scores):
        c.score = float(rs)
        c.reason = "vector-rerank"

    candidates.sort(key=lambda x: x.score, reverse=True)
    return candidates[:top_k]
