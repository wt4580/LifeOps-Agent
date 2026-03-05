from pydantic_settings import BaseSettings
from pathlib import Path
from pydantic import Field


class Settings(BaseSettings):
    """全局配置对象。

    设计目标：
    - 所有可变参数统一由 `.env` 注入，代码中不硬编码环境差异。
    - 通过类型注解 + 默认值，降低配置错误概率。
    - 对“初次运行”的本地开发场景尽量给出可用默认值。
    """

    # -------------------------
    # LLM 基础配置（Qwen / DashScope）
    # -------------------------
    qwen_api_key: str
    qwen_model: str = "qwen-turbo"
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    # -------------------------
    # 数据库配置
    # -------------------------
    # 这里显式构造“基于项目目录的绝对路径”，避免从不同工作目录启动时产生多份 DB。
    database_url: str = f"sqlite:///{(Path(__file__).resolve().parent.parent / 'data' / 'lifeops.db').as_posix()}"

    # -------------------------
    # 文档与系统级配置
    # -------------------------
    docs_dir: str = r"E:\work\agent\docu"
    tesseract_cmd: str | None = None
    log_level: str = "INFO"

    # -------------------------
    # OCR 配置（用于 PNG 图片文字识别）
    # -------------------------
    # alias 的作用：让环境变量名保持大写风格（如 OCR_LANG），
    # 但在代码中仍使用 Python 风格字段名（ocr_lang）。
    ocr_lang: str = Field(default="chi_sim+eng", alias="OCR_LANG")
    ocr_psm: int = Field(default=6, alias="OCR_PSM")
    ocr_oem: int = Field(default=3, alias="OCR_OEM")
    ocr_debug: bool = Field(default=False, alias="OCR_DEBUG")

    # -------------------------
    # RAG 配置（检索后端与向量参数）
    # -------------------------
    # rag_backend:
    # - bm25: 仅走旧检索链路（最稳）
    # - langchain: 仅走向量检索链路
    # - hybrid: bm25 + 向量融合（推荐逐步灰度）
    rag_backend: str = Field(default="bm25", alias="RAG_BACKEND")
    rag_top_k: int = Field(default=5, alias="RAG_TOP_K")
    rag_candidate_k: int = Field(default=24, alias="RAG_CANDIDATE_K")
    rag_vector_dir: str = Field(default="data/vector/faiss", alias="RAG_VECTOR_DIR")
    rag_embed_model: str = Field(default="BAAI/bge-small-zh-v1.5", alias="RAG_EMBED_MODEL")
    rag_rerank_model: str = Field(default="BAAI/bge-reranker-base", alias="RAG_RERANK_MODEL")
    rag_use_rerank: bool = Field(default=True, alias="RAG_USE_RERANK")
    rag_chunk_size: int = Field(default=600, alias="RAG_CHUNK_SIZE")
    rag_chunk_overlap: int = Field(default=120, alias="RAG_CHUNK_OVERLAP")

    class Config:
        # 指定 .env 读取位置与编码。
        env_file = ".env"
        env_file_encoding = "utf-8"


# 单例配置对象：其他模块直接 `from .settings import settings` 使用。
settings = Settings()
