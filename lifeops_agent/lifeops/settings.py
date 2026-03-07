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
    # 可选：本地模型目录（例如 HuggingFace snapshots 的绝对路径）；配置后优先本地加载。
    rag_embed_model_local: str | None = Field(default=None, alias="RAG_EMBED_MODEL_LOCAL")
    rag_rerank_model_local: str | None = Field(default=None, alias="RAG_RERANK_MODEL_LOCAL")
    rag_use_rerank: bool = Field(default=True, alias="RAG_USE_RERANK")
    rag_chunk_size: int = Field(default=600, alias="RAG_CHUNK_SIZE")
    rag_chunk_overlap: int = Field(default=120, alias="RAG_CHUNK_OVERLAP")
    rag_evidence_min_score: float = Field(default=0.02, alias="RAG_EVIDENCE_MIN_SCORE")
    rag_no_evidence_template: str = Field(
        default="当前证据不足以支持确定结论。我可以继续按关键词重检，或请你提供更具体的文件名/术语。",
        alias="RAG_NO_EVIDENCE_TEMPLATE",
    )

    # -------------------------
    # 主动建议阈值配置
    # -------------------------
    proactive_advice_threshold: float = Field(default=0.62, alias="PROACTIVE_ADVICE_THRESHOLD")
    proactive_todo_hours: int = Field(default=72, alias="PROACTIVE_TODO_HOURS")
    # 按时间窗口读取个人事件，替代固定条数截断。
    proactive_event_window_hours: int = Field(default=168, alias="PROACTIVE_EVENT_WINDOW_HOURS")

    # -------------------------
    # 个人记忆范围配置
    # -------------------------
    # session: 仅当前会话可见；global: 跨会话共享（单用户推荐）
    personal_memory_scope: str = Field(default="global", alias="PERSONAL_MEMORY_SCOPE")
    # 当 scope=global 时，所有个人事件/画像写入这个 owner id
    personal_memory_global_id: str = Field(default="default_user", alias="PERSONAL_MEMORY_GLOBAL_ID")

    # -------------------------
    # 聊天留存策略
    # -------------------------
    # 当单 session 消息数超过上限时，触发“摘要 + 删除最旧消息”的滚动压缩。
    chat_retention_limit: int = Field(default=1000, alias="CHAT_RETENTION_LIMIT")
    # 压缩后希望保留的消息条数（应小于 retention_limit，避免每条消息都触发压缩）。
    chat_prune_target: int = Field(default=800, alias="CHAT_PRUNE_TARGET")

    class Config:
        # 指定 .env 读取位置与编码。
        env_file = ".env"
        env_file_encoding = "utf-8"


# 单例配置对象：其他模块直接 `from .settings import settings` 使用。
settings = Settings()
