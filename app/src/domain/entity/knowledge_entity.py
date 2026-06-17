from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from ...common.config.knowledge_db_config import KnowledgeBase


class DocumentChunk(KnowledgeBase):
    __tablename__ = "document_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    file_path: Mapped[str] = mapped_column(String(512), index=True)
    page: Mapped[int | None] = mapped_column(Integer, nullable=True)
    chunk_text: Mapped[str] = mapped_column(Text)
    chunk_hash: Mapped[str] = mapped_column(String(64), index=True)
    source_type: Mapped[str | None] = mapped_column(String(32), index=True, nullable=True)
    doc_topic: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    section_title: Mapped[str | None] = mapped_column(String(255), nullable=True)
    heading_level: Mapped[int | None] = mapped_column(Integer, nullable=True)
    is_table: Mapped[bool | None] = mapped_column(Integer, nullable=True)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
