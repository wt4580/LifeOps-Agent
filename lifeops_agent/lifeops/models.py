from __future__ import annotations

from datetime import datetime
from sqlalchemy import String, Integer, DateTime, Text, Float
from sqlalchemy.orm import Mapped, mapped_column

from .db import Base

# 聊天消息模型，用于存储用户和助手的对话记录
class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)  # 消息的唯一标识
    session_id: Mapped[str] = mapped_column(String(64), index=True)  # 会话 ID，用于区分不同的对话
    role: Mapped[str] = mapped_column(String(16))  # 消息的角色（用户或助手）
    content: Mapped[str] = mapped_column(Text)  # 消息的内容
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)  # 消息的创建时间


# 待办事项模型，用于存储用户的待办任务
class TodoItem(Base):
    __tablename__ = "todo_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)  # 待办事项的唯一标识
    title: Mapped[str] = mapped_column(String(255))  # 待办事项的标题
    due_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)  # 待办事项的截止时间
    source: Mapped[str | None] = mapped_column(String(64), nullable=True)  # 待办事项的来源（如用户输入或计划提案）
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)  # 待办事项的创建时间


# 记忆候选模型，用于存储从对话中提取的潜在记忆信息
class MemoryCandidate(Base):
    __tablename__ = "memory_candidates"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)  # 候选记忆的唯一标识
    kind: Mapped[str] = mapped_column(String(64))  # 候选记忆的类型（如事件、任务等）
    title: Mapped[str] = mapped_column(String(255))  # 候选记忆的标题
    occurred_at: Mapped[str | None] = mapped_column(String(64), nullable=True)  # 候选记忆发生的时间
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)  # 候选记忆的备注
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)  # 候选记忆的创建时间


# 文档内容块模型，用于存储文档的分块内容，支持检索
class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)  # 内容块的唯一标识
    file_path: Mapped[str] = mapped_column(String(512), index=True)  # 文档的路径
    page: Mapped[int | None] = mapped_column(Integer, nullable=True)  # 文档的页码（如果适用）
    chunk_text: Mapped[str] = mapped_column(Text)  # 内容块的文本内容
    chunk_hash: Mapped[str] = mapped_column(String(64), index=True)  # 内容块的哈希值，用于去重
    score: Mapped[float | None] = mapped_column(Float, nullable=True)  # 内容块的相关性评分
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)  # 内容块的创建时间


# 会话摘要模型，用于存储对话的摘要信息
class ConversationSummary(Base):
    __tablename__ = "conversation_summaries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)  # 摘要的唯一标识
    session_id: Mapped[str] = mapped_column(String(64), index=True)  # 会话 ID
    summary_text: Mapped[str] = mapped_column(Text)  # 摘要的文本内容
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)  # 摘要的更新时间

# （可选）为了给前端展示 Router 的决策轨迹，我们在响应层面会返回 trace。
# 这里的 ORM models 不需要改表结构；trace 不落库。
