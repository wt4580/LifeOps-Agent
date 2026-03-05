from __future__ import annotations

"""lifeops.models

这个文件定义了项目的数据库表结构（SQLAlchemy ORM）。

你可以把每个类理解成“一张表的 Python 映射”：
- 类名 = 业务概念
- 字段 = 表列
- `mapped_column(...)` = 列类型和约束
"""

from datetime import datetime
from sqlalchemy import String, Integer, DateTime, Text, Float
from sqlalchemy.orm import Mapped, mapped_column

from .db import Base


class ChatMessage(Base):
    """聊天消息表：记录 user/assistant 的每一条对话。"""

    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)  # 消息的唯一标识
    session_id: Mapped[str] = mapped_column(String(64), index=True)  # 会话 ID，用于区分不同的对话
    role: Mapped[str] = mapped_column(String(16))  # 消息的角色（用户或助手）
    content: Mapped[str] = mapped_column(Text)  # 消息的内容
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)  # 消息的创建时间


class TodoItem(Base):
    """待办表：用户确认后真正入库的任务项。"""

    __tablename__ = "todo_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)  # 待办事项的唯一标识
    title: Mapped[str] = mapped_column(String(255))  # 待办事项的标题
    due_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)  # 待办事项的截止时间
    source: Mapped[str | None] = mapped_column(String(64), nullable=True)  # 待办事项的来源（如用户输入或计划提案）
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)  # 待办事项的创建时间


class MemoryCandidate(Base):
    """记忆候选表：对话后自动抽取的结构化信息（候选态）。"""

    __tablename__ = "memory_candidates"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)  # 候选记忆的唯一标识
    kind: Mapped[str] = mapped_column(String(64))  # 候选记忆的类型（如事件、任务等）
    title: Mapped[str] = mapped_column(String(255))  # 候选记忆的标题
    occurred_at: Mapped[str | None] = mapped_column(String(64), nullable=True)  # 候选记忆发生的时间
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)  # 候选记忆的备注
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)  # 候选记忆的创建时间


class DocumentChunk(Base):
    """知识库分块表：保存可检索的文本片段与来源信息。"""

    __tablename__ = "document_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)  # 内容块的唯一标识
    file_path: Mapped[str] = mapped_column(String(512), index=True)  # 文档的路径
    page: Mapped[int | None] = mapped_column(Integer, nullable=True)  # 文档的页码（如果适用）
    chunk_text: Mapped[str] = mapped_column(Text)  # 内容块的文本内容
    chunk_hash: Mapped[str] = mapped_column(String(64), index=True)  # 内容块的哈希值，用于去重
    score: Mapped[float | None] = mapped_column(Float, nullable=True)  # 内容块的相关性评分
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)  # 内容块的创建时间


class ConversationSummary(Base):
    """会话摘要表：保存中期记忆摘要，减少长对话 token 压力。"""

    __tablename__ = "conversation_summaries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)  # 摘要的唯一标识
    session_id: Mapped[str] = mapped_column(String(64), index=True)  # 会话 ID
    summary_text: Mapped[str] = mapped_column(Text)  # 摘要的文本内容
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)  # 摘要的更新时间


# 说明：
# - trace 是响应层的可审计信息，不属于持久化业务实体，因此不建表。
# - 如果后续要做离线分析，可新增 trace_events 表专门落库。
