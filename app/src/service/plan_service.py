"""计划服务层 - 封装待办计划相关的业务逻辑"""

from __future__ import annotations

import logging
from typing import Any

from ..util.planner.planner import propose_plan, detect_affirmation, detect_rejection

logger = logging.getLogger(__name__)


class PlanService:
    """计划服务
    
    职责：
    1. 检测用户意图（确认/拒绝）
    2. 生成待办草案
    3. 管理待办提案缓存
    """
    
    def __init__(self):
        self.proposal_cache: dict[str, list[dict]] = {}
        self.pending_intents: dict[str, str] = {}
    
    def detect_user_intent(self, message: str) -> dict:
        """检测用户对计划的意图（确认或拒绝）
        
        Args:
            message: 用户消息
            
        Returns:
            包含intent类型的字典
        """
        affirmation = detect_affirmation(message)
        rejection = detect_rejection(message)
        
        if affirmation:
            return {"intent": "confirm"}
        elif rejection:
            return {"intent": "reject"}
        else:
            return {"intent": "unknown"}
    
    def generate_proposal(self, text: str, session_id: str) -> dict:
        """生成待办草案
        
        Args:
            text: 待办描述文本
            session_id: 会话ID
            
        Returns:
            包含proposal_id和草案内容的字典
        """
        from uuid import uuid4
        
        proposal_id = str(uuid4())
        _, proposal = propose_plan(text)
        items = [item.model_dump() for item in proposal.items]
        
        # 存入缓存
        self.proposal_cache[proposal_id] = items
        
        return {
            "proposal_id": proposal_id,
            "proposal": items
        }
    
    def confirm_proposal(self, proposal_id: str, session_id: str | None = None) -> bool:
        """确认并保存待办草案到数据库
        
        Args:
            proposal_id: 提案ID
            session_id: 会话ID（可选）
            
        Returns:
            是否成功
        """
        from sqlalchemy import select
        from ..common.config.db_config import SessionLocal
        from ..domain.entity.chat_entity import TodoItem
        from datetime import datetime
        
        if proposal_id not in self.proposal_cache:
            logger.warning("Proposal %s not found in cache", proposal_id)
            return False
        
        items = self.proposal_cache[proposal_id]
        
        with SessionLocal() as db:
            for item_data in items:
                title = (item_data.get("title") or "").strip()
                if not title:
                    continue
                due_at = datetime.fromisoformat(item_data["due_at"]) if item_data.get("due_at") else None

                duplicate_stmt = select(TodoItem).where(
                    TodoItem.title == title,
                    TodoItem.session_id == session_id,
                    TodoItem.is_completed == False,
                )
                if due_at is None:
                    duplicate_stmt = duplicate_stmt.where(TodoItem.due_at.is_(None))
                else:
                    duplicate_stmt = duplicate_stmt.where(TodoItem.due_at == due_at)

                duplicate = db.execute(duplicate_stmt.limit(1)).scalars().first()
                if duplicate:
                    continue

                todo = TodoItem(
                    session_id=session_id,
                    title=title,
                    due_at=due_at,
                    source="plan_proposal",
                    is_completed=False,
                    completed_at=None,
                )
                db.add(todo)
            db.commit()
        
        # 从缓存中移除
        del self.proposal_cache[proposal_id]
        
        return True
    
    def store_pending_intent(self, session_id: str, intent_text: str):
        """存储待处理的待办意图
        
        Args:
            session_id: 会话ID
            intent_text: 待办意图文本
        """
        self.pending_intents[session_id] = intent_text
    
    def get_pending_intent(self, session_id: str) -> str | None:
        """获取待处理的待办意图
        
        Args:
            session_id: 会话ID
            
        Returns:
            待办意图文本，如果不存在则返回None
        """
        return self.pending_intents.get(session_id)
    
    def clear_pending_intent(self, session_id: str):
        """清除待处理的待办意图
        
        Args:
            session_id: 会话ID
        """
        if session_id in self.pending_intents:
            del self.pending_intents[session_id]


# 单例模式
plan_service = PlanService()
