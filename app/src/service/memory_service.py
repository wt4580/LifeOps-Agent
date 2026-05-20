"""记忆服务层 - 封装记忆管理相关的业务逻辑"""

from __future__ import annotations

import json
import logging
from typing import Any

from sqlalchemy import select

from ..common.config.db_config import SessionLocal
from ..domain.entity.models import LifeEvent, UserProfile, MemoryCandidate

logger = logging.getLogger(__name__)


class MemoryService:
    """记忆管理服务
    
    职责：
    1. 管理个人生活事件
    2. 管理用户画像
    3. 管理记忆候选
    """
    
    def get_life_events(self, session_id: str, limit: int = 30) -> list[dict]:
        """获取个人生活事件
        
        Args:
            session_id: 会话ID（作为记忆owner）
            limit: 返回数量限制
            
        Returns:
            生活事件列表
        """
        with SessionLocal() as db:
            rows = (
                db.execute(
                    select(LifeEvent)
                    .where(LifeEvent.session_id == session_id)
                    .order_by(LifeEvent.created_at.desc())
                    .limit(limit)
                )
                .scalars()
                .all()
            )
        
        rows.reverse()
        out: list[dict] = []
        for r in rows:
            try:
                tags = json.loads(r.tags_json) if r.tags_json else []
            except json.JSONDecodeError:
                tags = []
            out.append(
                {
                    "id": r.id,
                    "category": r.category,
                    "title": r.title,
                    "event_time": r.event_time.isoformat() if r.event_time else None,
                    "amount": r.amount,
                    "amount_unit": r.amount_unit,
                    "tags": tags,
                    "notes": r.notes,
                }
            )
        return out
    
    def get_user_profile(self, session_id: str) -> dict | None:
        """获取用户画像
        
        Args:
            session_id: 会话ID（作为记忆owner）
            
        Returns:
            用户画像字典，如果不存在则返回None
        """
        with SessionLocal() as db:
            row = (
                db.execute(select(UserProfile).where(UserProfile.session_id == session_id).limit(1))
                .scalars()
                .first()
            )

        if not row:
            return None

        try:
            preferences = json.loads(row.preferences_json) if row.preferences_json else []
        except json.JSONDecodeError:
            preferences = []
        try:
            conditions = json.loads(row.conditions_json) if row.conditions_json else []
        except json.JSONDecodeError:
            conditions = []

        return {
            "height_cm": row.height_cm,
            "weight_kg": row.weight_kg,
            "preferences": preferences,
            "conditions": conditions,
            "notes": row.notes,
        }
    
    def get_memory_candidates(self, limit: int = 50) -> list[dict]:
        """获取记忆候选列表
        
        Args:
            limit: 返回数量限制
            
        Returns:
            记忆候选列表
        """
        with SessionLocal() as db:
            rows = (
                db.execute(
                    select(MemoryCandidate)
                    .order_by(MemoryCandidate.created_at.desc())
                    .limit(limit)
                )
                .scalars()
                .all()
            )
        
        return [
            {
                "id": r.id,
                "kind": r.kind,
                "title": r.title,
                "occurred_at": r.occurred_at.isoformat() if r.occurred_at else None,
                "notes": r.notes,
            }
            for r in rows
        ]


# 单例模式
memory_service = MemoryService()
