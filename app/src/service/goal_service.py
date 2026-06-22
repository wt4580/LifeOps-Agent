"""目标拆解服务层 - 封装目标相关的业务逻辑"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from sqlalchemy import select

from ..common.config.db_config import SessionLocal
from ..common.config.base_config import settings
from ..common.config.log_config import logger
from ..domain.entity.chat_entity import GoalItem


class GoalService:
    """目标拆解服务

    职责：
    1. 存储从对话中提取的目标
    2. 查询活跃目标
    3. 更新目标进度
    4. 构建目标上下文描述
    """

    def _serialize_goal(self, g: GoalItem) -> dict:
        return {
            "id": g.id,
            "title": g.title,
            "category": g.category,
            "target_date": g.target_date.isoformat() if g.target_date else None,
            "status": g.status,
            "progress_pct": g.progress_pct,
            "sub_goals": json.loads(g.sub_goals_json) if g.sub_goals_json else [],
            "milestones": json.loads(g.milestones_json) if g.milestones_json else [],
            "notes": g.notes,
            "created_at": g.created_at.isoformat() if g.created_at else None,
            "updated_at": g.updated_at.isoformat() if g.updated_at else None,
        }

    def _apply_scope_filter(self, stmt, session_id: str):
        scope = (settings.personal_memory_scope or "global").strip().lower()
        session_col = getattr(GoalItem, "session_id", None)
        if scope == "session" and session_col is not None:
            return stmt.where(session_col == session_id)
        return stmt

    def get_active_goals(self, session_id: str) -> list[dict]:
        """获取所有活跃目标（状态为 active）。"""
        with SessionLocal() as db:
            stmt = select(GoalItem).where(GoalItem.status == "active").order_by(GoalItem.created_at.desc())
            stmt = self._apply_scope_filter(stmt, session_id)
            goals = db.execute(stmt).scalars().all()
            return [self._serialize_goal(g) for g in goals]

    def get_goal_by_id(self, goal_id: int) -> dict | None:
        with SessionLocal() as db:
            g = db.get(GoalItem, goal_id)
            return self._serialize_goal(g) if g else None

    def save_goal(self, session_id: str, title: str, category: str | None = None,
                  target_date: str | None = None,
                  sub_goals: list[dict] | None = None,
                  milestones: list[dict] | None = None,
                  notes: str | None = None) -> dict:
        """保存一个新目标，并写入拆解结果。"""
        target_dt = None
        if target_date:
            try:
                target_dt = datetime.fromisoformat(target_date)
            except ValueError:
                pass
        with SessionLocal() as db:
            g = GoalItem(
                session_id=session_id,
                title=title,
                category=category,
                target_date=target_dt,
                status="active",
                progress_pct=0.0,
                sub_goals_json=json.dumps(sub_goals, ensure_ascii=False) if sub_goals else None,
                milestones_json=json.dumps(milestones, ensure_ascii=False) if milestones else None,
                notes=notes,
            )
            db.add(g)
            db.commit()
            db.refresh(g)
            logger.info("goal saved id=%s title=%s", g.id, title[:40])
            return self._serialize_goal(g)

    def update_progress(self, goal_id: int, progress_pct: float) -> bool:
        with SessionLocal() as db:
            g = db.get(GoalItem, goal_id)
            if not g:
                return False
            g.progress_pct = max(0.0, min(100.0, progress_pct))
            db.commit()
            return True

    def mark_completed(self, goal_id: int) -> bool:
        with SessionLocal() as db:
            g = db.get(GoalItem, goal_id)
            if not g:
                return False
            g.status = "completed"
            g.progress_pct = 100.0
            db.commit()
            return True

    def build_goal_context(self, goals: list[dict]) -> str:
        """将目标列表压缩为一段描述文本（供 normal_chat context 使用）。"""
        if not goals:
            return ""
        lines = ["🎯 活跃目标："]
        for i, g in enumerate(goals[:5], 1):
            parts = [f"{i}. {g['title']}"]
            if g.get("category"):
                parts.append(f"[{g['category']}]")
            if g.get("sub_goals"):
                subs = [s.get("title", "") for s in g["sub_goals"][:3]]
                parts.append("→ " + "、".join(subs))
            if g.get("progress_pct", 0) > 0:
                parts.append(f"({g['progress_pct']:.0f}%)")
            lines.append("  ".join(parts))
        return "\n".join(lines)


goal_service = GoalService()
