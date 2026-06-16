"""聊天服务层 - 封装所有聊天相关的业务逻辑"""
from __future__ import annotations
import asyncio
import json
from datetime import datetime, timedelta
from typing import Any
from uuid import uuid4
from sqlalchemy import select, func, delete
from ..common.config.db_config import SessionLocal
from ..common.config.base_config import settings
import app.src.common.config.app_config as app_cfg
from ..domain.entity.chat_entity import (
    ChatMessage, TodoItem, MemoryCandidate,
    ConversationSummary, LifeEvent, UserProfile
)
from ..util.llm.llm_qwen import chat_completion
from ..util.memory.memory_extractor import extract_memory_from_dialogue
from ..util.memory.personal_memory import (
    build_profile_context,
    extract_personal_events,
    extract_profile_facts,
    generate_profile_pre_advice,
    parse_event_time,
    decide_proactive_advice,
)
from ..util.graph_chat import aload_chat_checkpoint_context, arun_chat_graph, load_chat_checkpoint_context
from ..util.retrieval.retrieval import index_documents
from .plan_service import plan_service
from ..common.config.log_config import logger

class ChatService:
    """聊天核心服务"""

    def __init__(self):
        # 服务只保留自身参数，不直接持有 graph；graph 在应用启动阶段统一创建。
        self.summary_refresh_turns = 10

    # ----------------------------------------------------------------------
    # 主聊天流程
    # ----------------------------------------------------------------------

    async def chat(self, message: str, session_id: str | None = None) -> dict:
        """执行一次完整聊天请求。

        这条链路会先保存用户消息，再抽取画像/事件/预建议，
        最后交给 LangGraph 统一做路由、工具调用和 HITL 处理。
        """
        session_id = session_id or str(uuid4())
        # 优先使用 checkpointer 中恢复的上下文，避免只依赖数据库回读。
        checkpoint_context = await aload_chat_checkpoint_context(app_cfg.chat_graph, session_id) if app_cfg.chat_graph else {}

        # A) 用户消息先落库，保证审计链完整。
        with SessionLocal() as db:
            db.add(ChatMessage(session_id=session_id, role="user", content=message))
            db.commit()

        # B) 前置处理：尽量在进图之前补齐可用上下文。
        recent_for_profile = checkpoint_context.get("recent_dialogue") or self._load_recent_dialogue(session_id, limit=12)
        memory_owner_id = self._resolve_memory_owner_id(session_id)
        now_dt = datetime.now()
        now_iso = now_dt.isoformat(timespec="seconds")

        # 画像抽取：记录稳定偏好、身高体重、限制条件等信息。
        profile_patch = extract_profile_facts(
            message, recent_for_profile, now_iso=now_iso, session_id=session_id,
        )
        profile_changed = self._upsert_user_profile(
            memory_owner_id, profile_patch.model_dump() if profile_patch else None
        )
        profile = self._load_user_profile(memory_owner_id)
        profile_context = build_profile_context(profile)

        # 个人事件抽取：把事实性内容存下来，供后续建议与画像分析使用。
        personal = extract_personal_events(
            message, recent_for_profile, now_iso=now_iso, session_id=session_id,
        )
        personal_inserted = 0
        if personal and personal.items:
            personal_inserted = self._store_personal_events(
                memory_owner_id, message, [item.model_dump() for item in personal.items],
            )

        # 近期事件窗口：不要只看固定条数，尽量按时间范围判断上下文。
        recent_events, event_window_start, event_window_end = self._load_life_events_in_window(
            session_id=memory_owner_id,
            hours=settings.proactive_event_window_hours,
            now_dt=now_dt,
        )
        # 基于画像与近期事件生成一个前置建议，交给 Graph 作为软提示。
        pre_advice = generate_profile_pre_advice(
            user_message=message,
            profile_context=profile_context,
            recent_events=recent_events,
            now_iso=now_iso,
            session_id=session_id,
        )

        pre_steps = [
            {
                "type": "preprocess",
                "name": "profile_context",
                "ts": datetime.now().isoformat(timespec="seconds"),
                "output": {
                    "profile_changed": profile_changed,
                    "has_profile": bool(profile_context),
                    "profile_context": profile_context or "",
                    "memory_scope": settings.personal_memory_scope,
                    "memory_owner_id": memory_owner_id,
                },
            },
            {
                "type": "preprocess",
                "name": "personal_events",
                "ts": datetime.now().isoformat(timespec="seconds"),
                "output": {
                    "inserted": personal_inserted,
                    "recent_events_count": len(recent_events),
                    "event_window_hours": settings.proactive_event_window_hours,
                    "event_window_start": event_window_start,
                    "event_window_end": event_window_end,
                },
            },
            {
                "type": "preprocess",
                "name": "pre_graph_hint",
                "ts": datetime.now().isoformat(timespec="seconds"),
                "output": {
                    "pre_advice": pre_advice or "NONE",
                    "source": "profile+life_events",
                },
            },
        ]

          # C) 执行状态机：统一交给 LangGraph 处理路由、工具调用、HITL 和主动建议。
        # 将主动决策所需的上下文传递给 Graph，由 Graph 内部的 proactive_decision 节点处理。
        upcoming_todos = self._load_upcoming_todos_for_advice(
            session_id=memory_owner_id,
            hours=settings.proactive_todo_hours,
        )
        
        state = await arun_chat_graph(
            compiled_graph=app_cfg.chat_graph,   # ← 用导入的全局实例
            session_id=session_id,
            user_message=message,
            profile_context=profile_context or None,
            pre_advice=pre_advice,
            pre_steps=pre_steps,
            recent_events=recent_events,
            upcoming_todos=upcoming_todos,
            event_window_start=event_window_start,
            event_window_end=event_window_end,
        )
        answer = state.get("answer") or ""
        proposal_id = state.get("proposal_id")
        proposal_payload = state.get("proposal")
        if proposal_id and isinstance(proposal_payload, dict):
            items = proposal_payload.get("items") or []
            if isinstance(items, list):
                # 图里生成的待办草案缓存起来，供确认接口继续使用。
                plan_service.proposal_cache[proposal_id] = items

        used_tool = state.get("used_tool")
        citations = state.get("citations") or []
        trace = state.get("trace")

        
        # D) 保存助手消息：和 user 消息配对入库，方便后续摘要与回放。
        with SessionLocal() as db:
            db.add(ChatMessage(session_id=session_id, role="assistant", content=answer))
            db.commit()

        # E) 按轮次刷新摘要，并在必要时裁剪历史，控制长期上下文长度。
        await self._maybe_update_summary(session_id)
        self._maybe_prune_chat_history(session_id)

        # F) 旁路增强：抽取结构化记忆候选，但不阻塞主响应返回。
        latest_checkpoint_context = await aload_chat_checkpoint_context(app_cfg.chat_graph, session_id) if app_cfg.chat_graph else {}
        dialogue = latest_checkpoint_context.get("recent_dialogue") or self._load_recent_dialogue(session_id, limit=8)
        extraction = extract_memory_from_dialogue(dialogue)
        if extraction:
            with SessionLocal() as db:
                for item in extraction.items:
                    db.add(
                        MemoryCandidate(
                            kind=item.kind,
                            title=item.title,
                            occurred_at=item.occurred_at,
                            notes=item.notes,
                        )
                    )
                db.commit()

            # G) 返回前端所需结果。
        return {
            "answer": answer,
            "session_id": session_id,
            "proposal_id": proposal_id,
            "proposal": proposal_payload,
            "used_tool": used_tool,
            "citations": citations,
            "trace": trace,
        }

    async def chat_stream(self, message: str, session_id: str | None = None):
        """以 SSE 事件流方式返回聊天结果。

        先持续发送 status 事件，再在主任务完成后按片段输出答案。
        这样前端看起来更像“流式回复”。
        """

        # 轮播式提示文案，减少用户等待时的空白感。
        progress_messages = [
            "正在分析你的问题...",
            "正在检索可用信息...",
            "正在组织回答内容...",
        ]
        progress_index = 0
        task = asyncio.create_task(self.chat(message=message, session_id=session_id))

        # 主任务未结束前，周期性发送状态消息。
        while not task.done():
            msg = progress_messages[progress_index % len(progress_messages)]
            progress_index += 1
            yield f"data: {json.dumps({'type': 'status', 'message': msg}, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.7)

        # 主任务结束后，把答案拆成更小的片段输出。
        result = await task
        answer = result.get("answer") or ""
        chunks = self._chunk_text(answer)

        if not chunks:
            yield f"data: {json.dumps({'type': 'final', 'result': result}, ensure_ascii=False)}\n\n"
            return

        for chunk in chunks:
            yield f"data: {json.dumps({'type': 'delta', 'delta': chunk}, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.01)

        yield f"data: {json.dumps({'type': 'final', 'result': result}, ensure_ascii=False)}\n\n"

    def _chunk_text(self, text: str) -> list[str]:
        """把长文本拆成适合 SSE 推送的小片段。"""
        text = (text or "").strip()
        if not text:
            return []

        chunks: list[str] = []
        buffer = ""
        for char in text:
            buffer += char
            if char in {"。", "！", "？", "\n"} or len(buffer) >= 24:
                chunks.append(buffer)
                buffer = ""
        if buffer:
            chunks.append(buffer)
        return chunks

    def _load_recent_dialogue(self, session_id: str, limit: int = 12) -> list[dict]:
        """读取最近 N 条对话消息。"""
        with SessionLocal() as db:
            rows = (
                db.execute(
                    select(ChatMessage)
                    .where(ChatMessage.session_id == session_id)
                    .order_by(ChatMessage.created_at.desc())
                    .limit(limit)
                )
                .scalars()
                .all()
            )
        
        rows.reverse()
        return [{"role": r.role, "content": r.content} for r in rows]
    
    def _resolve_memory_owner_id(self, session_id: str) -> str:
        """解析记忆所有者 ID。

        `session` 模式按会话隔离；`global` 模式则共享到统一 owner。
        """
        scope = (settings.personal_memory_scope or "global").strip().lower()
        if scope == "session":
            return session_id
        return "global"
    
    def _load_user_profile(self, session_id: str) -> dict | None:
        """读取用户画像，并还原成普通字典供上层使用。"""
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
    
    def _upsert_user_profile(self, session_id: str, patch: dict | None) -> bool:
        """增量更新用户画像。

        返回值表示这次是否真的写入了变化，方便上层判断。
        """
        if not patch:
            return False
        
        with SessionLocal() as db:
            row = (
                db.execute(select(UserProfile).where(UserProfile.session_id == session_id).limit(1))
                .scalars()
                .first()
            )
            if not row:
                row = UserProfile(session_id=session_id)
                db.add(row)
                db.flush()
            
            changed = False
            
            h = patch.get("height_cm")
            if h is not None and h > 0:
                row.height_cm = float(h)
                changed = True
            
            w = patch.get("weight_kg")
            if w is not None and w > 0:
                row.weight_kg = float(w)
                changed = True
            
            try:
                cur_prefs = json.loads(row.preferences_json) if row.preferences_json else []
            except json.JSONDecodeError:
                cur_prefs = []
            merged_prefs = self._merge_unique(cur_prefs, patch.get("preferences") or [])
            if merged_prefs != cur_prefs:
                row.preferences_json = json.dumps(merged_prefs, ensure_ascii=False)
                changed = True
            
            try:
                cur_conds = json.loads(row.conditions_json) if row.conditions_json else []
            except json.JSONDecodeError:
                cur_conds = []
            merged_conds = self._merge_unique(cur_conds, patch.get("conditions") or [])
            if merged_conds != cur_conds:
                row.conditions_json = json.dumps(merged_conds, ensure_ascii=False)
                changed = True
            
            notes = (patch.get("notes") or "").strip()
            if notes and notes != (row.notes or ""):
                row.notes = notes
                changed = True
            
            if changed:
                db.commit()
            else:
                db.rollback()
        
        return changed
    
    def _merge_unique(self, base: list[str], patch: list[str]) -> list[str]:
        """合并列表并去重，尽量保留原顺序。"""
        seen = {x.strip() for x in base if x and x.strip()}
        out = [x for x in base if x and x.strip()]
        for item in patch:
            t = (item or "").strip()
            if not t or t in seen:
                continue
            seen.add(t)
            out.append(t)
        return out
    
    def _store_personal_events(self, session_id: str, source_text: str, events: list[dict]) -> int:
        """存储个人事件候选，供后续主动建议与画像分析复用。"""
        if not events:
            return 0
        
        inserted = 0
        with SessionLocal() as db:
            for item in events:
                db.add(
                    LifeEvent(
                        session_id=session_id,
                        category=item.get("category", "other"),
                        title=item.get("title", ""),
                        event_time=parse_event_time(item.get("event_time")),
                        amount=item.get("amount"),
                        amount_unit=item.get("amount_unit"),
                        tags_json=json.dumps(item.get("tags", []), ensure_ascii=False),
                        notes=item.get("notes"),
                        source_text=source_text,
                    )
                )
                inserted += 1
            db.commit()
        return inserted
    
    def _load_life_events_in_window(self, session_id: str, hours: int, now_dt: datetime) -> tuple[list[dict], str, str]:
        """按时间窗口读取个人事件，避免只看固定条数导致时间上下文丢失。"""
        window_start = now_dt - timedelta(hours=hours)
        with SessionLocal() as db:
            rows = (
                db.execute(
                    select(LifeEvent)
                    .where(LifeEvent.session_id == session_id)
                    .where(func.coalesce(LifeEvent.event_time, LifeEvent.created_at) >= window_start)
                    .where(func.coalesce(LifeEvent.event_time, LifeEvent.created_at) <= now_dt)
                    .order_by(func.coalesce(LifeEvent.event_time, LifeEvent.created_at).asc())
                )
                .scalars()
                .all()
            )
        
        out: list[dict] = []
        for r in rows:
            try:
                tags = json.loads(r.tags_json) if r.tags_json else []
            except json.JSONDecodeError:
                tags = []
            out.append(
                {
                    "category": r.category,
                    "title": r.title,
                    "event_time": r.event_time.isoformat() if r.event_time else None,
                    "recorded_at": r.created_at.isoformat() if r.created_at else None,
                    "effective_time": (r.event_time or r.created_at).isoformat() if (r.event_time or r.created_at) else None,
                    "amount": r.amount,
                    "amount_unit": r.amount_unit,
                    "tags": tags,
                    "notes": r.notes,
                }
            )
        
        return (
            out,
            window_start.isoformat(timespec="seconds"),
            now_dt.isoformat(timespec="seconds"),
        )
    
    def _load_upcoming_todos_for_advice(self, session_id: str, hours: int = 24) -> list[dict]:
        """读取未来待办事项，用于判断是否要主动提醒用户。"""
        now = datetime.now()
        end = now + timedelta(hours=hours)
        with SessionLocal() as db:
            stmt = (
                select(TodoItem)
                .where(TodoItem.due_at != None)
                .where(TodoItem.due_at >= now)
                .where(TodoItem.due_at < end)
                .order_by(TodoItem.due_at.asc())
            )
            scope = (settings.personal_memory_scope or "global").strip().lower()
            session_col = getattr(TodoItem, "session_id", None)
            if scope == "session" and session_col is not None:
                stmt = stmt.where(session_col == session_id)
            
            rows = db.execute(stmt).scalars().all()
        
        return [{"title": r.title, "due_at": r.due_at.isoformat() if r.due_at else None} for r in rows]
    
    def _get_summary(self, session_id: str) -> str | None:
        """获取最近一条会话摘要。"""
        with SessionLocal() as db:
            row = (
                db.execute(
                    select(ConversationSummary)
                    .where(ConversationSummary.session_id == session_id)
                    .order_by(ConversationSummary.updated_at.desc())
                    .limit(1)
                )
                .scalars()
                .first()
            )
        return row.summary_text if row else None
    
    def _save_summary(self, session_id: str, summary_text: str) -> None:
        """保存会话摘要。"""
        with SessionLocal() as db:
            db.add(ConversationSummary(session_id=session_id, summary_text=summary_text))
            db.commit()
    
    async def _maybe_update_summary(self, session_id: str) -> None:
        """按轮次定期刷新摘要，避免长对话无限增长。"""
        with SessionLocal() as db:
            count = db.execute(
                select(func.count()).select_from(ChatMessage).where(ChatMessage.session_id == session_id)
            ).scalar_one()
        
        if count > 0 and count % self.summary_refresh_turns == 0:
            # 使用异步版本加载 checkpoint 上下文
            checkpoint_context = await aload_chat_checkpoint_context(app_cfg.chat_graph, session_id) if app_cfg.chat_graph else {}
            recent = checkpoint_context.get("recent_dialogue") or self._load_recent_dialogue(
                session_id, limit=self.summary_refresh_turns * 2
            )
            if not recent:
                return
            
            prompt = (
                "Summarize the conversation for later retrieval. "
                "Focus on commitments, plans, preferences, and open questions. "
                "Return a concise paragraph."
            )
            
            summary_text = chat_completion(
                [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": json.dumps(recent, ensure_ascii=False)},
                ],
                temperature=0.2,
            )
            self._save_summary(session_id, summary_text)
    
    def _maybe_prune_chat_history(self, session_id: str) -> None:
        """滚动压缩聊天记录。

        当消息过多时，把更早的内容压缩成摘要并删除原消息，控制上下文体积。
        """
        with SessionLocal() as db:
            rows = (
                db.execute(
                    select(ChatMessage)
                    .where(ChatMessage.session_id == session_id)
                    .order_by(ChatMessage.created_at.asc())
                )
                .scalars()
                .all()
            )
        
        total = len(rows)
        if total <= settings.chat_retention_limit:
            return
        
        target = settings.chat_prune_target
        if target >= settings.chat_retention_limit:
            target = max(100, settings.chat_retention_limit - 100)
        
        prune_count = max(0, total - target)
        if prune_count <= 0:
            return
        
        old_rows = rows[:prune_count]
        old_dialogue = [{"role": r.role, "content": r.content} for r in old_rows]
        
        summary_text = ""
        try:
            summary_text = chat_completion(
                [
                    {
                        "role": "system",
                        "content": (
                            "请把这段历史对话压缩成长期记忆摘要。"
                            "重点保留：个人偏好、长期目标、反复出现的任务、已确认事实。"
                            "输出一段简洁中文。"
                        ),
                    },
                    {"role": "user", "content": json.dumps(old_dialogue, ensure_ascii=False)},
                ],
                temperature=0.2,
            )
        except Exception as exc:
            logger.warning("Chat prune summary failed: %s", exc)
        
        if summary_text.strip():
            self._save_summary(session_id, f"[rolling-prune] {summary_text.strip()}")
        
        ids = [r.id for r in old_rows]
        with SessionLocal() as db:
            db.execute(delete(ChatMessage).where(ChatMessage.id.in_(ids)))
            db.commit()
        
        logger.info(
            "Pruned chat history session_id=%s total=%s pruned=%s kept=%s",
            session_id,
            total,
            prune_count,
            total - prune_count,
        )


# 导出单例
chat_service = ChatService()
