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
from ..common.config.chat_graph_holder import get_chat_graph
from ..domain.entity.chat_entity import (
    ChatMessage, TodoItem, MemoryCandidate,
    ConversationSummary, LifeEvent, UserProfile
)
from ..common.config.llm_config import chat_completion
from ..agent.memory.memory_extractor import extract_memory_from_dialogue
from ..agent.memory.personal_memory import (
    build_profile_context,
    extract_personal_events,
    extract_profile_facts,
    generate_profile_pre_advice,
    parse_event_time,
    decide_proactive_advice,
)
from ..agent.graph_chat import aload_chat_checkpoint_context, arun_chat_graph, load_chat_checkpoint_context
from ..util.retrieval.retrieval import index_documents
from .plan_service import plan_service
from ..common.config.log_config import logger
from .todo_service import reminder_service

class ChatService:
    """聊天核心服务"""

    def __init__(self):
        self.summary_refresh_turns = 10
        self.deep_refresh_interval = 3
        self.cleanup_interval = 20
        self._turn_counter: dict[str, int] = {}
        self._cleanup_counter = 0

    # ----------------------------------------------------------------------
    # 主聊天流程
    # ----------------------------------------------------------------------

    async def chat(self, message: str, session_id: str | None = None, progress_queue: asyncio.Queue | None = None) -> dict:
        """执行一次完整聊天请求。

        这条链路会先保存用户消息，再抽取画像/事件/预建议，
        最后交给 LangGraph 统一做路由、工具调用和 HITL 处理。
        """
        def _push(msg: str):
            if progress_queue is not None:
                progress_queue.put_nowait(msg)

        session_id = session_id or str(uuid4())
        # 优先使用 checkpointer 中恢复的上下文，避免只依赖数据库回读。
        checkpoint_context = await aload_chat_checkpoint_context(get_chat_graph(), session_id) if get_chat_graph() else {}

        # A) 用户消息先落库，保证审计链完整。
        _push("正在保存消息...")
        with SessionLocal() as db:
            db.add(ChatMessage(session_id=session_id, role="user", content=message))
            db.commit()

        # B) 前置处理：尽量在进图之前补齐可用上下文。
        _push("正在分析输入...")
        recent_for_profile = checkpoint_context.get("recent_dialogue") or self._load_recent_dialogue(session_id, limit=12)
        memory_owner_id = self._resolve_memory_owner_id(session_id)
        now_dt = datetime.now()
        now_iso = now_dt.isoformat(timespec="seconds")

        # 画像抽取：记录稳定偏好、身高体重、限制条件等信息。
        _push("正在更新用户画像...")
        profile_patch = extract_profile_facts(
            message, recent_for_profile, now_iso=now_iso, session_id=session_id,
        )
        profile_changed = self._upsert_user_profile(
            memory_owner_id, profile_patch.model_dump() if profile_patch else None
        )
        profile = self._load_user_profile(memory_owner_id)
        profile_context = build_profile_context(profile)

        # 个人事件抽取：把事实性内容存下来，供后续建议与画像分析使用。
        _push("正在提取生活事件...")
        personal = extract_personal_events(
            message, recent_for_profile, now_iso=now_iso, session_id=session_id,
        )
        personal_inserted = 0
        if personal and personal.items:
            personal_inserted = self._store_personal_events(
                memory_owner_id, message, [item.model_dump() for item in personal.items],
            )

        # 近期事件窗口：不要只看固定条数，尽量按时间范围判断上下文。
        _push("正在整理背景信息...")
        recent_events, event_window_start, event_window_end = self._load_life_events_in_window(
            session_id=memory_owner_id,
            hours=settings.proactive_event_window_hours,
            now_dt=now_dt,
        )
        # 加载待提醒事项（DB 持久化的提醒，按时间退避）。
        pending_reminders = reminder_service.load_pending_reminders(memory_owner_id)

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
            {
                "type": "preprocess",
                "name": "pending_reminders",
                "ts": datetime.now().isoformat(timespec="seconds"),
                "output": {
                    "count": len(pending_reminders),
                    "reminders": pending_reminders,
                },
            },
        ]

          # C) 执行状态机：统一交给 LangGraph 处理路由、工具调用、HITL 和主动建议。
        # 将主动决策所需的上下文传递给 Graph，由 Graph 内部的 proactive_decision 节点处理。
        upcoming_todos = self._load_upcoming_todos_for_advice(
            session_id=memory_owner_id,
            hours=settings.proactive_todo_hours,
        )
        
        _push("正在进入思考流程...")
        chat_graph = get_chat_graph()
        checkpoint_context = await aload_chat_checkpoint_context(chat_graph, session_id) if chat_graph else {}
        pending_proposal_id = checkpoint_context.get("proposal_id")

        state = await arun_chat_graph(
            compiled_graph=chat_graph,
            session_id=session_id,
            user_message=message,
            progress_queue=progress_queue,
            profile_context=profile_context or None,
            pre_advice=pre_advice,
            pre_steps=pre_steps,
            recent_events=recent_events,
            upcoming_todos=upcoming_todos,
            pending_reminders=pending_reminders,
            event_window_start=event_window_start,
            event_window_end=event_window_end,
        )
        answer = state.get("answer") or ""
        proposal_id = state.get("proposal_id")
        proposal_payload = state.get("proposal")
        if proposal_id and isinstance(proposal_payload, dict):
            items = proposal_payload.get("items") or []
            if isinstance(items, list):
                plan_service.proposal_cache[proposal_id] = items

        if state.get("proposal_confirmed") and pending_proposal_id:
            plan_service.confirm_proposal(pending_proposal_id, session_id)
            proposal_id = None

        used_tool = state.get("used_tool")
        citations = state.get("citations") or []
        trace = state.get("trace")

        # C.5) 从回答中提取"补充建议"并持久化为提醒（DB 时间退避）。
        if answer:
            advice_section = self._extract_advice_from_answer(answer)
            if advice_section:
                reminder_service.upsert_reminder(memory_owner_id, advice_section)

        # D) 保存助手消息：和 user 消息配对入库，方便后续摘要与回放。
        _push("正在保存对话...")
        with SessionLocal() as db:
            db.add(ChatMessage(session_id=session_id, role="assistant", content=answer))
            db.commit()

        # E) 按轮次刷新摘要，并在必要时裁剪历史，控制长期上下文长度。
        await self._maybe_update_summary(session_id)
        self._maybe_prune_chat_history(session_id)

        # F) 旁路增强：抽取结构化记忆候选，但不阻塞主响应返回。
        _push("正在提取记忆...")
        latest_checkpoint_context = await aload_chat_checkpoint_context(get_chat_graph(), session_id) if get_chat_graph() else {}
        dialogue = latest_checkpoint_context.get("recent_dialogue") or self._load_recent_dialogue(session_id, limit=8)
        extraction = extract_memory_from_dialogue(dialogue)

        # G) 周期性深度反思：分析所有积累数据，推断隐含模式并更新画像。
        _push("正在分析深层模式...")
        await self._maybe_deep_reflection(session_id, memory_owner_id)

        # H) 定期清理过时数据，防止表无限膨胀。
        self._cleanup_counter += 1
        if self._cleanup_counter % self.cleanup_interval == 0:
            self._maybe_cleanup_stale_data()
        if extraction:
            with SessionLocal() as db:
                for item in extraction.items:
                    db.add(
                        MemoryCandidate(
                            kind=item.kind,
                            title=item.title,
                            occurred_at=item.occurred_at,
                            notes=item.notes,
                            confidence=item.confidence,
                            insight_type=item.insight_type,
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
        progress_queue: asyncio.Queue = asyncio.Queue()
        task = asyncio.create_task(self.chat(message=message, session_id=session_id, progress_queue=progress_queue))

        last_status = ""
        while not task.done():
            drained = False
            while True:
                try:
                    msg = progress_queue.get_nowait()
                    last_status = msg
                    if isinstance(msg, dict):
                        yield f"data: {json.dumps({'type': msg.get('type', 'status'), 'message': msg.get('message', '')}, ensure_ascii=False)}\n\n"
                    else:
                        yield f"data: {json.dumps({'type': 'status', 'message': msg}, ensure_ascii=False)}\n\n"
                    drained = True
                except asyncio.QueueEmpty:
                    break
            if not task.done():
                await asyncio.sleep(0.3 if not drained else 0.05)

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

    @staticmethod
    def _extract_advice_from_answer(answer: str) -> str | None:
        """从回答中提取"补充建议"文本，若无则返回 None。"""
        marker = "补充建议："
        if marker not in answer:
            return None
        idx = answer.rfind(marker)
        text = answer[idx + len(marker):].strip()
        if not text:
            return None
        return text

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

            for scalar_field in ("age", "gender", "occupation", "city", "sleep_schedule", "exercise_habits", "work_hours", "family_status"):
                val = patch.get(scalar_field)
                if val is not None:
                    setattr(row, scalar_field, val)
                    changed = True

            for json_field, key in [("diet_json", "diet"), ("allergies_json", "allergies"), ("goals_json", "goals")]:
                val = patch.get(key)
                if val:
                    current = json.loads(getattr(row, json_field)) if getattr(row, json_field) else []
                    merged = self._merge_unique(current, val)
                    if merged != current:
                        setattr(row, json_field, json.dumps(merged, ensure_ascii=False))
                        changed = True

            if changed:
                db.commit()
            else:
                db.rollback()
        
        return changed
    
    def _merge_unique(self, base: list[str], patch: list[str]) -> list[str]:
        """合并列表并去重，尽量保留原顺序。

        也会处理偏好冲突：如果新增了"不喜欢X"，自动移除对应的"喜欢X"。
        """
        seen = {x.strip() for x in base if x and x.strip()}
        out = [x for x in base if x and x.strip()]
        for item in patch:
            t = (item or "").strip()
            if not t or t in seen:
                continue
            contradiction = self._find_contradiction(t, out)
            if contradiction:
                out = [x for x in out if x != contradiction]
                seen.discard(contradiction)
            seen.add(t)
            out.append(t)
        return out

    @staticmethod
    def _find_contradiction(new_pref: str, existing: list[str]) -> str | None:
        """检测新增偏好与已有偏好是否矛盾，返回需移除的旧条目。"""
        new_clean = new_pref.replace("不喜欢", "").replace("讨厌", "").replace("不爱", "").replace("不吃", "").strip()
        if new_clean and new_pref != new_clean:
            for old in existing:
                old_clean = old.replace("喜欢", "").replace("爱吃", "").replace("爱", "").strip()
                if old_clean and old_clean == new_clean:
                    return old
                if new_clean in old or old_clean in new_pref:
                    return old
        return None
    
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
    
    async def _maybe_deep_reflection(self, session_id: str, owner_id: str) -> None:
        turns = self._turn_counter
        turns[session_id] = turns.get(session_id, 0) + 1
        if turns[session_id] % self.deep_refresh_interval != 0:
            return

        with SessionLocal() as db:
            memories = db.execute(
                select(MemoryCandidate).order_by(MemoryCandidate.created_at.desc()).limit(30)
            ).scalars().all()
            events = db.execute(
                select(LifeEvent).where(LifeEvent.session_id == owner_id)
                .order_by(LifeEvent.event_time.desc().nulls_last()).limit(20)
            ).scalars().all()
            todos = db.execute(
                select(TodoItem).where(TodoItem.is_completed == False)
                .order_by(TodoItem.due_at.asc()).limit(10)
            ).scalars().all()
            profile_row = db.execute(
                select(UserProfile).where(UserProfile.session_id == owner_id).limit(1)
            ).scalars().first()

        profile_data = {}
        if profile_row:
            try:
                prefs = json.loads(profile_row.preferences_json) if profile_row.preferences_json else []
            except json.JSONDecodeError:
                prefs = []
            try:
                conds = json.loads(profile_row.conditions_json) if profile_row.conditions_json else []
            except json.JSONDecodeError:
                conds = []
            try:
                diet = json.loads(profile_row.diet_json) if profile_row.diet_json else []
            except json.JSONDecodeError:
                diet = []
            try:
                goals = json.loads(profile_row.goals_json) if profile_row.goals_json else []
            except json.JSONDecodeError:
                goals = []
            profile_data = {
                "preferences": prefs, "conditions": conds, "notes": profile_row.notes,
                "age": profile_row.age, "gender": profile_row.gender, "occupation": profile_row.occupation,
                "city": profile_row.city, "diet": diet, "sleep_schedule": profile_row.sleep_schedule,
                "exercise_habits": profile_row.exercise_habits, "work_hours": profile_row.work_hours,
                "family_status": profile_row.family_status, "goals": goals,
            }

        payload = {
            "memories": [{"kind": m.kind, "title": m.title, "notes": m.notes, "confidence": m.confidence, "insight_type": m.insight_type} for m in memories],
            "life_events": [{"title": e.title, "category": e.category, "tags": e.tags_json} for e in events],
            "pending_todos": [{"title": t.title, "due_at": str(t.due_at)} for t in todos],
            "current_profile": profile_data,
        }

        system_prompt = (
            "你是深度分析器。分析用户积累数据，推断隐含模式、偏好和习惯，更新画像。\n"
            "输出严格 JSON，不要解释。\n"
            'schema: {\n'
            '  "inferred_preferences":[{"name":"...","reason":"...","confidence":0-1}],\n'
            '  "inferred_conditions":[{"name":"...","reason":"...","confidence":0-1}],\n'
            '  "inferred_diet":[{"name":"...","reason":"...","confidence":0-1}],\n'
            '  "inferred_sleep_schedule":string|null,\n'
            '  "inferred_exercise_habits":string|null,\n'
            '  "inferred_work_hours":string|null,\n'
            '  "inferred_goals":[{"name":"...","reason":"...","confidence":0-1}],\n'
            '  "notes":"...或null"\n'
            "}\n"
            "规则：\n"
            "- 只提炼有足够证据支撑的推断（比如多次同类型事件）；\n"
            "- confidence 反映证据强度；\n"
            "- 已有画像中已确认的事实不要重复输出；\n"
            "- 若没有新推断，返回空列表。"
        )

        try:
            text = chat_completion(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                temperature=0.3,
                runtime_context={"scenario": "deep_reflection", "session_id": session_id},
            )
            data = json.loads(text)
            new_prefs = [f"[推断:{p['confidence']}] {p['name']}（{p['reason']}）" for p in data.get("inferred_preferences") or []]
            new_conds = [f"[推断:{c['confidence']}] {c['name']}（{c['reason']}）" for c in data.get("inferred_conditions") or []]
            notes = data.get("notes") or ""

            if new_prefs or new_conds or notes or data.get("inferred_diet") or data.get("inferred_sleep_schedule") or data.get("inferred_exercise_habits") or data.get("inferred_work_hours") or data.get("inferred_goals"):
                with SessionLocal() as db:
                    row = db.execute(
                        select(UserProfile).where(UserProfile.session_id == owner_id).limit(1)
                    ).scalars().first()
                    if row:
                        changed = False
                        if new_prefs:
                            try:
                                cur = json.loads(row.preferences_json) if row.preferences_json else []
                            except json.JSONDecodeError:
                                cur = []
                            merged = self._merge_unique(cur, new_prefs)
                            if merged != cur:
                                row.preferences_json = json.dumps(merged, ensure_ascii=False)
                                changed = True
                        if new_conds:
                            try:
                                cur = json.loads(row.conditions_json) if row.conditions_json else []
                            except json.JSONDecodeError:
                                cur = []
                            merged_conds = self._merge_unique(cur, new_conds)
                            if merged_conds != cur:
                                row.conditions_json = json.dumps(merged_conds, ensure_ascii=False)
                                changed = True
                        if notes:
                            current_notes = (row.notes or "") + f"\n[深度反思] {notes}" if row.notes else f"[深度反思] {notes}"
                            row.notes = current_notes.strip()
                            changed = True

                        for inferred_key, col in [
                            ("inferred_sleep_schedule", "sleep_schedule"),
                            ("inferred_exercise_habits", "exercise_habits"),
                            ("inferred_work_hours", "work_hours"),
                        ]:
                            val = data.get(inferred_key)
                            if val and val != getattr(row, col):
                                setattr(row, col, f"[推断] {val}")
                                changed = True

                        inferred_diet = data.get("inferred_diet") or []
                        if inferred_diet:
                            try:
                                cur = json.loads(row.diet_json) if row.diet_json else []
                            except json.JSONDecodeError:
                                cur = []
                            new_diet = [f"[推断:{d['confidence']}] {d['name']}（{d['reason']}）" for d in inferred_diet]
                            merged = self._merge_unique(cur, new_diet)
                            if merged != cur:
                                row.diet_json = json.dumps(merged, ensure_ascii=False)
                                changed = True

                        inferred_goals = data.get("inferred_goals") or []
                        if inferred_goals:
                            try:
                                cur = json.loads(row.goals_json) if row.goals_json else []
                            except json.JSONDecodeError:
                                cur = []
                            new_goals = [f"[推断:{g['confidence']}] {g['name']}（{g['reason']}）" for g in inferred_goals]
                            merged = self._merge_unique(cur, new_goals)
                            if merged != cur:
                                row.goals_json = json.dumps(merged, ensure_ascii=False)
                                changed = True

                        if changed:
                            db.commit()
                            logger.info("Deep reflection updated profile for session=%s", session_id)
        except Exception as exc:
            logger.warning("Deep reflection failed session=%s err=%s", session_id, exc)

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

    def _maybe_cleanup_stale_data(self) -> None:
        try:
            cutoff = datetime.utcnow() - timedelta(days=30)
            with SessionLocal() as db:
                deleted_candidates = db.execute(
                    delete(MemoryCandidate).where(MemoryCandidate.created_at < cutoff)
                ).rowcount
                deleted_events = db.execute(
                    delete(LifeEvent).where(LifeEvent.created_at < cutoff)
                ).rowcount
            if deleted_candidates or deleted_events:
                logger.info("Cleaned up stale data: memory_candidates=%s life_events=%s", deleted_candidates, deleted_events)
        except Exception as exc:
            logger.warning("Stale data cleanup failed: %s", exc)

    def _save_summary(self, session_id: str, summary_text: str) -> None:
        with SessionLocal() as db:
            row = db.execute(
                select(ConversationSummary).where(ConversationSummary.session_id == session_id).limit(1)
            ).scalars().first()
            if row:
                row.summary_text = summary_text
                row.updated_at = datetime.utcnow()
            else:
                db.add(ConversationSummary(session_id=session_id, summary_text=summary_text))
            db.commit()
    
    async def _maybe_update_summary(self, session_id: str) -> None:
        """按轮次定期刷新摘要，避免长对话无限增长。"""
        with SessionLocal() as db:
            count = db.execute(
                select(func.count()).select_from(ChatMessage).where(ChatMessage.session_id == session_id)
            ).scalar_one()
        
        if count > 0 and count % self.summary_refresh_turns == 0:
            g = get_chat_graph()
            checkpoint_context = await aload_chat_checkpoint_context(g, session_id) if g else {}
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
            logger.warning("Chat prune summary failed session=%s total=%s err=%s", session_id, total, exc)
        
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
