"""话题/任务焦点服务。

把对话分为两类：
- 任务段（task）：产生了"待办/目标/提案/待确认状态"的一连串对话，
  进入 topic_episodes + 快照 topic_episode_messages，关闭后可被用户重新提起时恢复全文。
- 闲聊段（chatter）：普通聊天/查询/叙事，不建轨迹、不快照、不摘要，
  只在短期焦点窗口中自然滑走，由既有 ConversationSummary 兜底。

核心职责：
- annotate_turn: 每轮对话后标注段类型，管理任务段的开启/延续/关闭与快照摘要。
- build_focus_context: 构建分层上下文（当前焦点消息 + 过往任务概览）。
- revive_topic: 用户主动提起过往任务时，按关键词找回并返回该任务段消息。
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import func, select, update as sa_update

from ..common.config.base_config import settings
from ..common.config.db_config import SessionLocal
from ..common.config.llm_config import chat_completion
from ..domain.entity.chat_entity import ChatMessage, TopicEpisode, TopicEpisodeMessage

logger = logging.getLogger(__name__)

# 当前焦点单次最多展示的任务段消息条数
EPISODE_FOCUS_LIMIT = 30

# 判定"产生了有状态动作"（task）的路由信号。纯查询/闲聊/随手记账不属于任务段。
TASK_ACTIONS: set[str] = {
    "propose_todo",
    "revise_proposal",
    "ask_user_confirm_proposal",
    "decompose_goal",
    "create_account",
    "ask_user_confirm_balance",
}
CLOSE_TOOLS: set[str] = {"plan_gate_confirm", "plan_gate_cancel"}
# 明确"换话题"的关键词（无需 LLM，直接判定开新任务段）
SWITCH_HINTS: tuple[str, ...] = (
    "换个话题",
    "换个事",
    "先不说",
    "不聊这个",
    "说点别的",
    "先放一放",
    "还有一件事",
    "另外还有",
)

_SUMMARY_PROMPT = (
    "你是生活助理的记忆系统。为下面这个已结束的任务段生成一份紧凑摘要，\n"
    "供日后用户主动提起时快速回顾。\n"
    "## 输出格式（Markdown，保持精简，每行一句话）\n"
    "- 主题：<一句话概括这个任务>\n"
    "- 要点/决定：<关键信息，用分号分隔；没有写'无'>\n"
    "- 待办/遗留：<已产生或未完成的事项；没有写'无'>"
)


# ----------------------------------------------------------------------
# 判定
# ----------------------------------------------------------------------

def _classify_turn(state: dict[str, Any]) -> str:
    """判定本轮是 task / chatter / close。

    只用"本轮活动产生的信号"，不用可能被 checkpointer 带过来的残留 proposal，
    避免上一轮的旧提案把后续闲聊误判成任务段。
    """
    if state.get("proposal_confirmed"):
        return "close"
    tool = state.get("used_tool") or ""
    if tool in CLOSE_TOOLS:
        return "close"
    if tool == "plan_proposal":
        return "task"
    action = (state.get("decision") or {}).get("action", "normal_chat")
    if action in TASK_ACTIONS:
        return "task"
    return "chatter"


def _make_topic_label(user_message: str) -> str:
    msg = (user_message or "").replace("\n", " ").replace("\r", " ").strip()
    msg = " ".join(msg.split())
    if not msg:
        msg = "未命名任务"
    return msg[:24]


def _should_open_new(active: dict[str, Any], user_message: str) -> bool:
    """有活跃任务段时，判断是否应结束它、开新任务段。

    走到这里说明本轮仍是 task 类：继续旧段还是换新段，取决于
    明确换话题关键词 或 消息间隔超过阈值。
    """
    if not active:
        return True
    msg = user_message or ""
    if any(h in msg for h in SWITCH_HINTS):
        return True
    gap_minutes = (datetime.utcnow() - active["last_active_at"]).total_seconds() / 60.0
    if gap_minutes > settings.topic_switch_gap_minutes:
        return True
    return False


# ----------------------------------------------------------------------
# 存取
# ----------------------------------------------------------------------

def _fmt(dt: datetime | None) -> str:
    if dt is None:
        return ""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone().strftime("%m-%d %H:%M")


def _load_active_episode(session_id: str) -> dict[str, Any] | None:
    with SessionLocal() as db:
        row = (
            db.execute(
                select(TopicEpisode)
                .where(TopicEpisode.session_id == session_id, TopicEpisode.status == "active")
                .order_by(TopicEpisode.last_active_at.desc())
                .limit(1)
            )
            .scalars()
            .first()
        )
        if not row:
            return None
        return {
            "id": row.id,
            "topic": row.topic,
            "last_active_at": row.last_active_at,
            "summary_text": row.summary_text,
        }


def _load_episode_messages(episode_id: int, limit: int = EPISODE_FOCUS_LIMIT) -> list[dict[str, Any]]:
    with SessionLocal() as db:
        rows = (
            db.execute(
                select(TopicEpisodeMessage)
                .where(TopicEpisodeMessage.episode_id == episode_id)
                .order_by(TopicEpisodeMessage.seq.asc())
                .limit(limit)
            )
            .scalars()
            .all()
        )
    return [{"role": r.role, "content": r.content, "ts": _fmt(r.created_at)} for r in rows]


def _open_episode(session_id: str, label: str) -> tuple[int, str]:
    with SessionLocal() as db:
        ep = TopicEpisode(session_id=session_id, kind="task", topic=label, status="active")
        db.add(ep)
        db.commit()
        return ep.id, label


def _close_episode(episode_id: int) -> None:
    msgs = _load_episode_messages(episode_id, limit=200)
    summary = _summarize_episode(msgs) if msgs else ""
    with SessionLocal() as db:
        row = db.execute(select(TopicEpisode).where(TopicEpisode.id == episode_id)).scalars().first()
        if not row:
            return
        row.status = "closed"
        row.closed_at = datetime.utcnow()
        if summary:
            row.summary_text = summary
        db.commit()
        logger.info("topic episode closed id=%s topic=%s msgs=%d", episode_id, row.topic, len(msgs))


def _set_topic(db, rows: list[Any], label: str) -> None:
    ids = [r.id for r in rows]
    if not ids:
        return
    db.execute(
        sa_update(ChatMessage).where(ChatMessage.id.in_(ids)).values(topic=label)
    )


def _snapshot_turn(db, episode_id: int, rows: list[Any]) -> None:
    seq_start = db.execute(
        select(func.coalesce(func.max(TopicEpisodeMessage.seq), 0)).where(
            TopicEpisodeMessage.episode_id == episode_id
        )
    ).scalar_one()
    seq = int(seq_start) + 1
    for r in rows:
        db.add(
            TopicEpisodeMessage(
                episode_id=episode_id,
                seq=seq,
                role=r.role,
                content=r.content,
                created_at=r.created_at,
            )
        )
        seq += 1
    db.execute(
        sa_update(TopicEpisode).where(TopicEpisode.id == episode_id).values(last_active_at=datetime.utcnow())
    )


def _summarize_episode(messages: list[dict[str, Any]]) -> str:
    dialogue = [{"role": m["role"], "content": m["content"]} for m in messages]
    try:
        return chat_completion(
            [
                {"role": "system", "content": _SUMMARY_PROMPT},
                {"role": "user", "content": json.dumps(dialogue, ensure_ascii=False)},
            ],
            temperature=0.2,
            runtime_context={"scenario": "episode_summary"},
        ).strip()
    except Exception as exc:
        logger.warning("episode summary failed: %s", exc)
        user_parts = [m["content"] for m in messages if m.get("role") == "user"]
        return f"主题：{(user_parts[0] if user_parts else '')[:40]}"


# ----------------------------------------------------------------------
# 对外主流程
# ----------------------------------------------------------------------

def annotate_turn(session_id: str, state: dict[str, Any], answer: str) -> None:
    """每轮对话后调用：标注段类型，管理任务段的开启/延续/关闭与快照。

    顺序：chat_service 保存完 user + assistant 两条消息之后、压缩删除之前调用。
    """
    try:
        kind = _classify_turn(state)
    except Exception as exc:
        logger.warning("topic classify failed: %s", exc)
        kind = "chatter"

    active = _load_active_episode(session_id)
    label = _make_topic_label(state.get("user_message") or "")

    with SessionLocal() as db:
        rows = list(
            db.execute(
                select(ChatMessage)
                .where(ChatMessage.session_id == session_id)
                .order_by(ChatMessage.created_at.desc())
                .limit(2)
            )
            .scalars()
            .all()
        )
        rows.reverse()

    # 闲聊段：不建轨迹。若有进行中的任务段，说明用户已切换到闲聊，结束旧任务段。
    if kind == "chatter":
        if active:
            _close_episode(active["id"])
        return

    # 确认/取消一轮：结束进行中的任务段，把它标成完整归档。
    if kind == "close":
        if active:
            with SessionLocal() as db:
                _set_topic(db, rows, active["topic"])
                db.commit()
            _close_episode(active["id"])
        return

    # task 轮：开新段 或 延续当前段
    if active and _should_open_new(active, state.get("user_message") or ""):
        _close_episode(active["id"])
        active = None
    if not active:
        episode_id, label = _open_episode(session_id, label)
    else:
        episode_id, label = active["id"], active["topic"]

    with SessionLocal() as db:
        _set_topic(db, rows, label)
        _snapshot_turn(db, episode_id, rows)
        db.commit()


def build_focus_context(session_id: str) -> tuple[list[dict[str, str]], list[str]]:
    """构建分层上下文。

    返回 (focus_messages, overview_lines)：
    - focus_messages：当前焦点（进行中任务段的消息；无任务段时为最近几轮闲聊）。
    - overview_lines：过往（已关闭）任务段一行概览。
    """
    active = _load_active_episode(session_id)
    if active:
        msgs = _load_episode_messages(active["id"], limit=EPISODE_FOCUS_LIMIT)
        if msgs:
            return msgs, _episode_overview(session_id, exclude_id=active["id"])

    with SessionLocal() as db:
        rows = list(
            db.execute(
                select(ChatMessage)
                .where(ChatMessage.session_id == session_id, ChatMessage.topic.is_(None))
                .order_by(ChatMessage.created_at.desc())
                .limit(settings.chatter_focus_turns * 2)
            )
            .scalars()
            .all()
        )
        rows.reverse()
    msgs = [{"role": r.role, "content": r.content, "ts": _fmt(r.created_at)} for r in rows]
    return msgs, _episode_overview(session_id, exclude_id=(active["id"] if active else None))


def _episode_overview(session_id: str, exclude_id: int | None = None) -> list[str]:
    with SessionLocal() as db:
        rows = (
            db.execute(
                select(TopicEpisode)
                .where(TopicEpisode.session_id == session_id, TopicEpisode.status == "closed")
                .order_by(TopicEpisode.closed_at.desc())
                .limit(8)
            )
            .scalars()
            .all()
        )
    lines: list[str] = []
    for r in rows:
        if exclude_id is not None and r.id == exclude_id:
            continue
        when = _fmt(r.closed_at or r.last_active_at)
        s = (r.summary_text or "").strip()
        brief = ""
        for line in s.splitlines():
            line = line.strip()
            if line.startswith("- 主题"):
                brief = line.split("：", 1)[1] if "：" in line else line
                break
            if line.startswith("主题"):
                brief = line.split("：", 1)[1] if "：" in line else line
                break
        if not brief and s:
            brief = s[:40]
        lines.append(f"- {r.topic}（{when}）：{brief[:50]}" if brief else f"- {r.topic}（{when}）")
    return lines


def load_task_index(session_id: str) -> list[dict[str, Any]]:
    """供 decompose 使用：所有（活跃+已关闭）任务段的一行索引，用于决定是否 revive。"""
    with SessionLocal() as db:
        rows = (
            db.execute(
                select(TopicEpisode)
                .where(TopicEpisode.session_id == session_id)
                .order_by(TopicEpisode.status, TopicEpisode.last_active_at.desc())
                .limit(6)
            )
            .scalars()
            .all()
        )
    out = []
    for r in rows:
        out.append(
            {
                "id": r.id,
                "topic": r.topic,
                "status": r.status,
                "time": _fmt(r.last_active_at),
                "brief": (r.summary_text or "")[:60],
            }
        )
    return out


def revive_topic(session_id: str, query: str) -> dict[str, Any] | None:
    """用户主动提起过往任务时，按 id / 关键词找回任务段，返回其消息与摘要。"""
    q = str(query or "").strip()
    if not q:
        return None

    if q.isdigit():
        with SessionLocal() as db:
            r = db.execute(
                select(TopicEpisode).where(TopicEpisode.id == int(q), TopicEpisode.session_id == session_id)
            ).scalars().first()
        if r:
            return _revive_payload(r)

    with SessionLocal() as db:
        rows = (
            db.execute(
                select(TopicEpisode)
                .where(TopicEpisode.session_id == session_id, TopicEpisode.status == "closed")
                .order_by(TopicEpisode.closed_at.desc())
                .limit(20)
            )
            .scalars()
            .all()
        )

    best = None
    best_score = 0
    tokens = [t for t in re.split(r"[\s，,、。.!！?？;；:：\-\d]+", q) if t]
    for r in rows:
        hay = f"{r.topic} {r.summary_text or ''}"
        score = sum(1 for tok in tokens if tok and tok in hay)
        if score > best_score:
            best, best_score = r, score

    if best is not None and best_score > 0:
        return _revive_payload(best)

    for r in rows:
        if q in r.topic:
            return _revive_payload(r)
    return None


def _revive_payload(ep: TopicEpisode) -> dict[str, Any]:
    msgs = _load_episode_messages(ep.id, limit=EPISODE_FOCUS_LIMIT)
    return {
        "episode_id": ep.id,
        "topic": ep.topic,
        "summary": ep.summary_text or "",
        "messages": msgs,
    }