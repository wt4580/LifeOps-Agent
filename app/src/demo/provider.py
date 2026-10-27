"""A rule-based model simulator for wiring demos, never a model-quality benchmark."""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from types import SimpleNamespace as NS


def _json(value):
    return json.dumps(value, ensure_ascii=False)


def _payload(messages):
    text = str(messages[-1].get("content", ""))
    try:
        return json.loads(text), text
    except (ValueError, TypeError):
        return {}, text


def _date(text: str, now: datetime) -> str | None:
    match = re.search(r"\d{4}-\d{2}-\d{2}", text)
    if match:
        return match.group()
    for word, delta in (("大后天", 3), ("后天", 2), ("明天", 1), ("今天", 0), ("昨天", -1)):
        if word in text:
            return (now + timedelta(days=delta)).date().isoformat()
    return None


def _item(text: str, now: datetime) -> dict:
    date = _date(text, now)
    match = re.search(r"([0-2]?\d)(?:[:：](\d{2})|[点时](?:(\d{1,2})分?)?)", text)
    hour = int(match.group(1)) if match else None
    minute = int(match.group(2) or match.group(3) or 0) if match else 0
    if hour is not None and any(word in text for word in ("下午", "晚上")) and hour < 12:
        hour += 12
    due = f"{date}T{hour:02d}:{minute:02d}:00" if date and hour is not None else date
    title = re.sub(r"\d{4}-\d{2}-\d{2}|大后天|后天|明天|今天|昨天|上午|下午|晚上|早上", "", text)
    title = re.sub(r"([0-2]?\d)(?:[:：]\d{2}|[点时](?:\d{1,2}分?)?)", "", title)
    title = re.sub(r"帮我|请|提醒我|添加待办|创建待办|安排待办|安排一下", "", title).strip(" ：:，,。去")
    return {"title": title or "演示待办", "due_at": due}


def decision(text: str, now: datetime) -> dict:
    action, args = "normal_chat", {}
    if "演示安全测试" in text:
        action, args = "query_database", {"sql": "SELECT * FROM todo_items WHERE session_id='another-user'"}
    elif "同时" in text and "天气" in text and "待办" in text:
        steps = [
            {"action": "query_weather", "args": {"city": "杭州"}, "purpose": "演示天气"},
            {"action": "query_todos", "args": {}, "purpose": "待办查询"},
        ]
        return {
            "action": "query_weather",
            "args": steps[0]["args"],
            "needs_plan": True,
            "steps": steps,
            "mode": "normal",
        }
    elif "创建账户" in text:
        name = re.split(r"余额|，|,", text.split("创建账户", 1)[1])[0].strip(' ：:「」"')
        balance = re.search(r"余额\s*([-\d.]+)", text)
        action, args = "create_account", {"name": name, "balance": float(balance.group(1)) if balance else 0}
    elif "记账" in text or re.search(r"(?:花了|收入)\s*\d", text):
        amount = re.search(r"(?:花了|收入|记账)\s*(\d+(?:\.\d+)?)", text)
        action, args = (
            "record_expense",
            {
                "amount": float(amount.group(1)) if amount else 25,
                "transaction_type": "income" if "收入" in text else "expense",
                "category": "餐饮",
                "description": text,
            },
        )
    elif "查询账户" in text or "账户余额" in text:
        action = "query_accounts"
    elif any(word in text for word in ("报销", "会议准备", "学习节奏", "知识库")):
        action, args = "query_knowledge", {"question": text}
    elif "天气" in text:
        action, args = "query_weather", {"city": "杭州"}
    elif any(word in text for word in ("查询待办", "查看待办", "有哪些待办")):
        action = "query_todos"
    elif any(word in text for word in ("待办", "提醒", "开会", "交报告", "体检")):
        action, args = "propose_todo", {"text": text}
        parts = [part for part in re.split(r"[，,；;]|然后", text) if part.strip()]
        if len(parts) == 1:
            item = _item(text, now)
            due = item["due_at"] or ""
            args.update(
                title=item["title"],
                due_date=due.split("T")[0] or None,
                due_time=due.split("T")[1] if "T" in due else None,
            )
    return {"action": action, "args": args, "mode": "quick", "needs_plan": False, "assistant_message": ""}


def response_for(messages, context):
    payload, text = _payload(messages)
    scenario = context.get("scenario", "unknown")
    now = datetime.fromisoformat(context.get("now_local") or datetime.now().astimezone().isoformat())
    if "[演示故障]" in text:
        raise RuntimeError("simulated provider failure")
    if scenario == "decompose":
        return _json(decision(payload.get("user_message", ""), now))
    if scenario == "normalize_date_hint":
        source = payload.get("text", "")
        multiple = len(re.findall(r"明天|后天|\d{4}-\d{2}-\d{2}", source)) > 1
        date = None if multiple else _date(source, now)
        return _json({"date": date, "confidence": 1 if date else 0})
    if scenario == "propose_plan":
        if "用户修改：" in text:
            correction = text.split("用户修改：", 1)[1]
            try:
                original, _ = json.JSONDecoder().raw_decode(text[text.index("{") :])
                items = original["items"]
                for item in items:
                    changed = _item(correction, now)
                    if changed["due_at"]:
                        old = item.get("due_at") or ""
                        new = changed["due_at"]
                        if "T" not in new and "T" in old:
                            new += "T" + old.split("T", 1)[1]
                        item["due_at"] = new
                return _json({"items": items})
            except (ValueError, KeyError):
                pass
        parts = [part for part in re.split(r"[，,；;]|然后", text) if part.strip()]
        return _json({"items": [_item(part, now) for part in parts]})
    if scenario in {"verify_step", "quality_gate"}:
        return _json({"passed": True})
    if scenario == "rewrite_query":
        return payload.get("question") or payload.get("query") or text
    if scenario == "rag_answer":
        evidence = text.split("Context:\n", 1)[-1].split("\n\nQuestion:", 1)[0]
        lines = [line.lstrip("> ") for line in evidence.splitlines() if line.strip() and not line.startswith("[1]")]
        return "[演示回答] 根据虚构资料：" + " ".join(lines)[:420] + " [1]"
    if scenario == "query_todos_summarize":
        return "[演示查询] " + "；".join(
            f"{row.get('title')}（{row.get('due_at') or '无截止日期'}）" for row in payload
        )
    if scenario == "query_weather_summarize":
        return "[虚构天气] 杭州，晴，20℃。这是离线演示数据。"
    if scenario == "calendar_summarize":
        return "[演示日历] " + "；".join(row.get("summary", "") for row in payload)
    if scenario == "finalize_merge":
        return "[演示汇总]\n" + text[:2400]
    if scenario == "post_chat_learn":
        return _json({"has_update": False})
    if "proactive" in scenario:
        return _json({"score": 0, "should_add": False, "advice": None, "reasons": ["demo mode"]})
    if any(word in scenario for word in ("extract", "profile", "reflection")):
        return _json({"items": []})
    if "summary" in scenario:
        return "虚构演示资料摘要；不代表真实用户事件。"
    return "[演示模式] 可尝试：明天下午3点开会、查询待办、报销需要哪些材料、创建账户 演示账户 余额200。"


class DemoStream:
    def __init__(self, text):
        self.text = text
        self.closed = False

    def __iter__(self):
        for offset in range(0, len(self.text), 24):
            yield NS(choices=[NS(delta=NS(content=self.text[offset : offset + 24]), finish_reason=None)], usage=None)

    def close(self):
        self.closed = True


class DemoClient:
    def __init__(self, context):
        self.context = context
        self.chat = NS(completions=self)

    def with_options(self, **kwargs):
        return self

    def create(self, *, messages, stream=False, **kwargs):
        text = response_for(messages, self.context)
        if stream:
            return DemoStream(text)
        return NS(choices=[NS(message=NS(content=text), finish_reason="stop")], usage=None)
