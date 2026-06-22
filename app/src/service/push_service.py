"""主动推送服务。

通过 Server酱 (https://sct.ftqq.com) 向用户微信发送通知。
也支持 Pushover 作为备选后端。
"""

import json
import logging
from datetime import datetime

import requests

from ..common.config.base_config import settings

logger = logging.getLogger("LifeOps-Agent")


def push_notification(title: str, content: str, backend: str = "auto") -> bool:
    """推送一条通知到用户手机。

    Args:
        title: 标题（微信通知栏显示）
        content: 正文（纯文本或 Markdown）
        backend: auto | serverchan | pushover

    Returns:
        是否推送成功
    """
    if backend == "auto":
        if settings.serverchan_key:
            return _push_serverchan(title, content)
        if settings.pushover_user_key and settings.pushover_api_token:
            return _push_pushover(title, content)
        logger.warning("未配置任何推送渠道 (SERVERCHAN_KEY / PUSHOVER_*)")
        return False

    if backend == "serverchan":
        return _push_serverchan(title, content)
    if backend == "pushover":
        return _push_pushover(title, content)
    return False


def _push_serverchan(title: str, content: str) -> bool:
    key = (settings.serverchan_key or "").strip()
    if not key:
        logger.warning("SERVERCHAN_KEY 未配置")
        return False

    try:
        resp = requests.post(
            f"https://sctapi.ftqq.com/{key}.send",
            json={"title": title, "content": content, "channel": 9},
            timeout=10,
        )
        data = resp.json()
        if data.get("code") == 0:
            logger.info("Server酱推送成功: %s", title[:30])
            return True
        logger.warning("Server酱推送失败: %s", data.get("message", resp.text))
        return False
    except requests.RequestException as exc:
        logger.warning("Server酱请求异常: %s", exc)
        return False


def _push_pushover(title: str, content: str) -> bool:
    user = (settings.pushover_user_key or "").strip()
    token = (settings.pushover_api_token or "").strip()
    if not user or not token:
        logger.warning("PUSHOVER_USER_KEY / PUSHOVER_API_TOKEN 未配置")
        return False

    try:
        resp = requests.post(
            "https://api.pushover.net/1/messages.json",
            data={"user": user, "token": token, "title": title[:100], "message": content[:1024]},
            timeout=10,
        )
        if resp.status_code == 200:
            logger.info("Pushover推送成功: %s", title[:30])
            return True
        logger.warning("Pushover推送失败: %s", resp.text)
        return False
    except requests.RequestException as exc:
        logger.warning("Pushover请求异常: %s", exc)
        return False


def push_reminder_due(content: str, remind_count: int) -> bool:
    """提醒到期时推送通知。"""
    title = "LifeOps 提醒"
    body = f"{content}\n(第{remind_count + 1}次提醒)"
    return push_notification(title, body)


def push_proactive_advice(advice: str, reasons: list[str]) -> bool:
    """主动建议通知（如即将到来的日程冲突等）。"""
    title = "LifeOps 小建议"
    body = advice
    if reasons:
        body += "\n\n原因：" + "；".join(reasons[:3])
    return push_notification(title, body)


def push_daily_review(summary: str) -> bool:
    """晚间/日终复盘推送。"""
    today = datetime.now().strftime("%m月%d日")
    title = f"LifeOps {today} 复盘"
    return push_notification(title, summary)


def push_abnormal_detection(issue: str) -> bool:
    """异常检测推送（如连续未记录、预算超支等）。"""
    title = "LifeOps 提醒"
    return push_notification(title, f"⚠️ {issue}")
