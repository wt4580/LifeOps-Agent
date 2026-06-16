"""Google Calendar 集成服务。

依赖 google-api-python-client / google-auth-oauthlib / google-auth-httplib2。
这些依赖是可选的：未安装时本模块仍可导入，仅在实际调用 list_events 时返回友好错误。

首次使用需要交互式 OAuth2 授权（浏览器登录）。之后 token 自动刷新，无需再次授权。
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from ..common.config.base_config import settings
from ..common.config.log_config import logger


class CalendarServiceError(Exception):
    """日历服务层的可预期错误（缺依赖、缺凭据、授权失败、网络异常）。"""


class CalendarService:
    """Google Calendar 只读封装。

    设计取舍：
    - 仅暴露 list_events，不写事件（写入不在本次范围）。
    - 首次授权优先走本地服务器回调；非交互环境退化为 console flow。
    - 所有外部异常都包成 CalendarServiceError，避免上层崩溃。
    """

    _SCOPES = ["https://www.googleapis.com/auth/calendar.readonly"]

    def _load_google_libs(self):
        try:
            from google.oauth2.credentials import Credentials
            from google_auth_oauthlib.flow import InstalledAppFlow
            from google.auth.transport.requests import Request
            from googleapiclient.discovery import build
        except ImportError as exc:
            raise CalendarServiceError(
                "Google Calendar 依赖未安装。请执行: pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib"
            ) from exc
        return Credentials, InstalledAppFlow, Request, build

    def _get_credentials(self, *, interactive: bool):
        Credentials, InstalledAppFlow, Request, _ = self._load_google_libs()

        creds_path = (settings.google_calendar_credentials_json or "").strip()
        if not creds_path:
            raise CalendarServiceError(
                "未配置 GOOGLE_CALENDAR_CREDENTIALS_JSON。请到 Google Cloud Console 下载 OAuth 客户端 JSON，并把路径填进 .env。"
            )
        if not os.path.exists(creds_path):
            raise CalendarServiceError(f"Google Calendar 凭据文件不存在: {creds_path}")

        token_path = settings.google_calendar_token_path or ".cache/google_token.json"
        creds = None
        if os.path.exists(token_path):
            try:
                creds = Credentials.from_authorized_user_file(token_path, self._SCOPES)
            except Exception as exc:
                logger.warning("Google token 加载失败，将重新授权: %s", exc)
                creds = None

        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception as exc:
                logger.warning("Google token 刷新失败，将重新授权: %s", exc)
                creds = None

        if creds is None:
            if not interactive:
                raise CalendarServiceError(
                    "Google Calendar 尚未授权，且当前环境非交互，无法自动完成授权。"
                )
            flow = InstalledAppFlow.from_client_secrets_file(creds_path, self._SCOPES)
            try:
                creds = flow.run_local_server(port=0)
            except (EOFError, OSError):
                # 无浏览器 / 无终端回显的环境：退化为 console flow。
                auth_url, _ = flow.authorization_url(prompt="consent")
                logger.info("Google 授权链接: %s", auth_url)
                code = input(f"请打开链接完成授权，并把回调里的 code 粘贴到这里:\n{auth_url}\ncode=")
                flow.fetch_token(code=code.strip())
                creds = flow.credentials

        Path(token_path).parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(token_path, "w", encoding="utf-8") as fp:
                fp.write(creds.to_json())
        except OSError as exc:
            logger.warning("Google token 持久化失败: %s", exc)

        return creds

    def list_events(
        self,
        start_iso: str,
        end_iso: str,
        *,
        calendar_id: Optional[str] = None,
        max_results: int = 50,
        interactive: bool = True,
    ) -> list[dict[str, Any]]:
        """拉取 [start_iso, end_iso) 区间内的事件，按开始时间升序。"""

        _, _, _, build = self._load_google_libs()
        creds = self._get_credentials(interactive=interactive)

        if not start_iso or not end_iso:
            raise CalendarServiceError("list_events 必须传入 start_iso / end_iso")

        try:
            service = build("calendar", "v3", credentials=creds, cache_discovery=False)
            events_result = (
                service.events()
                .list(
                    calendarId=calendar_id or settings.google_calendar_id or "primary",
                    timeMin=start_iso,
                    timeMax=end_iso,
                    singleEvents=True,
                    orderBy="startTime",
                    maxResults=min(max_results, 250),
                )
                .execute()
            )
        except Exception as exc:
            raise CalendarServiceError(f"Google Calendar 请求失败: {exc}") from exc

        items = events_result.get("items", []) or []
        out: list[dict[str, Any]] = []
        for ev in items:
            start = ev.get("start") or {}
            end = ev.get("end") or {}
            out.append({
                "summary": ev.get("summary") or "(无标题)",
                "start": start.get("dateTime") or start.get("date"),
                "end": end.get("dateTime") or end.get("date"),
                "location": ev.get("location"),
                "status": ev.get("status"),
                "htmlLink": ev.get("htmlLink"),
                "all_day": "date" in start and "dateTime" not in start,
            })
        return out
