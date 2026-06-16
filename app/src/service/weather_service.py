"""高德天气服务。

依赖 requests。
使用 v3 地理编码（取城市 adcode）+ v3 天气查询（实况 + 4 天预报）。
内存缓存默认 5 分钟，节省免费额度。

Key 未配置或服务异常时抛 WeatherServiceError，由 graph 层捕获后生成友好回复。
"""

from __future__ import annotations

import re
import time
from typing import Any, Optional

import requests

from ..common.config.base_config import settings
from ..common.config.log_config import logger


class WeatherServiceError(Exception):
    """天气服务的可预期错误（缺 Key、网络失败、限流、无效城市）。"""


class WeatherService:
    """高德天气只读封装。

    - 城市识别优先用用户输入（对话里指定的城市）
    - 其次是 profile 的 default_city（从 profile_context 字符串识别 "所在城市:XX"）
    - 再次是 .env 的 DEFAULT_WEATHER_CITY
    - 都没有时由上层 LLM 反问用户
    """

    _GEO_URL = "https://restapi.amap.com/v3/config/district"
    _WEATHER_URL = "https://restapi.amap.com/v3/weather/weatherInfo"
    _TIMEOUT = 6

    def __init__(self):
        self._cache: dict[str, tuple[float, dict]] = {}

    def _require_key(self):
        key = (settings.amap_weather_key or "").strip()
        if not key:
            raise WeatherServiceError(
                "未配置 AMAP_WEATHER_KEY。请到 https://console.amap.com/dev/key/app 申请 Web 服务 Key，并把值填进 .env。"
            )
        return key

    def _resolve_default_city(self, profile_context: Optional[str]) -> str:
        if profile_context:
            m = re.search(r"所在城市[:：]\s*([^\s；;]+)", profile_context)
            if m:
                return m.group(1).strip()
        return (settings.default_weather_city or "").strip()

    def _get_adcode(self, city_name: str) -> tuple[str, str]:
        """返回 (adcode, 规范化城市名)。失败抛 WeatherServiceError。"""
        key = self._require_key()
        try:
            resp = requests.get(
                self._GEO_URL,
                params={"key": key, "keywords": city_name, "subdistrict": 0},
                timeout=self._TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            raise WeatherServiceError(f"高德地理编码请求失败: {exc}") from exc

        if data.get("status") != "1" or not data.get("districts"):
            raise WeatherServiceError(f"无法识别城市: {city_name}")

        district = data["districts"][0]
        return str(district.get("adcode") or ""), str(district.get("name") or city_name)

    def _fetch_weather(self, adcode: str) -> dict:
        key = self._require_key()
        try:
            resp = requests.get(
                self._WEATHER_URL,
                params={"key": key, "city": adcode, "extensions": "all", "output": "JSON"},
                timeout=self._TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            raise WeatherServiceError(f"高德天气请求失败: {exc}") from exc

        if data.get("status") != "1":
            info = data.get("info") or data.get("infocode") or "未知错误"
            raise WeatherServiceError(f"高德天气返回错误: {info}")
        return data

    def _parse(self, data: dict, city_name: str) -> dict[str, Any]:
        lives = data.get("lives") or []
        forecasts = data.get("forecasts") or []

        result: dict[str, Any] = {"city": city_name, "live": None, "forecast": []}
        if lives:
            lv = lives[0]
            result["live"] = {
                "weather": lv.get("weather"),
                "temperature": lv.get("temperature"),
                "winddirection": lv.get("winddirection"),
                "windpower": lv.get("windpower"),
                "humidity": lv.get("humidity"),
                "reporttime": lv.get("reporttime"),
                "province": lv.get("province"),
            }
        if forecasts:
            for day in forecasts[0].get("casts") or []:
                result["forecast"].append({
                    "date": day.get("date"),
                    "week": day.get("week"),
                    "dayweather": day.get("dayweather"),
                    "nightweather": day.get("nightweather"),
                    "daytemp": day.get("daytemp"),
                    "nighttemp": day.get("nighttemp"),
                    "daywind": day.get("daywind"),
                    "nightwind": day.get("nightwind"),
                })
        return result

    def query(self, city_name: str, profile_context: Optional[str] = None) -> dict[str, Any]:
        city = (city_name or "").strip()
        if not city:
            city = self._resolve_default_city(profile_context)
        if not city:
            raise WeatherServiceError(
                "未指定城市，且没有默认城市。请告诉我你想查哪个城市的天气，或在 .env 里填写 DEFAULT_WEATHER_CITY。"
            )

        cache_key = city
        ttl = max(0, int(settings.weather_cache_ttl_seconds or 0))
        now = time.time()
        if ttl > 0 and cache_key in self._cache:
            ts, cached = self._cache[cache_key]
            if now - ts < ttl:
                cached["cached"] = True
                return cached

        adcode, real_city = self._get_adcode(city)
        data = self._fetch_weather(adcode)
        result = self._parse(data, real_city)
        result["adcode"] = adcode
        result["cached"] = False

        if ttl > 0:
            self._cache[cache_key] = (now, result)

        return result


weather_service = WeatherService()
