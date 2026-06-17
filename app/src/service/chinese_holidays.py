"""动态计算中国传统节日，完全离线，无需手动录入。"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import cnlunar


# cnlunar 内置的法定农历节日映射
_LEGAL_LUNAR = {
    (1, 1): "春节",
    (5, 5): "端午节",
    (8, 15): "中秋节",
}

# 法定公历节日
_LEGAL_SOLAR = {
    (1, 1): "元旦",
    (5, 1): "劳动节",
    (10, 1): "国庆节",
}

# 其他常用农历节日（只取大众熟知的）
_COMMON_LUNAR = {
    (1, 15): "元宵节",
    (7, 7): "七夕",
    (9, 9): "重阳节",
    (12, 30): "除夕",
}


def get_holidays_in_range(start_dt: datetime, end_dt: datetime) -> list[dict[str, Any]]:
    """返回 [start_dt, end_dt) 区间内的所有中国传统节日事件。"""
    holidays: list[dict[str, Any]] = []
    current = start_dt
    while current < end_dt:
        try:
            a = cnlunar.Lunar(current)
        except Exception:
            current += timedelta(days=1)
            continue

        names: list[str] = []

        # 法定农历节日（春节/端午/中秋）
        if a.lunarMonth <= 12:
            key = (a.lunarMonth, a.lunarDay)
            if key in _LEGAL_LUNAR:
                names.append(_LEGAL_LUNAR[key])
            if key in _COMMON_LUNAR:
                names.append(_COMMON_LUNAR[key])

        # 法定公历节日（元旦/劳动节/国庆）
        solar_key = (current.month, current.day)
        if solar_key in _LEGAL_SOLAR:
            names.append(_LEGAL_SOLAR[solar_key])

        # 清明节由节气决定
        if a.todaySolarTerms == "清明":
            names.append("清明节")

        if names:
            name_str = "、".join(names)
            holidays.append({
                "summary": name_str,
                "start": current.isoformat(),
                "end": current.replace(hour=23, minute=59, second=59).isoformat(),
                "location": "全国",
                "status": "confirmed",
                "htmlLink": None,
                "all_day": True,
            })

        current += timedelta(days=1)

    return holidays
