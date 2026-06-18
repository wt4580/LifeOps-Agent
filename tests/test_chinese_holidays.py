from __future__ import annotations

from datetime import datetime

import pytest

from app.src.service.chinese_holidays import get_holidays_in_range


class TestGetHolidaysInRange:
    def test_spring_festival_found(self):
        start = datetime(2026, 1, 1)
        end = datetime(2026, 3, 1)
        holidays = get_holidays_in_range(start, end)
        names = [h["summary"] for h in holidays]
        assert any("春节" in n for n in names)

    def test_mid_autumn_found(self):
        start = datetime(2026, 9, 1)
        end = datetime(2026, 10, 1)
        holidays = get_holidays_in_range(start, end)
        names = [h["summary"] for h in holidays]
        assert any("中秋节" in n for n in names)

    def test_dragon_boat_found(self):
        start = datetime(2026, 6, 1)
        end = datetime(2026, 7, 1)
        holidays = get_holidays_in_range(start, end)
        names = [h["summary"] for h in holidays]
        assert any("端午节" in n for n in names)

    def test_national_day_found(self):
        start = datetime(2026, 10, 1)
        end = datetime(2026, 10, 7)
        holidays = get_holidays_in_range(start, end)
        names = [h["summary"] for h in holidays]
        assert any("国庆节" in n for n in names)

    def test_empty_range_returns_empty(self):
        start = datetime(2026, 6, 1)
        end = datetime(2026, 6, 1)
        holidays = get_holidays_in_range(start, end)
        assert holidays == []

    def test_holiday_structure(self):
        start = datetime(2026, 10, 1)
        end = datetime(2026, 10, 2)
        holidays = get_holidays_in_range(start, end)
        if holidays:
            h = holidays[0]
            assert "summary" in h
            assert "start" in h
            assert "end" in h
            assert h.get("all_day") is True
