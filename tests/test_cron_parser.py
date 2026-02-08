#!/usr/bin/env python3
"""Tests for CronExpression parser - zero-dependency cron scheduling."""

import pytest
from datetime import datetime
from singularity.cron_parser import (
    CronExpression, CronParseError, parse_cron, next_cron_time,
    _parse_field, ALIASES,
)


class TestParseField:
    def test_wildcard(self):
        assert _parse_field("*", 0, 59) == set(range(0, 60))

    def test_single_value(self):
        assert _parse_field("5", 0, 59) == {5}

    def test_range(self):
        assert _parse_field("1-5", 0, 59) == {1, 2, 3, 4, 5}

    def test_list(self):
        assert _parse_field("1,3,5", 0, 59) == {1, 3, 5}

    def test_step(self):
        assert _parse_field("*/15", 0, 59) == {0, 15, 30, 45}

    def test_range_step(self):
        assert _parse_field("1-10/3", 0, 59) == {1, 4, 7, 10}

    def test_out_of_bounds(self):
        with pytest.raises(CronParseError):
            _parse_field("60", 0, 59)

    def test_invalid_step(self):
        with pytest.raises(CronParseError):
            _parse_field("*/abc", 0, 59)

    def test_named_months(self):
        from singularity.cron_parser import MONTH_NAMES
        assert _parse_field("jan", 1, 12, MONTH_NAMES) == {1}
        assert _parse_field("jan,mar,dec", 1, 12, MONTH_NAMES) == {1, 3, 12}

    def test_named_days(self):
        from singularity.cron_parser import DAY_NAMES
        assert _parse_field("mon", 0, 6, DAY_NAMES) == {0}
        assert _parse_field("mon-fri", 0, 6, DAY_NAMES) == {0, 1, 2, 3, 4}


class TestCronExpression:
    def test_every_minute(self):
        cron = CronExpression("* * * * *")
        assert cron.minutes == set(range(0, 60))
        assert cron.hours == set(range(0, 24))

    def test_every_5_minutes(self):
        cron = CronExpression("*/5 * * * *")
        assert cron.minutes == {0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55}

    def test_specific_time(self):
        cron = CronExpression("30 9 * * *")
        assert cron.minutes == {30}
        assert cron.hours == {9}

    def test_weekday_morning(self):
        cron = CronExpression("0 9 * * mon-fri")
        assert cron.minutes == {0}
        assert cron.hours == {9}
        assert cron.days_of_week == {0, 1, 2, 3, 4}

    def test_monthly(self):
        cron = CronExpression("0 0 1 * *")
        assert cron.days_of_month == {1}
        assert cron.hours == {0}
        assert cron.minutes == {0}

    def test_invalid_field_count(self):
        with pytest.raises(CronParseError):
            CronExpression("* * *")

    def test_invalid_value(self):
        with pytest.raises(CronParseError):
            CronExpression("abc * * * *")


class TestAliases:
    def test_daily(self):
        cron = CronExpression("@daily")
        assert cron.minutes == {0}
        assert cron.hours == {0}

    def test_hourly(self):
        cron = CronExpression("@hourly")
        assert cron.minutes == {0}
        assert cron.hours == set(range(0, 24))

    def test_weekly(self):
        cron = CronExpression("@weekly")
        assert cron.minutes == {0}
        assert cron.hours == {0}
        assert cron.days_of_week == {0}

    def test_monthly_alias(self):
        cron = CronExpression("@monthly")
        assert cron.days_of_month == {1}

    def test_yearly(self):
        cron = CronExpression("@yearly")
        assert cron.months == {1}
        assert cron.days_of_month == {1}


class TestMatches:
    def test_every_minute_matches(self):
        cron = CronExpression("* * * * *")
        assert cron.matches(datetime(2026, 1, 15, 10, 30))

    def test_specific_minute(self):
        cron = CronExpression("30 * * * *")
        assert cron.matches(datetime(2026, 1, 15, 10, 30))
        assert not cron.matches(datetime(2026, 1, 15, 10, 31))

    def test_specific_hour(self):
        cron = CronExpression("0 9 * * *")
        assert cron.matches(datetime(2026, 1, 15, 9, 0))
        assert not cron.matches(datetime(2026, 1, 15, 10, 0))

    def test_weekday_match(self):
        cron = CronExpression("0 9 * * mon-fri")
        # 2026-01-12 is Monday
        assert cron.matches(datetime(2026, 1, 12, 9, 0))
        # 2026-01-17 is Saturday
        assert not cron.matches(datetime(2026, 1, 17, 9, 0))


class TestNextRun:
    def test_next_minute(self):
        cron = CronExpression("* * * * *")
        after = datetime(2026, 1, 15, 10, 30, 0)
        nxt = cron.next_run(after=after)
        assert nxt == datetime(2026, 1, 15, 10, 31, 0)

    def test_next_hour(self):
        cron = CronExpression("0 * * * *")
        after = datetime(2026, 1, 15, 10, 30, 0)
        nxt = cron.next_run(after=after)
        assert nxt == datetime(2026, 1, 15, 11, 0, 0)

    def test_next_day(self):
        cron = CronExpression("0 9 * * *")
        after = datetime(2026, 1, 15, 10, 0, 0)
        nxt = cron.next_run(after=after)
        assert nxt == datetime(2026, 1, 16, 9, 0, 0)

    def test_next_specific_weekday(self):
        cron = CronExpression("0 9 * * 0")  # Monday
        # 2026-01-15 is Thursday
        after = datetime(2026, 1, 15, 10, 0, 0)
        nxt = cron.next_run(after=after)
        # Next Monday is Jan 19
        assert nxt == datetime(2026, 1, 19, 9, 0, 0)

    def test_next_n_runs(self):
        cron = CronExpression("0 * * * *")
        after = datetime(2026, 1, 15, 10, 0, 0)
        runs = cron.next_n_runs(3, after=after)
        assert len(runs) == 3
        assert runs[0] == datetime(2026, 1, 15, 11, 0, 0)
        assert runs[1] == datetime(2026, 1, 15, 12, 0, 0)
        assert runs[2] == datetime(2026, 1, 15, 13, 0, 0)


class TestDescribe:
    def test_daily_alias(self):
        cron = CronExpression("@daily")
        assert "midnight" in cron.describe().lower() or "day" in cron.describe().lower()

    def test_every_5_min(self):
        cron = CronExpression("*/5 * * * *")
        desc = cron.describe()
        assert "5 minutes" in desc

    def test_specific_time(self):
        cron = CronExpression("30 9 * * *")
        desc = cron.describe()
        assert "30" in desc and "9" in desc


class TestConvenienceFunctions:
    def test_parse_cron(self):
        cron = parse_cron("*/10 * * * *")
        assert isinstance(cron, CronExpression)

    def test_next_cron_time(self):
        after = datetime(2026, 1, 15, 10, 0, 0)
        nxt = next_cron_time("0 * * * *", after=after)
        assert nxt == datetime(2026, 1, 15, 11, 0, 0)
