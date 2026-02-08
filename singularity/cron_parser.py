#!/usr/bin/env python3
"""
Cron Expression Parser - Zero-dependency cron scheduling for Singularity.

Parses standard 5-field cron expressions and computes next run times.
Supports: minute, hour, day-of-month, month, day-of-week.

Format: "minute hour day_of_month month day_of_week"

Field values:
  - minute:       0-59
  - hour:         0-23
  - day_of_month: 1-31
  - month:        1-12
  - day_of_week:  0-6 (0=Monday, 6=Sunday)  [Python weekday convention]

Supports:
  - Wildcards: *
  - Ranges: 1-5
  - Lists: 1,3,5
  - Steps: */5, 1-10/2
  - Named months: jan-dec
  - Named days: mon-sun
  - Common aliases: @hourly, @daily, @weekly, @monthly, @yearly

Examples:
  "*/5 * * * *"     -> every 5 minutes
  "0 9 * * mon-fri" -> 9 AM every weekday
  "0 0 1 * *"       -> midnight on 1st of every month
  "@daily"          -> midnight every day
"""

from datetime import datetime, timedelta
from typing import List, Optional, Set, Tuple


# Named month/day mappings
MONTH_NAMES = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

# Python weekday: 0=Monday ... 6=Sunday
DAY_NAMES = {
    "mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6,
}

ALIASES = {
    "@yearly":   "0 0 1 1 *",
    "@annually": "0 0 1 1 *",
    "@monthly":  "0 0 1 * *",
    "@weekly":   "0 0 * * 0",
    "@daily":    "0 0 * * *",
    "@midnight": "0 0 * * *",
    "@hourly":   "0 * * * *",
}

# Days in each month (non-leap)
DAYS_IN_MONTH = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


def _is_leap_year(year: int) -> bool:
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)


def _days_in_month(month: int, year: int) -> int:
    if month == 2 and _is_leap_year(year):
        return 29
    return DAYS_IN_MONTH[month]


class CronParseError(Exception):
    """Raised when a cron expression is invalid."""
    pass


def _parse_field(field: str, min_val: int, max_val: int, names: dict = None) -> Set[int]:
    """Parse a single cron field into a set of valid integer values."""
    result = set()

    for part in field.split(","):
        part = part.strip().lower()

        # Resolve named values
        if names:
            for name, val in names.items():
                part = part.replace(name, str(val))

        # Handle step: */N or M-N/S
        step = 1
        if "/" in part:
            range_part, step_str = part.split("/", 1)
            try:
                step = int(step_str)
            except ValueError:
                raise CronParseError(f"Invalid step value: {step_str}")
            if step < 1:
                raise CronParseError(f"Step must be >= 1, got {step}")
            part = range_part

        # Handle wildcard
        if part == "*":
            result.update(range(min_val, max_val + 1, step))
            continue

        # Handle range: M-N
        if "-" in part:
            try:
                low_str, high_str = part.split("-", 1)
                low = int(low_str)
                high = int(high_str)
            except ValueError:
                raise CronParseError(f"Invalid range: {part}")
            if low < min_val or high > max_val:
                raise CronParseError(
                    f"Range {low}-{high} out of bounds [{min_val}-{max_val}]"
                )
            if low > high:
                # Wrap-around (e.g., day-of-week 5-1 means Sat,Sun,Mon,Tue)
                result.update(range(low, max_val + 1, step))
                result.update(range(min_val, high + 1, step))
            else:
                result.update(range(low, high + 1, step))
            continue

        # Single value
        try:
            val = int(part)
        except ValueError:
            raise CronParseError(f"Invalid value: {part}")
        if val < min_val or val > max_val:
            raise CronParseError(
                f"Value {val} out of bounds [{min_val}-{max_val}]"
            )
        result.add(val)

    if not result:
        raise CronParseError(f"Field produced no valid values: {field}")

    return result


class CronExpression:
    """Parsed cron expression with next-run-time computation."""

    def __init__(self, expression: str):
        self.original = expression.strip()
        expr = ALIASES.get(self.original.lower(), self.original)

        fields = expr.split()
        if len(fields) != 5:
            raise CronParseError(
                f"Expected 5 fields (minute hour dom month dow), got {len(fields)}: '{expr}'"
            )

        self.minutes = _parse_field(fields[0], 0, 59)
        self.hours = _parse_field(fields[1], 0, 23)
        self.days_of_month = _parse_field(fields[2], 1, 31)
        self.months = _parse_field(fields[3], 1, 12, MONTH_NAMES)
        self.days_of_week = _parse_field(fields[4], 0, 6, DAY_NAMES)

        # Track if day-of-month or day-of-week is restricted (not wildcard)
        self._dom_restricted = fields[2] != "*"
        self._dow_restricted = fields[4] != "*"

    def matches(self, dt: datetime) -> bool:
        """Check if a datetime matches this cron expression."""
        if dt.minute not in self.minutes:
            return False
        if dt.hour not in self.hours:
            return False
        if dt.month not in self.months:
            return False

        # Day matching: if both DOM and DOW are restricted, match either (OR).
        # If only one is restricted, match that one. If neither, match any day.
        dom_match = dt.day in self.days_of_month
        dow_match = dt.weekday() in self.days_of_week

        if self._dom_restricted and self._dow_restricted:
            # Standard cron behavior: OR when both specified
            if not (dom_match or dow_match):
                return False
        elif self._dom_restricted:
            if not dom_match:
                return False
        elif self._dow_restricted:
            if not dow_match:
                return False
        # else: both are wildcards, any day matches

        return True

    def next_run(self, after: Optional[datetime] = None, max_years: int = 4) -> Optional[datetime]:
        """
        Compute the next datetime that matches this cron expression.

        Args:
            after: Start searching after this time (default: now).
            max_years: Maximum years to search ahead before giving up.

        Returns:
            Next matching datetime (second=0), or None if not found within max_years.
        """
        if after is None:
            after = datetime.now()

        # Start from the next minute
        candidate = after.replace(second=0, microsecond=0) + timedelta(minutes=1)
        end = after + timedelta(days=365 * max_years)

        while candidate <= end:
            # Fast-skip: wrong month
            if candidate.month not in self.months:
                # Jump to first day of next valid month
                candidate = self._advance_to_next_month(candidate)
                continue

            # Fast-skip: wrong day
            if not self._day_matches(candidate):
                candidate = candidate.replace(hour=0, minute=0) + timedelta(days=1)
                continue

            # Fast-skip: wrong hour
            if candidate.hour not in self.hours:
                next_hour = self._next_in_set(candidate.hour, self.hours)
                if next_hour is not None and next_hour > candidate.hour:
                    candidate = candidate.replace(hour=next_hour, minute=0)
                else:
                    # Next valid hour is tomorrow
                    candidate = candidate.replace(hour=0, minute=0) + timedelta(days=1)
                continue

            # Fast-skip: wrong minute
            if candidate.minute not in self.minutes:
                next_min = self._next_in_set(candidate.minute, self.minutes)
                if next_min is not None and next_min > candidate.minute:
                    candidate = candidate.replace(minute=next_min)
                else:
                    # Next valid minute is next hour
                    candidate = candidate.replace(minute=0) + timedelta(hours=1)
                continue

            # All fields match
            return candidate

        return None  # No match within max_years

    def next_n_runs(self, n: int, after: Optional[datetime] = None) -> List[datetime]:
        """Get the next N run times."""
        runs = []
        current = after
        for _ in range(n):
            nxt = self.next_run(after=current)
            if nxt is None:
                break
            runs.append(nxt)
            current = nxt
        return runs

    def _day_matches(self, dt: datetime) -> bool:
        """Check if the day (DOM/DOW) matches."""
        dom_match = dt.day in self.days_of_month
        dow_match = dt.weekday() in self.days_of_week

        if self._dom_restricted and self._dow_restricted:
            return dom_match or dow_match
        elif self._dom_restricted:
            return dom_match
        elif self._dow_restricted:
            return dow_match
        return True

    def _advance_to_next_month(self, dt: datetime) -> datetime:
        """Jump to the 1st of the next valid month."""
        month = dt.month
        year = dt.year
        for _ in range(48):  # max 4 years of months
            month += 1
            if month > 12:
                month = 1
                year += 1
            if month in self.months:
                return datetime(year, month, 1, 0, 0)
        return dt + timedelta(days=365 * 5)  # give up

    @staticmethod
    def _next_in_set(current: int, valid_set: Set[int]) -> Optional[int]:
        """Find the next value in valid_set that is > current."""
        for v in sorted(valid_set):
            if v > current:
                return v
        return None

    def describe(self) -> str:
        """Return a human-readable description of this cron schedule."""
        # Check common patterns for readable descriptions
        if self.original.lower() in ALIASES:
            alias = self.original.lower()
            descriptions = {
                "@yearly": "Once a year (Jan 1 midnight)",
                "@annually": "Once a year (Jan 1 midnight)",
                "@monthly": "Once a month (1st, midnight)",
                "@weekly": "Once a week (Monday midnight)",
                "@daily": "Once a day (midnight)",
                "@midnight": "Once a day (midnight)",
                "@hourly": "Once an hour (at :00)",
            }
            return descriptions.get(alias, self.original)

        parts = []

        # Minutes
        if self.minutes == set(range(0, 60)):
            parts.append("every minute")
        elif len(self.minutes) == 1:
            m = next(iter(self.minutes))
            parts.append(f"at minute {m}")
        else:
            mins = sorted(self.minutes)
            # Check for step pattern
            if len(mins) > 2:
                step = mins[1] - mins[0]
                if all(mins[i+1] - mins[i] == step for i in range(len(mins)-1)):
                    parts.append(f"every {step} minutes")
                else:
                    parts.append(f"at minutes {','.join(str(m) for m in mins)}")
            else:
                parts.append(f"at minutes {','.join(str(m) for m in mins)}")

        # Hours
        if self.hours == set(range(0, 24)):
            parts.append("every hour")
        elif len(self.hours) == 1:
            h = next(iter(self.hours))
            parts.append(f"at hour {h}")
        else:
            parts.append(f"at hours {','.join(str(h) for h in sorted(self.hours))}")

        # Days of month
        if self._dom_restricted:
            if len(self.days_of_month) == 1:
                d = next(iter(self.days_of_month))
                parts.append(f"on day {d}")
            else:
                parts.append(f"on days {','.join(str(d) for d in sorted(self.days_of_month))}")

        # Months
        if self.months != set(range(1, 13)):
            month_strs = {v: k.capitalize() for k, v in MONTH_NAMES.items()}
            parts.append(f"in {','.join(month_strs.get(m, str(m)) for m in sorted(self.months))}")

        # Days of week
        if self._dow_restricted:
            day_strs = {v: k.capitalize() for k, v in DAY_NAMES.items()}
            parts.append(f"on {','.join(day_strs.get(d, str(d)) for d in sorted(self.days_of_week))}")

        return "; ".join(parts) if parts else self.original

    def __repr__(self) -> str:
        return f"CronExpression('{self.original}')"


def parse_cron(expression: str) -> CronExpression:
    """Parse a cron expression string into a CronExpression object."""
    return CronExpression(expression)


def next_cron_time(expression: str, after: Optional[datetime] = None) -> Optional[datetime]:
    """Convenience: parse expression and get next run time."""
    return CronExpression(expression).next_run(after=after)
