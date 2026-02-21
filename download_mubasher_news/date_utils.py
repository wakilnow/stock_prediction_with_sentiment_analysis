"""
date_utils.py
~~~~~~~~~~~~~
Utilities for parsing and normalising date strings scraped from Mubasher pages.

Public API
----------
    normalize_date(raw: str) -> str
        Accept any supported date string and return it as YYYY-MM-DD.
        Falls back to the original string if no format matches.
"""

import re
from datetime import datetime

# ---------------------------------------------------------------------------
# Latin / ISO format table
# ---------------------------------------------------------------------------

_DATE_FORMATS = [
    "%Y-%m-%dT%H:%M:%S%z",   # ISO 8601 with timezone  e.g. 2024-03-15T10:30:00+02:00
    "%Y-%m-%dT%H:%M:%SZ",    # ISO 8601 UTC             e.g. 2024-03-15T10:30:00Z
    "%Y-%m-%dT%H:%M:%S",     # ISO 8601 no tz           e.g. 2024-03-15T10:30:00
    "%Y-%m-%d",               # plain ISO date           e.g. 2024-03-15
    "%d/%m/%Y",               # DD/MM/YYYY               e.g. 15/03/2024
    "%m/%d/%Y",               # MM/DD/YYYY               e.g. 03/15/2024
    "%d-%m-%Y",               # DD-MM-YYYY               e.g. 15-03-2024
    "%B %d, %Y",              # Month DD, YYYY           e.g. March 15, 2024
    "%b %d, %Y",              # Mon DD, YYYY             e.g. Mar 15, 2024
    "%d %B %Y",               # DD Month YYYY            e.g. 15 March 2024
    "%d %b %Y",               # DD Mon YYYY              e.g. 15 Mar 2024
]

# ---------------------------------------------------------------------------
# Arabic month helpers
# ---------------------------------------------------------------------------

# Arabic month name → month number (includes common spelling variants)
_AR_MONTHS = {
    "يناير": 1,
    "فبراير": 2,
    "مارس": 3,
    "أبريل": 4,
    "ابريل": 4,
    "مايو": 5,
    "يونيو": 6,
    "يوليو": 7,
    "أغسطس": 8,
    "اغسطس": 8,
    "سبتمبر": 9,
    "أكتوبر": 10,
    "اكتوبر": 10,
    "نوفمبر": 11,
    "ديسمبر": 12,
}

# Matches: "DD MonthAR [YYYY] HH:MM ص|م"
# Year group is optional; ص = AM, م = PM
_AR_DATE_RE = re.compile(
    r"""^(\d{1,2})\s+       # day
        ([\u0600-\u06FF]+)\s+ # Arabic month name
        (?:(\d{4})\s+)?       # optional year
        (\d{1,2}:\d{2})\s+   # HH:MM
        ([صم])$               # ص=AM, م=PM
    """,
    re.VERBOSE,
)


def _parse_arabic_date(text: str) -> str:
    """Parse an Arabic-locale date string and return YYYY-MM-DD, or '' on failure.

    Supported patterns
    ------------------
    With year:    "29 ديسمبر 2025 04:36 م"  →  "2025-12-29"
    Without year: "19 فبراير 01:57 م"        →  "<current_year>-02-19"
    """
    m = _AR_DATE_RE.match(text.strip())
    if not m:
        return ""
    day, month_ar, year, _time, _ampm = m.groups()
    month = _AR_MONTHS.get(month_ar)
    if not month:
        return ""
    if not year:
        year = str(datetime.now().year)
    return f"{int(year):04d}-{month:02d}-{int(day):02d}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def normalize_date(raw: str) -> str:
    """Return *raw* as a YYYY-MM-DD string, or the original value if parsing fails.

    Parsing order
    -------------
    1. Arabic locale format  (DD arabic-month [YYYY] HH:MM ص/م)
    2. Common Latin formats  (ISO 8601, DD/MM/YYYY, month-name strings …)
    3. ISO prefix extraction (e.g. "2024-03-15T…")
    4. Return raw unchanged  (last-resort fallback)
    """
    if not raw:
        return raw
    cleaned = raw.strip()

    # 1. Arabic format
    ar = _parse_arabic_date(cleaned)
    if ar:
        return ar

    # 2. Latin / ISO formats
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(cleaned, fmt).strftime("%Y-%m-%d")
        except ValueError:
            pass

    # 3. ISO prefix fallback
    m = re.match(r"(\d{4}-\d{2}-\d{2})", cleaned)
    if m:
        return m.group(1)

    # 4. Give up – return as-is
    return cleaned
