#!/usr/bin/env python3
"""
Scrape / download historical US stock news headlines.

Sources
-------
1. GDELT DOC 2.0 API  – free, no API key, ~2013-present, full date range.
   https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/

2. Finviz             – free scrape, no API key, last ~100 articles (recent only).
   https://finviz.com/quote.ashx?t=JPM

GDELT is the recommended source for any historical range (2013-present).
Finviz is useful only for the latest news (past few days).

Output CSV columns: date, title, url, publisher, source, ticker

Usage examples
--------------
# GDELT – historical range (recommended)
python download_us_news/download_finviz.py --tickers JPM BAC \\
    --start 2020-01-01 --end 2023-12-31 \\
    --source gdelt --out-dir data/news

# Finviz – recent headlines only
python download_us_news/download_finviz.py --tickers JPM --source finviz

# Both sources combined
python download_us_news/download_finviz.py --tickers JPM \\
    --start 2023-01-01 --source both

Options
-------
--tickers   One or more ticker symbols (required)
--start     Keep only articles on/after YYYY-MM-DD (default: no filter)
--end       Keep only articles on/before YYYY-MM-DD (default: no filter)
--source    gdelt | finviz | both  (default: gdelt)
--out-dir   Output directory (default: data)
--sleep     Seconds between tickers (default: 2.0)
--max-gdelt Max GDELT records to fetch per ticker (default: 250)
"""

import argparse
import csv
import os
import re
import time
from datetime import datetime, date, timedelta

import requests
from bs4 import BeautifulSoup


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# GDELT DOC 2.0 Article Search API
# Docs: https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/
_GDELT_API = (
    "https://api.gdeltproject.org/api/v2/doc/doc"
    "?query={query}"
    "&mode=artlist"
    "&maxrecords={max_records}"
    "&timespan={timespan}"
    "&startdatetime={start_dt}"
    "&enddatetime={end_dt}"
    "&format=json"
    "&sort=DateDesc"
    "&sourcelang=english"
    "&theme=ECON_STOCK"            # Restrict to finance/stock articles
)

# Finviz quote page
_FINVIZ_URL = "https://finviz.com/quote.ashx?t={ticker}&p=d"

# Finviz date formats: "Feb-25-26" (2-digit year) and "Feb-25-2026" (4-digit)
_DATE_FMTS = ["%b-%d-%y", "%b-%d-%Y"]
_TIME_FMT  = "%I:%M%p"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _in_range(date_str: str, start: str | None, end: str | None) -> bool:
    """Return True if date_str (YYYY-MM-DD) falls within [start, end]."""
    if not date_str:
        return True
    if start and date_str < start:
        return False
    if end and date_str > end:
        return False
    return True


def _to_gdelt_dt(d: str, end_of_day: bool = False) -> str:
    """Convert YYYY-MM-DD to GDELT datetime format YYYYMMDDHHmmSS."""
    dt = datetime.strptime(d, "%Y-%m-%d")
    if end_of_day:
        dt = dt.replace(hour=23, minute=59, second=59)
    return dt.strftime("%Y%m%d%H%M%S")


# ---------------------------------------------------------------------------
# Source 1 – GDELT DOC 2.0 API
# ---------------------------------------------------------------------------

def fetch_gdelt_news(
    ticker: str,
    session: requests.Session,
    start: str | None = None,
    end: str | None = None,
    max_records: int = 250,
) -> list[dict]:
    """
    Query the GDELT DOC 2.0 Article Search API for stock news.

    GDELT covers global news from ~2013 to the present.
    It supports full date range filtering via startdatetime / enddatetime.
    Max records per request: 250 (API limit).
    """
    articles: list[dict] = []

    # GDELT needs absolute start/end datetimes.
    today = date.today()
    start_dt = _to_gdelt_dt(start) if start else _to_gdelt_dt("2013-01-01")
    end_dt   = _to_gdelt_dt(end, end_of_day=True) if end else _to_gdelt_dt(
        today.strftime("%Y-%m-%d"), end_of_day=True
    )

    # Timespan in days (GDELT requires this too; max 3 months per call)
    s = datetime.strptime(start_dt[:8], "%Y%m%d")
    e = datetime.strptime(end_dt[:8], "%Y%m%d")
    total_days = max((e - s).days, 1)

    # GDELT API limits to ~90 days per call, so we chunk long ranges
    chunk_days = 90
    chunks: list[tuple[str, str]] = []
    cursor = s
    while cursor <= e:
        chunk_end = min(cursor + timedelta(days=chunk_days - 1), e)
        chunks.append((
            cursor.strftime("%Y%m%d%H%M%S"),
            chunk_end.replace(hour=23, minute=59, second=59).strftime("%Y%m%d%H%M%S"),
        ))
        cursor = chunk_end + timedelta(days=1)

    print(f"  [GDELT] Querying {len(chunks)} chunk(s) for {total_days} day range...")

    for chunk_start, chunk_end in chunks:
        # Use ticker as query; company name may be appended for better coverage
        query = f'"{ticker}"'
        url = (
            "https://api.gdeltproject.org/api/v2/doc/doc"
            f"?query={requests.utils.quote(query)}"
            f"&mode=artlist"
            f"&maxrecords={max_records}"
            f"&startdatetime={chunk_start}"
            f"&enddatetime={chunk_end}"
            f"&format=json"
            f"&sort=DateDesc"
            f"&sourcelang=english"
        )
        try:
            resp = session.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            print(f"  [GDELT] Request error: {e}")
            continue
        except ValueError:
            print(f"  [GDELT] Could not parse JSON response.")
            continue

        items = data.get("articles", [])
        for item in items:
            raw_date = item.get("seendate", "")
            # GDELT seendate format: "20231015T123000Z"
            try:
                art_date = datetime.strptime(raw_date[:8], "%Y%m%d").strftime("%Y-%m-%d")
            except (ValueError, TypeError):
                art_date = ""

            if not _in_range(art_date, start, end):
                continue

            articles.append({
                "date":      art_date,
                "title":     item.get("title", "").strip(),
                "url":       item.get("url", "").strip(),
                "publisher": item.get("domain", "").strip(),
                "source":    "gdelt",
                "ticker":    ticker.upper(),
            })

        # Polite pause between chunks
        if chunk_end != chunks[-1][1]:
            time.sleep(1)

    return articles


# ---------------------------------------------------------------------------
# Source 2 – Finviz scraper (recent headlines only, ~last 100 articles)
# ---------------------------------------------------------------------------

def _parse_finviz_datetime(cell_text: str, last_known: str) -> tuple[str, str]:
    """
    Parse Finviz's combined date-time cell.

    Cell examples:
      "Today 02:15AM"       -> today's date
      "Feb-25-26 09:31PM"   -> explicit date + time
      "07:34PM"             -> time only, reuse last_known date
    """
    date_str = last_known
    time_str = ""

    parts = cell_text.strip().split()
    raw_time_token = ""

    if len(parts) == 1:
        raw_time_token = parts[0]
    elif len(parts) >= 2:
        raw_date_token = parts[0]
        raw_time_token = parts[1]
        if raw_date_token.lower() == "today":
            date_str = date.today().strftime("%Y-%m-%d")
        else:
            for fmt in _DATE_FMTS:
                try:
                    date_str = datetime.strptime(raw_date_token, fmt).strftime("%Y-%m-%d")
                    break
                except ValueError:
                    pass

    raw_time_clean = re.sub(r"\s+", "", raw_time_token).upper()
    try:
        time_str = datetime.strptime(raw_time_clean, _TIME_FMT).strftime("%H:%M")
    except ValueError:
        time_str = raw_time_token

    return date_str, time_str


def fetch_finviz_news(
    ticker: str,
    session: requests.Session,
    start: str | None = None,
    end: str | None = None,
) -> list[dict]:
    """
    Scrape Finviz quote page for the latest headlines.

    NOTE: Finviz free pages only show the last ~100 articles (a few days).
    For any historical range older than a week, use GDELT instead.
    """
    articles: list[dict] = []
    url = _FINVIZ_URL.format(ticker=ticker.upper())

    try:
        resp = session.get(url, timeout=20)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  [Finviz] Request error for {ticker}: {e}")
        return articles

    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table", id="news-table")
    if table is None:
        print(f"  [Finviz] News table not found for {ticker}.")
        return articles

    last_date = date.today().strftime("%Y-%m-%d")

    for row in table.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) < 2:
            continue

        date_time_cell = cells[0].get_text(strip=True)
        headline_cell  = cells[1]

        article_date, _ = _parse_finviz_datetime(date_time_cell, last_date)
        if article_date:
            last_date = article_date

        if not _in_range(article_date, start, end):
            continue

        link_tag = headline_cell.find("a")
        if link_tag is None:
            continue

        title = link_tag.get_text(strip=True)
        article_url = link_tag.get("href", "")

        source_span = headline_cell.find("span", class_=re.compile(r"label", re.I))
        if source_span is None:
            source_span = headline_cell.find("span")
        publisher = (
            source_span.get_text(strip=True).strip("()") if source_span else "Finviz"
        )

        if title:
            articles.append({
                "date":      article_date,
                "title":     title,
                "url":       article_url,
                "publisher": publisher,
                "source":    "finviz",
                "ticker":    ticker.upper(),
            })

    return articles


# ---------------------------------------------------------------------------
# Per-ticker pipeline
# ---------------------------------------------------------------------------

def download_ticker_news(
    ticker: str,
    start: str | None,
    end: str | None,
    source: str,
    out_dir: str,
    session: requests.Session,
    max_gdelt: int = 250,
) -> None:
    print(f"\nFetching news for {ticker}  [source={source}]")
    articles: list[dict] = []

    if source in ("gdelt", "both"):
        gdelt_arts = fetch_gdelt_news(ticker, session, start=start, end=end,
                                      max_records=max_gdelt)
        print(f"  GDELT       → {len(gdelt_arts)} articles")
        articles.extend(gdelt_arts)

    if source in ("finviz", "both"):
        if source == "both" and start and start < "2024-01-01":
            print(
                "  [Finviz] Skipping – Finviz only has recent articles (~last week). "
                "Use --source finviz if you only need latest headlines."
            )
        else:
            fin_arts = fetch_finviz_news(ticker, session, start=start, end=end)
            print(f"  Finviz      → {len(fin_arts)} articles")
            articles.extend(fin_arts)

    # Deduplicate by URL (or title if URL missing)
    seen: set[str] = set()
    unique: list[dict] = []
    for a in articles:
        key = a.get("url") or a.get("title", "")
        if key and key not in seen:
            seen.add(key)
            unique.append(a)

    # Sort newest first
    unique.sort(key=lambda a: a.get("date", ""), reverse=True)

    print(f"  After dedup → {len(unique)} articles")

    if not unique:
        if start or end:
            print(
                f"  NOTE: No articles found in [{start or 'any'} → {end or 'any'}]. "
                "GDELT coverage starts ~2013; gaps may exist for some tickers/periods."
            )
        print(f"  WARNING: nothing to save for {ticker}")
        return

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{ticker}_news.csv")
    fieldnames = ["date", "title", "publisher", "url", "source", "ticker"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(unique)

    print(f"  Saved → {out_path}  ({len(unique)} rows)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Download historical stock news headlines.\n"
            "Primary source: GDELT DOC 2.0 API (free, ~2013-present).\n"
            "Secondary source: Finviz scrape (free, recent ~100 articles only)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--tickers",    nargs="+", required=True, metavar="TICKER",
                   help="Ticker symbol(s), e.g.  JPM BAC GS")
    p.add_argument("--start",      default=None,
                   help="Keep articles on/after YYYY-MM-DD")
    p.add_argument("--end",        default=None,
                   help="Keep articles on/before YYYY-MM-DD")
    p.add_argument("--source",     default="gdelt",
                   choices=["gdelt", "finviz", "both"],
                   help="News source (default: gdelt)")
    p.add_argument("--out-dir",    default="data",
                   help="Output directory (default: data)")
    p.add_argument("--sleep",      type=float, default=2.0,
                   help="Seconds between tickers (default: 2.0)")
    p.add_argument("--max-gdelt",  type=int,   default=250,
                   help="Max GDELT records per 90-day chunk (default: 250, max: 250)")
    args = p.parse_args()

    session = requests.Session()
    session.headers.update(_HEADERS)

    for i, ticker in enumerate(args.tickers):
        download_ticker_news(
            ticker=ticker.upper(),
            start=args.start,
            end=args.end,
            source=args.source,
            out_dir=args.out_dir,
            session=session,
            max_gdelt=args.max_gdelt,
        )
        if i < len(args.tickers) - 1:
            time.sleep(args.sleep)

    print("\nAll done.")


if __name__ == "__main__":
    main()
