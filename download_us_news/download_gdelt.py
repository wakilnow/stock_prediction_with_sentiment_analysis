#!/usr/bin/env python3
"""
Scrape / download historical news headlines from the GDELT DOC 2.0 API.

GDELT DOC 2.0 API  – free, no API key, ~2013-present, full date range.
https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/

Usage examples
--------------
# Download by specific query (e.g., "economy" or '"Apple Inc"')
python download_us_news/download_gdelt.py --query "economy" \
    --start 2024-01-01 --end 2024-01-31 \
    --out-dir data/news

# Download by stock tickers (will query each ticker separately)
python download_us_news/download_gdelt.py --tickers JPM BAC \
    --start 2023-01-01 --end 2023-12-31 \
    --out-dir data/news

Options
-------
--query     A specific string query for GDELT
--tickers   One or more ticker symbols to search for
--start     Keep only articles on/after YYYY-MM-DD (default: no filter, goes back to 2013)
--end       Keep only articles on/before YYYY-MM-DD (default: no filter)
--out-dir   Output directory (default: data)
--sleep     Seconds between tickers/requests (default: 6.0)
--max-gdelt Max GDELT records to fetch per 90-day chunk (default: 250, max API allows is 250)
--theme     Optional GDELT theme to restrict search (e.g., ECON_STOCK)
"""

import argparse
import csv
import os
import time
from datetime import datetime, date, timedelta

import requests

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
# GDELT Download Logic
# ---------------------------------------------------------------------------

def fetch_gdelt_news(
    query: str,
    session: requests.Session,
    start: str | None = None,
    end: str | None = None,
    max_records: int = 250,
    theme: str | None = None,
) -> list[dict]:
    """
    Query the GDELT DOC 2.0 Article Search API.
    
    GDELT covers global news from ~2013 to the present.
    It supports full date range filtering via startdatetime / enddatetime.
    Max records per request: 250 (API limit).
    Requires query to be >= 4 characters.
    """
    articles: list[dict] = []
    
    # Strip quotes for length check length
    raw_query = query.strip('"').strip("'")
    if len(raw_query) < 4:
        print(f"  [GDELT] WARNING: The query '{raw_query}' is too short. GDELT requires at least 4 characters.")

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
        # Build API URL
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
        # Optional theme filter
        if theme:
            url += f"&theme={theme}"

        # Add a retry loop for API rate limits and sporadic failures
        max_retries = 3
        data = None
        for attempt in range(max_retries):
            try:
                # GDELT strictly requires 1 request per 5 seconds
                resp = session.get(url, timeout=30)
                if resp.status_code == 429:
                    wait_time = 6 + (attempt * 5)
                    print(f"  [GDELT] Rate limited (429). Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                resp.raise_for_status()
                try:
                    data = resp.json()
                except ValueError as ve:
                    # GDELT sends 200 OK with plain text errors sometimes
                    if "too short" in resp.text.lower():
                        print(f"  [GDELT] API Error: {resp.text.strip()}")
                        data = None
                        break # Skip retries for this fatal error
                    else:
                        print(f"  [GDELT] Could not parse JSON response (attempt {attempt+1}/{max_retries}). Response: {resp.text[:100]}")
                        break
                break # Success
            except requests.RequestException as e:
                wait_time = 6 + (attempt * 5)
                # JSONDecodeError might be caught here if using newer requests
                if hasattr(e, "response") and e.response is not None and "too short" in e.response.text.lower():
                    print(f"  [GDELT] API Error: {e.response.text.strip()}")
                    data = None
                    break
                print(f"  [GDELT] Request error (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(wait_time)
                    continue
                else:
                    break
        
        if data is None:
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
                "query":     query,
            })

        # Polite pause between chunks to respect GDELT's 5s limit
        if chunk_end != chunks[-1][1]:
            # Always sleep at least 5.5 seconds between requests
            time.sleep(6.0)

    return articles


def download_gdelt_news(
    query: str,
    start: str | None,
    end: str | None,
    out_dir: str,
    session: requests.Session,
    max_gdelt: int = 250,
    theme: str | None = None,
    filename_prefix: str = "gdelt",
) -> None:
    print(f"\nFetching news for query: '{query}'")
    articles = fetch_gdelt_news(
        query=query, session=session, start=start, end=end,
        max_records=max_gdelt, theme=theme
    )
    print(f"  GDELT → {len(articles)} articles")

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
                "GDELT coverage starts ~2013; gaps may exist for some queries/periods."
            )
        print(f"  WARNING: nothing to save for '{query}'")
        return

    os.makedirs(out_dir, exist_ok=True)
    # create a safe filename from prefix
    safe_prefix = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in filename_prefix).strip("_")
    out_path = os.path.join(out_dir, f"{safe_prefix}_news.csv")
    fieldnames = ["date", "title", "publisher", "url", "source", "query"]
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
            "Download historical news headlines using GDELT DOC 2.0 API.\n"
            "Supports searching by explicit query or by stock ticker(s)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Allow either --query or --tickers
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--query",    type=str,
                       help="A specific query string for GDELT (e.g., 'economy')")
    group.add_argument("--tickers",  nargs="+", metavar="TICKER",
                       help="One or more ticker symbols, e.g.  JPM BAC GS")

    p.add_argument("--start",      default=None,
                   help="Keep articles on/after YYYY-MM-DD")
    p.add_argument("--end",        default=None,
                   help="Keep articles on/before YYYY-MM-DD")
    p.add_argument("--out-dir",    default="data",
                   help="Output directory (default: data)")
    p.add_argument("--sleep",      type=float, default=6.0,
                   help="Seconds between tickers/queries (default: 6.0)")
    p.add_argument("--max-gdelt",  type=int,   default=250,
                   help="Max GDELT records per 90-day chunk (default: 250, max: 250)")
    p.add_argument("--theme",      default=None,
                   help="Optional GDELT theme to restrict search (e.g., ECON_STOCK)")
    args = p.parse_args()

    session = requests.Session()
    session.headers.update(_HEADERS)

    if args.query:
        # Just one query
        download_gdelt_news(
            query=args.query,
            start=args.start,
            end=args.end,
            out_dir=args.out_dir,
            session=session,
            max_gdelt=args.max_gdelt,
            theme=args.theme,
            filename_prefix=args.query,
        )
    elif args.tickers:
        # Multiple tickers
        for i, ticker in enumerate(args.tickers):
            # For tickers, use literal quotes for exact match of ticker symbol
            q = f'"{ticker.upper()}"'
            download_gdelt_news(
                query=q,
                start=args.start,
                end=args.end,
                out_dir=args.out_dir,
                session=session,
                max_gdelt=args.max_gdelt,
                theme=args.theme,
                filename_prefix=ticker.upper(),
            )
            if i < len(args.tickers) - 1:
                time.sleep(args.sleep)

    print("\nAll done.")


if __name__ == "__main__":
    main()
