#!/usr/bin/env python3
"""
Download US-market news headlines for one or more tickers.

Sources (no API key required)
------------------------------
1. yfinance  – ticker.news      (~10 very recent articles, instant)
2. Google News RSS              (up to ~100 articles per query, good date range)

Output CSV columns: date, title, publisher, url, ticker

Usage examples
--------------
# Single ticker
python download_us_news/download.py --tickers JPM

# Multiple tickers with a date filter
python download_us_news/download.py --tickers JPM BAC GS \\
    --start 2020-01-01 \\
    --out-dir data/news

# Use only yfinance (fastest, ~10 items)
python download_us_news/download.py --tickers JPM --source yfinance

# Use only Google News RSS (~100 items)
python download_us_news/download.py --tickers JPM --source rss

Options
-------
--tickers     One or more ticker symbols (required)
--start       Keep only articles on/after YYYY-MM-DD (default: no filter)
--end         Keep only articles on/before YYYY-MM-DD (default: no filter)
--source      yfinance | rss | both  (default: both)
--out-dir     Output directory (default: data)
--sleep       Seconds between tickers to be polite (default: 1.0)
"""

import argparse
import csv
import os
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus

import requests
import yfinance as yf


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts_to_date(ts: int) -> str:
    """Unix timestamp → YYYY-MM-DD."""
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")


def _rfc2822_to_date(text: str) -> str:
    """RFC-2822 <pubDate> string → YYYY-MM-DD."""
    try:
        return parsedate_to_datetime(text).strftime("%Y-%m-%d")
    except Exception:
        return ""


def _in_range(date_str: str, start: str | None, end: str | None) -> bool:
    """Return True if date_str (YYYY-MM-DD) is within [start, end]."""
    if not date_str:
        return True   # keep items with unparseable dates
    if start and date_str < start:
        return False
    if end and date_str > end:
        return False
    return True


# ---------------------------------------------------------------------------
# Source 1 – yfinance .news
# ---------------------------------------------------------------------------

def fetch_yfinance_news(ticker: str) -> list[dict]:
    """Use yfinance Ticker.news to fetch recent headlines."""
    articles = []
    try:
        raw = yf.Ticker(ticker).news or []
        for item in raw:
            # yfinance ≥ 0.2.x wraps articles under a 'content' key
            if isinstance(item, dict) and "content" in item:
                content = item["content"]
                title = content.get("title", "")
                url = (
                    (content.get("canonicalUrl") or {}).get("url", "")
                    or (content.get("clickThroughUrl") or {}).get("url", "")
                )
                publisher = (content.get("provider") or {}).get("displayName", "")
                pub_iso = content.get("pubDate", "")
                if pub_iso:
                    try:
                        date = datetime.fromisoformat(
                            pub_iso.replace("Z", "+00:00")
                        ).strftime("%Y-%m-%d")
                    except Exception:
                        date = pub_iso[:10]
                else:
                    date = ""
            else:
                title = item.get("title", "")
                url = item.get("link", "")
                publisher = item.get("publisher", "")
                ts = item.get("providerPublishTime")
                date = _ts_to_date(ts) if ts else ""

            if title:
                articles.append(
                    {"date": date, "title": title,
                     "publisher": publisher, "url": url, "ticker": ticker}
                )
    except Exception as e:
        print(f"  [yfinance] error for {ticker}: {e}")
    return articles


# ---------------------------------------------------------------------------
# Source 2 – Google News RSS
# ---------------------------------------------------------------------------

# Google News RSS – query by ticker symbol
# Returns up to ~100 items; dates are RFC-2822 in <pubDate>
_GNEWS_URL = (
    "https://news.google.com/rss/search"
    "?q={query}&hl=en-US&gl=US&ceid=US:en"
)


def fetch_google_news(
    ticker: str,
    session: requests.Session,
    start: str | None = None,
    end: str | None = None,
) -> list[dict]:
    """Fetch news from Google News RSS for a ticker.

    Uses Google's ``after:`` / ``before:`` search operators so the feed
    itself is pre-filtered by date, which matters for historical ranges
    that would otherwise return zero results after local filtering.
    """
    articles = []
    # Build the query with server-side date operators when available.
    base = f"{ticker} stock"
    if start:
        base += f" after:{start}"
    if end:
        base += f" before:{end}"
    query = quote_plus(base)
    url = _GNEWS_URL.format(query=query)
    try:
        resp = session.get(url, timeout=15)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        for item in root.iter("item"):
            title     = (item.findtext("title")   or "").strip()
            link      = (item.findtext("link")    or "").strip()
            pub_date  = (item.findtext("pubDate") or "").strip()
            source_el = item.find("source")
            publisher = (
                source_el.text.strip()
                if source_el is not None and source_el.text
                else "Google News"
            )
            date = _rfc2822_to_date(pub_date)
            if title:
                articles.append(
                    {"date": date, "title": title,
                     "publisher": publisher, "url": link, "ticker": ticker}
                )
    except Exception as e:
        print(f"  [Google News] error for {ticker}: {e}")
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
) -> None:
    print(f"\nFetching news for {ticker}  (source={source})")
    articles: list[dict] = []

    if source in ("yfinance", "both"):
        yf_arts = fetch_yfinance_news(ticker)
        print(f"  yfinance     → {len(yf_arts)} articles")
        articles.extend(yf_arts)

    if source in ("rss", "both"):
        gn_arts = fetch_google_news(ticker, session, start=start, end=end)
        print(f"  Google News  → {len(gn_arts)} articles")
        articles.extend(gn_arts)

    # Deduplicate by URL (or title if URL is missing)
    seen: set[str] = set()
    unique: list[dict] = []
    for a in articles:
        key = a["url"] or a["title"]
        if key not in seen:
            seen.add(key)
            unique.append(a)

    # Apply date filter
    filtered = [a for a in unique if _in_range(a["date"], start, end)]

    # Sort newest first
    filtered.sort(key=lambda a: a["date"] or "", reverse=True)

    print(f"  After dedup + filter → {len(filtered)} articles")

    if not filtered:
        if unique:
            dates = [a["date"] for a in unique if a["date"]]
            if dates:
                print(
                    f"  NOTE: fetched {len(unique)} article(s) but none fall "
                    f"in [{start or 'any'} → {end or 'any'}]. "
                    f"Oldest fetched: {min(dates)}, newest fetched: {max(dates)}. "
                    "Google News RSS only serves recent articles; historical "
                    "data before ~2-3 years ago is typically unavailable via RSS."
                )
        print(f"  WARNING: nothing to save for {ticker}")
        return

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{ticker}_news.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["date", "title", "publisher", "url", "ticker"]
        )
        writer.writeheader()
        writer.writerows(filtered)

    print(f"  Saved → {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Download US stock news headlines (no API key needed)."
    )
    p.add_argument("--tickers", nargs="+", required=True, metavar="TICKER",
                   help="Ticker symbol(s), e.g.  JPM BAC")
    p.add_argument("--start",  default=None,
                   help="Keep articles on/after YYYY-MM-DD")
    p.add_argument("--end",    default=None,
                   help="Keep articles on/before YYYY-MM-DD")
    p.add_argument("--source", default="both",
                   choices=["yfinance", "rss", "both"],
                   help="News source: yfinance | rss (Google News) | both  "
                        "(default: both)")
    p.add_argument("--out-dir", default="data",
                   help="Output directory (default: data)")
    p.add_argument("--sleep", type=float, default=1.0,
                   help="Seconds to wait between tickers (default: 1.0)")
    args = p.parse_args()

    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"
        )
    })

    for i, ticker in enumerate(args.tickers):
        download_ticker_news(
            ticker=ticker.upper(),
            start=args.start,
            end=args.end,
            source=args.source,
            out_dir=args.out_dir,
            session=session,
        )
        if i < len(args.tickers) - 1:
            time.sleep(args.sleep)

    print("\nAll done.")


if __name__ == "__main__":
    main()
