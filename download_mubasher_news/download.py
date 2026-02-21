#!/usr/bin/env python3
"""
Download Mubasher news list using cookies saved in `cookies.json`.

Writes a CSV with columns: date,title,url

Usage examples:
  .venv9/bin/python download_mubasher_news/download_with_cookies.py \
      --cookies download_news/cookies.json \
      --base-url "https://www.mubasher.info/markets/EGX/stocks/COMI/news/" \
      --out-csv result/news/COMI.csv \
      --sleep 0.3 --fetch-dates

Options:
  --pages N        Stop after N pages (0 = until no more new items)
  --fetch-dates    Fetch each article page to extract publication date when listing has no date
"""

import argparse
import csv
import json
import os
import re
import time
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from date_utils import normalize_date


def load_cookies(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Support either dict {name: value} or list of {name:, value:} browser exports
    if isinstance(data, dict):
        return data
    cookies = {}
    if isinstance(data, list):
        for item in data:
            name = item.get("name") or item.get("Name")
            value = item.get("value") or item.get("Value")
            if name and value is not None:
                cookies[name] = value
    return cookies


HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}


def fetch_page(session, url, timeout=15):
    try:
        r = session.get(url, headers=HEADERS, timeout=timeout)
        if r.status_code == 200:
            return r.text
    except Exception:
        pass
    return None


def parse_listing(html, base_url):
    soup = BeautifulSoup(html, "html.parser")
    found = []

    # Look for links matching /news/<digits>/ (canonical news URL pattern)
    for a in soup.find_all("a", href=True):
        href = a["href"]
        # Match /news/<digits>/ or /news/<digits>/<slug>
        if re.match(r"/news/\d+", href):
            url = urljoin(base_url, href)
            title = a.get_text(strip=True)
            if not title:
                continue
            # look for date nearby in parent or siblings
            date = ""
            parent = a.parent
            if parent:
                t = parent.find("time") or parent.find(
                    "span", class_=lambda c: c and "date" in c.lower()
                )
                if t:
                    date = normalize_date(t.get("datetime") or t.get_text(strip=True))
            found.append({"title": title, "date": date, "url": url})

    # Deduplicate by url while preserving order
    out = []
    seen = set()
    for it in found:
        if it["url"] not in seen:
            seen.add(it["url"])
            out.append(it)
    return out


def extract_date_from_article(session, url):
    html = fetch_page(session, url)
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    # common meta tags
    metas = [
        ("meta", {"property": "article:published_time"}),
        ("meta", {"name": "pubdate"}),
        ("meta", {"name": "publishdate"}),
        ("meta", {"itemprop": "datePublished"}),
        ("meta", {"name": "date"}),
    ]
    for tag, attrs in metas:
        el = soup.find(tag, attrs=attrs)
        if el and el.get("content"):
            return normalize_date(el.get("content").strip())

    time_el = soup.find("time")
    if time_el:
        if time_el.get("datetime"):
            return normalize_date(time_el.get("datetime").strip())
        txt = time_el.get_text(strip=True)
        if txt:
            return normalize_date(txt)

    # JSON-LD
    for s in soup.find_all("script", type="application/ld+json"):
        try:
            obj = json.loads(s.string or "{}")
        except Exception:
            continue

        # simplistic search
        def walk(o):
            if isinstance(o, dict):
                if "datePublished" in o and o["datePublished"]:
                    return o["datePublished"]
                for v in o.values():
                    r = walk(v)
                    if r:
                        return r
            elif isinstance(o, list):
                for item in o:
                    r = walk(item)
                    if r:
                        return r
            return None

        d = walk(obj)
        if d:
            return normalize_date(str(d))

    # fallback: small text near the title
    h1 = soup.find("h1")
    if h1 and h1.parent:
        for el in h1.parent.find_all(["span", "div"], limit=6):
            txt = el.get_text(strip=True)
            if txt and re.search(r"\d{4}|\d{1,2}\s", txt):
                return normalize_date(txt)

    return ""


def save_csv(items, out_csv):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    keys = ["date", "title", "url"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for it in items:
            w.writerow({k: it.get(k, "") for k in keys})


def download_all(
    base_url,
    cookies_path,
    pages=0,
    sleep=0.5,
    fetch_dates=False,
    out_csv="download_news/news_items.csv",
):
    session = requests.Session()
    if os.path.exists(cookies_path):
        try:
            cookies = load_cookies(cookies_path)
            session.cookies.update(cookies)
        except Exception as e:
            print("Failed to load cookies:", e)
    else:
        print("Warning: cookies file not found:", cookies_path)

    if not base_url.endswith("/"):
        base_url = base_url + "/"

    all_items = []
    existing_urls = set()
    page = 1
    consecutive_empty = 0
    max_empty = 2

    while True:
        page_url = urljoin(base_url, str(page))
        print("Fetching page", page, page_url)
        html = fetch_page(session, page_url)
        if not html:
            print("No content for page", page, "- stopping")
            break

        items = parse_listing(html, base_url)
        new_items = []
        for it in items:
            if it["url"] not in existing_urls:
                existing_urls.add(it["url"])
                new_items.append(it)

        print("  Found", len(items), "items, new", len(new_items))

        if not new_items:
            consecutive_empty += 1
            if pages == 0 and consecutive_empty >= max_empty:
                print("No new items for several pages, stopping.")
                break
        else:
            consecutive_empty = 0
            # optionally fetch date from article if missing
            if fetch_dates:
                for it in new_items:
                    if not it.get("date"):
                        it["date"] = extract_date_from_article(session, it["url"])
                        time.sleep(sleep)
            # ensure date key present
            for it in new_items:
                if "date" not in it:
                    it["date"] = ""
            all_items.extend(new_items)

        page += 1
        if pages and page > pages:
            break
        time.sleep(sleep)

    save_csv(all_items, out_csv)
    print("Saved", len(all_items), "items to", out_csv)
    return out_csv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cookies", default="download_news/cookies.json")
    p.add_argument("--base-url", required=True)
    p.add_argument("--pages", type=int, default=0)
    p.add_argument("--sleep", type=float, default=0.5)
    p.add_argument("--fetch-dates", action="store_true")
    p.add_argument("--out-csv", default="download_news/news_items.csv")
    args = p.parse_args()
    download_all(
        args.base_url,
        args.cookies,
        pages=args.pages,
        sleep=args.sleep,
        fetch_dates=args.fetch_dates,
        out_csv=args.out_csv,
    )


if __name__ == "__main__":
    main()
