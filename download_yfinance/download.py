#!/usr/bin/env python3
"""
Download historical OHLCV data from Yahoo Finance via yfinance.

Output CSVs have columns: Date, Open, High, Low, Close, Volume, Ticker

Usage examples
--------------
# Single ticker – EGX tickers need the .CA suffix on Yahoo Finance
python download_yfinance/download.py --tickers COMI.CA

# Multiple tickers
python download_yfinance/download.py --tickers COMI.CA CIEB.CA ETEL.CA

# Custom date range and output directory
python download_yfinance/download.py \\
    --tickers COMI.CA CIEB.CA \\
    --start 2020-01-01 \\
    --end   2026-02-21 \\
    --out-dir data/prices

Options
-------
--tickers    One or more Yahoo Finance ticker symbols (required)
--start      Start date  YYYY-MM-DD  (default: 2018-01-01)
--end        End date    YYYY-MM-DD  (default: today)
--interval   Bar size: 1d 1wk 1mo   (default: 1d)
--out-dir    Output directory        (default: data)
"""

import argparse
import os
import sys
from datetime import date

import yfinance as yf


def download_ticker(
    ticker: str,
    start: str,
    end: str,
    interval: str,
    out_dir: str,
) -> str:
    """Download data for a single ticker and save to CSV. Returns the output path."""
    print(f"Downloading {ticker}  [{start} → {end}]  interval={interval}")
    df = yf.download(
        ticker,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True,   # adjusted OHLCV (splits + dividends)
        progress=False,
    )

    if df.empty:
        print(f"  WARNING: no data returned for {ticker}")
        return ""

    # Flatten multi-level columns that yfinance sometimes produces
    if isinstance(df.columns, yf.utils.pd.MultiIndex if hasattr(yf.utils, "pd") else type(df.columns)):
        try:
            df.columns = df.columns.get_level_values(0)
        except Exception:
            pass

    df.index.name = "Date"
    df.reset_index(inplace=True)

    # Normalise Date column to plain YYYY-MM-DD strings
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")

    # Add ticker column for convenience when merging files later
    df.insert(0, "Ticker", ticker)

    os.makedirs(out_dir, exist_ok=True)
    safe_name = ticker.replace(".", "_")
    out_path = os.path.join(out_dir, f"{safe_name}.csv")
    df.to_csv(out_path, index=False)
    print(f"  Saved {len(df)} rows → {out_path}")
    return out_path


def main():
    today = date.today().strftime("%Y-%m-%d")

    p = argparse.ArgumentParser(
        description="Download historical stock prices from Yahoo Finance."
    )
    p.add_argument(
        "--tickers",
        nargs="+",
        required=True,
        metavar="TICKER",
        help="Yahoo Finance ticker symbol(s), e.g. COMI.CA CIEB.CA",
    )
    p.add_argument("--start",    default="2018-01-01", help="Start date YYYY-MM-DD")
    p.add_argument("--end",      default=today,        help="End date   YYYY-MM-DD")
    p.add_argument("--interval", default="1d",
                   choices=["1m","2m","5m","15m","30m","60m","90m",
                            "1h","1d","5d","1wk","1mo","3mo"],
                   help="Bar interval (default: 1d)")
    p.add_argument("--out-dir",  default="data",       help="Output directory")
    args = p.parse_args()

    saved = []
    failed = []
    for ticker in args.tickers:
        path = download_ticker(
            ticker=ticker,
            start=args.start,
            end=args.end,
            interval=args.interval,
            out_dir=args.out_dir,
        )
        (saved if path else failed).append(ticker)

    print()
    print(f"Done. Saved: {len(saved)}  Failed: {len(failed)}")
    if failed:
        print("Failed tickers:", ", ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
