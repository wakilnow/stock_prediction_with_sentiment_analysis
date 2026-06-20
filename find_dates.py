import pandas as pd
import glob
import os

prices = glob.glob('data/prices/*.csv')
news_en = glob.glob('data/news_investing.com/*_news.csv')
news_ar = glob.glob('data/news10/*.csv')

starts = []
ends = []

for f in prices:
    df = pd.read_csv(f)
    if 'Date' in df.columns: d = df['Date']
    elif 'date' in df.columns: d = df['date']
    d = pd.to_datetime(d, errors='coerce').dropna()
    starts.append(d.min())
    ends.append(d.max())
    print(f"{f}: {d.min().date()} to {d.max().date()}")

for f in news_en + news_ar:
    if "augmented" in f or "formatted" in f or "test" in f: continue
    df = pd.read_csv(f)
    if 'Date' in df.columns: d = df['Date']
    elif 'date' in df.columns: d = df['date']
    d = pd.to_datetime(d, errors='coerce').dropna()
    if len(d) > 0:
        starts.append(d.min())
        ends.append(d.max())
        print(f"{f}: {d.min().date()} to {d.max().date()}")

print("Common Intersection:")
print(f"Max Start Date: {max(starts).date()}")
print(f"Min End Date: {min(ends).date()}")
