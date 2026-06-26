import pandas as pd

files = [
    'data/news_investing.com/jpm_news.csv',
    'data/news_investing.com/bac_news.csv',
    'data/news10/COMI_mubasher.csv',
    'data/news10/CIEB_mubasher.csv'
]

date_sets = []

for f in files:
    df = pd.read_csv(f)
    if 'Date' in df.columns: d_col = 'Date'
    elif 'date' in df.columns: d_col = 'date'
    
    dates = pd.to_datetime(df[d_col], errors='coerce').dt.date.dropna()
    unique_dates = set(dates)
    date_sets.append(unique_dates)
    print(f"{f}: {len(unique_dates)} unique days with news")

intersection = set.intersection(*date_sets)
print(f"Number of days where ALL 4 stocks have news: {len(intersection)}")

if len(intersection) > 0:
    print(f"Sample overlapping dates: {list(intersection)[:5]}")
