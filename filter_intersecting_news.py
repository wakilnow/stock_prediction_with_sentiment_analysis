import pandas as pd
import os

def filter_intersection():
    files = [
        'data/news_investing.com/jpm_news.csv',
        'data/news_investing.com/bac_news.csv',
        'data/news10/COMI_mubasher.csv',
        'data/news10/CIEB_mubasher.csv'
    ]

    date_sets = []
    dfs = []

    # Read all files and extract dates
    for f in files:
        df = pd.read_csv(f)
        d_col = 'Date' if 'Date' in df.columns else 'date'
        
        # Parse dates
        df['_parsed_date'] = pd.to_datetime(df[d_col], errors='coerce').dt.date
        df = df.dropna(subset=['_parsed_date'])
        
        dfs.append(df)
        date_sets.append(set(df['_parsed_date']))

    # Find the intersection
    intersection = set.intersection(*date_sets)
    print(f"Found {len(intersection)} overlapping days.")

    # Filter and save
    for f, df in zip(files, dfs):
        df_filtered = df[df['_parsed_date'].isin(intersection)].copy()
        df_filtered = df_filtered.drop(columns=['_parsed_date'])
        
        out_path = f.replace('.csv', '_intersected.csv')
        df_filtered.to_csv(out_path, index=False)
        print(f"Saved {len(df_filtered)} rows to {out_path}")

if __name__ == "__main__":
    filter_intersection()
