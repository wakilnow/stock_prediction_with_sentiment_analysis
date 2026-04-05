import pandas as pd
import numpy as np
import torch
import os
from transformers import pipeline
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

import argparse

def prepare_data(prices_path, news_path, start_date=None, end_date=None, seq_length=30, test_size=0.2, save_dir="data/processed", include_sentiment=True):
    print("Loading datasets...")
    prices = pd.read_csv(prices_path)
    news = pd.read_csv(news_path)

    # Clean and format dates
    prices['Date'] = pd.to_datetime(prices['Date'])
    prices = prices.sort_values('Date')
    
    news_date_col = 'date' if 'date' in news.columns else 'Date'
    news[news_date_col] = pd.to_datetime(news[news_date_col])
    
    # Date Filtering
    if start_date:
        prices = prices[prices['Date'] >= pd.to_datetime(start_date)]
        news = news[news[news_date_col] >= pd.to_datetime(start_date)]
    if end_date:
        prices = prices[prices['Date'] <= pd.to_datetime(end_date)]
        news = news[news[news_date_col] <= pd.to_datetime(end_date)]

    
    news_date_col = 'date' if 'date' in news.columns else 'Date'
    news[news_date_col] = pd.to_datetime(news[news_date_col])
    
    # We only care about the 'Close' price for the numerical feature
    prices = prices[['Date', 'Close']]
    
    news_title_col = 'title' if 'title' in news.columns else 'Title'
    
    # Group news by date and aggregate titles
    print("Aggregating daily news...")
    daily_news = news.groupby(news_date_col)[news_title_col].apply(lambda x: ' '.join(x.dropna().astype(str))).reset_index()
    daily_news.rename(columns={news_date_col: 'Date', news_title_col: 'title'}, inplace=True)
    
    # Merge on Date. Use an inner join to keep only days where we have prices.
    merged = pd.merge(prices, daily_news, on='Date', how='left')
    
    # Fill missing news with empty string
    merged['title'] = merged['title'].fillna('')

    # Use FinBERT for sentiment extraction
    if include_sentiment:
        print("Extracting sentiment features using FinBERT...")
        device = 0 if torch.cuda.is_available() else (-1)
        
        if torch.backends.mps.is_available():
            # Fall back to CPU for huggingface pipelines to ensure compatibility, 
            # as MPS on some transformers versions can be unstable for specific ops
            device = -1
    
        try:
            # Newer transformers versions
            sentiment_pipeline = pipeline("text-classification", model="ProsusAI/finbert", top_k=None, device=device)
        except TypeError:
            # Older versions
            sentiment_pipeline = pipeline("text-classification", model="ProsusAI/finbert", return_all_scores=True, device=device)
    
        positive_scores, negative_scores, neutral_scores = [], [], []
        
        for text in tqdm(merged['title'], desc="Computing daily sentiment"):
            if text.strip() == '':
                positive_scores.append(0.0)
                negative_scores.append(0.0)
                neutral_scores.append(0.0)
            else:
                truncated_text = text[:1500]  # rough approximation to avoid exceeding max tokens
                try:
                    result = sentiment_pipeline(truncated_text)[0]
                    scores = {r['label']: r['score'] for r in result}
                    positive_scores.append(scores.get('positive', 0.0))
                    negative_scores.append(scores.get('negative', 0.0))
                    neutral_scores.append(scores.get('neutral', 0.0))
                except Exception as e:
                    # If error (e.g., still too long), fallback
                    positive_scores.append(0.0)
                    negative_scores.append(0.0)
                    neutral_scores.append(0.0)
                    
        merged['sentiment_positive'] = positive_scores
        merged['sentiment_negative'] = negative_scores
        merged['sentiment_neutral'] = neutral_scores

    
    # Scale Close Price
    print("Scaling features...")
    scaler = MinMaxScaler()
    merged['Close_Scaled'] = scaler.fit_transform(merged[['Close']])
    
    # Drop rows without Close prices if any
    merged = merged.dropna(subset=['Close_Scaled']).reset_index(drop=True)
    
    # Create Sequences
    print("Creating sliding windows...")
    if include_sentiment:
        features = merged[['Close_Scaled', 'sentiment_positive', 'sentiment_negative', 'sentiment_neutral']].values
    else:
        features = merged[['Close_Scaled']].values
    
    X, y = [], []
    for i in range(len(features) - seq_length):
        X.append(features[i:(i + seq_length)])
        y.append(features[i + seq_length, 0]) # 0 index is Close_Scaled
        
    X = np.array(X)
    y = np.array(y)
    
    # Train/Test Split (Chronological)
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Save Data
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "X_train.npy"), X_train)
    np.save(os.path.join(save_dir, "X_test.npy"), X_test)
    np.save(os.path.join(save_dir, "y_train.npy"), y_train)
    np.save(os.path.join(save_dir, "y_test.npy"), y_test)
    
    import joblib
    joblib.dump(scaler, os.path.join(save_dir, "scaler.save"))
    
    print(f"Data saved to {save_dir}/")
    print(f"Training shapes: X={X_train.shape}, y={y_train.shape}")
    print(f"Testing shapes: X={X_test.shape}, y={y_test.shape}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices", type=str, default="data/prices/JPM.csv", help="Path to prices CSV")
    parser.add_argument("--news", type=str, default="data/news_investing.com/jpm_news.csv", help="Path to news CSV")
    parser.add_argument("--start-date", type=str, default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--save-dir", type=str, default="data/processed", help="Directory to save numpy arrays")
    parser.add_argument("--no-sentiment", action="store_true", help="Flag to disable FinBERT sentiment extraction")
    args = parser.parse_args()
    
    if os.path.exists(args.prices) and os.path.exists(args.news):
        prepare_data(
            prices_path=args.prices,
            news_path=args.news,
            start_date=args.start_date,
            end_date=args.end_date,
            save_dir=args.save_dir,
            include_sentiment=not args.no_sentiment
        )
    else:
        print(f"Data files not found. Ensure {args.prices} and {args.news} exist.")

