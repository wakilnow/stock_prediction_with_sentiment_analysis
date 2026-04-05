#!/usr/bin/env python3
import subprocess
import os

def run_command(cmd_args):
    """Run a shell command and stream its output."""
    print(f"\n>> Running: {' '.join(cmd_args)}")
    process = subprocess.Popen(cmd_args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    output_lines = []
    for line in iter(process.stdout.readline, ''):
        print(line, end='')
        output_lines.append(line)
        
    process.stdout.close()
    return_code = process.wait()
    if return_code != 0:
        print(f"Command failed with exit code: {return_code}")
        # We don't exit to allow the non-sentiment version to run if sentiment fails, but typically we should.
        
    return "".join(output_lines)

def extract_metrics(output_text):
    """Extract final MAE and RMSE from the training script output."""
    mae = None
    rmse = None
    for line in output_text.split('\n'):
        if "Final Test MAE" in line:
            mae = line.split(":")[-1].strip()
        elif "Final Test RMSE" in line:
            rmse = line.split(":")[-1].strip()
    return mae, rmse

if __name__ == "__main__":
    # Base configuration
    PRICES_CSV = "data/prices/JPM.csv"
    NEWS_CSV = "data/news_investing.com/jpm_news.csv"
    START_DATE = "2020-01-01"
    END_DATE = "2025-12-31"
    TRIALS = "5" # Optuna trials per model

    # Paths
    DIR_SENTIMENT    = "data/processed_with_sentiment"
    MODEL_SENTIMENT  = "models/best_transformer_with_sentiment.pth"
    
    DIR_NO_SENTIMENT = "data/processed_no_sentiment"
    MODEL_NO_SENTIMENT = "models/best_transformer_no_sentiment.pth"

    # Step 1: Data Preparation WITH Sentiment
    print("=" * 60)
    print("STEP 1: Preparing Data WITH Sentiment")
    print("=" * 60)
    run_command([
        ".venv/bin/python3", "dataset_preparation.py",
        "--prices", PRICES_CSV,
        "--news", NEWS_CSV,
        "--start-date", START_DATE,
        "--end-date", END_DATE,
        "--save-dir", DIR_SENTIMENT
    ])

    # Step 2: Data Preparation WITHOUT Sentiment
    print("\n" + "=" * 60)
    print("STEP 2: Preparing Data WITHOUT Sentiment")
    print("=" * 60)
    run_command([
        ".venv/bin/python3", "dataset_preparation.py",
        "--prices", PRICES_CSV,
        "--news", NEWS_CSV,
        "--start-date", START_DATE,
        "--end-date", END_DATE,
        "--save-dir", DIR_NO_SENTIMENT,
        "--no-sentiment"
    ])

    # Step 3: Train Model WITH Sentiment
    print("\n" + "=" * 60)
    print("STEP 3: Training Model WITH Sentiment")
    print("=" * 60)
    out_sentiment = run_command([
        ".venv/bin/python3", "train_automl.py",
        "--trials", TRIALS,
        "--data-dir", DIR_SENTIMENT,
        "--save-model", MODEL_SENTIMENT
    ])
    mae_sent, rmse_sent = extract_metrics(out_sentiment)

    # Step 4: Train Model WITHOUT Sentiment
    print("\n" + "=" * 60)
    print("STEP 4: Training Model WITHOUT Sentiment")
    print("=" * 60)
    out_no_sentiment = run_command([
        ".venv/bin/python3", "train_automl.py",
        "--trials", TRIALS,
        "--data-dir", DIR_NO_SENTIMENT,
        "--save-model", MODEL_NO_SENTIMENT
    ])
    mae_no_sent, rmse_no_sent = extract_metrics(out_no_sentiment)

    # Step 5: Final Comparison
    print("\n" + "=" * 60)
    print("FINAL COMPARISON: JPM (2020 - 2025)")
    print("=" * 60)
    print(f"{'Metric':<10} | {'No Sentiment':<20} | {'With Sentiment':<20}")
    print("-" * 55)
    print(f"{'MAE':<10} | {str(mae_no_sent):<20} | {str(mae_sent):<20}")
    print(f"{'RMSE':<10} | {str(rmse_no_sent):<20} | {str(rmse_sent):<20}")
    print("=" * 60)
