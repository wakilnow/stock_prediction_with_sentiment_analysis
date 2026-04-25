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
    START_DATE = "2020-01-01"
    END_DATE = "2025-12-31"
    TRIALS = "5" # Optuna trials per model

    stocks_config = [
        {
            "symbol": "JPM",
            "prices_csv": "data/prices/JPM.csv",
            "news_csv": "data/news_investing.com/jpm_news.csv",
            "sentiment_model": "ProsusAI/finbert"
        },
        {
            "symbol": "BAC",
            "prices_csv": "data/prices/BAC.csv",
            "news_csv": "data/news_l/BAC_news.csv",
            "sentiment_model": "ProsusAI/finbert"
        },
        {
            "symbol": "COMI",
            "prices_csv": "data/prices/COMI_CA.csv",
            "news_csv": "data/news10/COMI_mubasher.csv",
            "sentiment_model": "CAMeL-Lab/bert-base-arabic-camelbert-msa-sentiment"
        },
        {
            "symbol": "CIEB",
            "prices_csv": "data/prices/CIEB_CA.csv",
            "news_csv": "data/news10/CIEB_mubasher.csv",
            "sentiment_model": "CAMeL-Lab/bert-base-arabic-camelbert-msa-sentiment"
        }
    ]

    results = []

    for config in stocks_config:
        symbol = config["symbol"]
        sentiment_model_name = config["sentiment_model"]
        print("\n" + "#" * 80)
        print(f"PROCESSING {symbol} with Sentiment Model {sentiment_model_name}")
        print("#" * 80)

        # Paths
        dir_sentiment    = f"data/processed_with_sentiment_{symbol}"
        model_sentiment  = f"models/best_transformer_with_sentiment_{symbol}.pth"
        
        dir_no_sentiment = f"data/processed_no_sentiment_{symbol}"
        model_no_sentiment = f"models/best_transformer_no_sentiment_{symbol}.pth"

        # Step 1: Data Preparation WITH Sentiment
        print("\n" + "=" * 60)
        print(f"STEP 1: Preparing Data WITH Sentiment - {symbol}")
        print("=" * 60)
        run_command([
            ".venv/bin/python3", "dataset_preparation.py",
            "--prices", config["prices_csv"],
            "--news", config["news_csv"],
            "--start-date", START_DATE,
            "--end-date", END_DATE,
            "--save-dir", dir_sentiment,
            "--sentiment-model", sentiment_model_name
        ])

        # Step 2: Data Preparation WITHOUT Sentiment
        print("\n" + "=" * 60)
        print(f"STEP 2: Preparing Data WITHOUT Sentiment - {symbol}")
        print("=" * 60)
        run_command([
            ".venv/bin/python3", "dataset_preparation.py",
            "--prices", config["prices_csv"],
            "--news", config["news_csv"],
            "--start-date", START_DATE,
            "--end-date", END_DATE,
            "--save-dir", dir_no_sentiment,
            "--no-sentiment"
        ])

        # Step 3: Train Model WITH Sentiment
        print("\n" + "=" * 60)
        print(f"STEP 3: Training Model WITH Sentiment - {symbol}")
        print("=" * 60)
        out_sentiment = run_command([
            ".venv/bin/python3", "train_automl.py",
            "--trials", TRIALS,
            "--data-dir", dir_sentiment,
            "--save-model", model_sentiment,
            "--plot-prefix", f"models/{symbol}_sentiment_"
        ])
        with open(f"models/{symbol}_sentiment_terminal_output.txt", "w") as f:
            f.write(out_sentiment)
        mae_sent, rmse_sent = extract_metrics(out_sentiment)

        # Step 4: Train Model WITHOUT Sentiment
        print("\n" + "=" * 60)
        print(f"STEP 4: Training Model WITHOUT Sentiment - {symbol}")
        print("=" * 60)
        out_no_sentiment = run_command([
            ".venv/bin/python3", "train_automl.py",
            "--trials", TRIALS,
            "--data-dir", dir_no_sentiment,
            "--save-model", model_no_sentiment,
            "--plot-prefix", f"models/{symbol}_no_sentiment_"
        ])
        with open(f"models/{symbol}_no_sentiment_terminal_output.txt", "w") as f:
            f.write(out_no_sentiment)
        mae_no_sent, rmse_no_sent = extract_metrics(out_no_sentiment)

        results.append({
            "Symbol": symbol,
            "Model": sentiment_model_name,
            "MAE_No_Sent": mae_no_sent,
            "MAE_With_Sent": mae_sent,
            "RMSE_No_Sent": rmse_no_sent,
            "RMSE_With_Sent": rmse_sent
        })

    # Step 5: Final Comparison Table
    print("\n" + "=" * 100)
    print("FINAL COMPARISON RESULTS (2020 - 2025)")
    print("=" * 100)
    print(f"{'Stock':<8} | {'Sentiment Model':<55} | {'No Sent MAE':<12} | {'Sent MAE':<12}")
    print("-" * 100)
    for r in results:
        print(f"{r['Symbol']:<8} | {r['Model']:<55} | {str(r['MAE_No_Sent']):<12} | {str(r['MAE_With_Sent']):<12}")
    
    print("\n" + "-" * 100)
    print(f"{'Stock':<8} | {'Sentiment Model':<55} | {'No Sent RMSE':<12} | {'Sent RMSE':<12}")
    print("-" * 100)
    for r in results:
        print(f"{r['Symbol']:<8} | {r['Model']:<55} | {str(r['RMSE_No_Sent']):<12} | {str(r['RMSE_With_Sent']):<12}")
    print("=" * 100)
