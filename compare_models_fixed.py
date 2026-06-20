#!/usr/bin/env python3
import subprocess
import os
import json
import base64
import csv
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
        
    return "".join(output_lines)

def extract_metrics(output_text):
    """Extract all metrics from the training script output."""
    m = {
        "mae": "N/A", "rmse": "N/A", "r2": "N/A", "mape": "N/A", 
        "ic": "N/A", "icir": "N/A", "sharpe": "N/A"
    }
    for line in output_text.split('\n'):
        if "Final Test MAE" in line:
            m["mae"] = line.split(":")[-1].strip().replace("$", "")
        elif "Final Test RMSE" in line:
            m["rmse"] = line.split(":")[-1].strip().replace("$", "")
        elif "Final Test R2 Score" in line:
            m["r2"] = line.split(":")[-1].strip()
        elif "Final Test MAPE" in line:
            m["mape"] = line.split(":")[-1].strip().replace("%", "")
        elif "Final Test IC:" in line:
            m["ic"] = line.split(":")[-1].strip()
        elif "Final Test ICIR" in line:
            m["icir"] = line.split(":")[-1].strip()
        elif "Final Test Sharpe Ratio" in line:
            m["sharpe"] = line.split(":")[-1].strip()
    return m

def img_to_base64(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return "data:image/png;base64," + base64.b64encode(f.read()).decode()
    return ""

def calc_imp(nv, sv, metric):
    try:
        nv, sv = float(nv), float(sv)
        if nv == 0: return 0.0
        return (nv - sv) / nv * 100 if metric in ["mae","rmse","mape"] else (sv - nv) / abs(nv) * 100
    except:
        return None

def imp_cell(imp, val_str):
    if imp is None:
        return f'<td class="neutral">{val_str}</td>'
    color = "positive" if imp > 0.5 else ("negative" if imp < -0.5 else "neutral")
    arrow = "▲" if imp > 0.5 else ("▼" if imp < -0.5 else "–")
    return f'<td class="{color}">{val_str}<span class="badge">{arrow} {abs(imp):.1f}%</span></td>'

if __name__ == "__main__":
    START_DATE = "2020-01-01"
    END_DATE = "2025-12-31"
    TRIALS = "0" 
    SEEDS = ["42", "1234", "7100"]

    FIXED_D_MODEL = "64"
    FIXED_NHEAD = "4"
    FIXED_NUM_LAYERS = "1"
    FIXED_DROPOUT = "0.15"
    FIXED_LR = "0.0003"
    FIXED_BATCH_SIZE = "16"

    stocks_config = [
        {"symbol": "JPM", "prices_csv": "data/prices/JPM.csv", "news_csv": "data/news_investing.com/jpm_news_augmented.csv", "sentiment_model": "ProsusAI/finbert"},
        {"symbol": "BAC", "prices_csv": "data/prices/BAC.csv", "news_csv": "data/news_investing.com/bac_news_augmented.csv", "sentiment_model": "ProsusAI/finbert"},
        {"symbol": "COMI", "prices_csv": "data/prices/COMI_CA.csv", "news_csv": "data/news10/COMI_mubasher_augmented.csv", "sentiment_model": "CAMeL-Lab/bert-base-arabic-camelbert-msa-sentiment"},
        {"symbol": "CIEB", "prices_csv": "data/prices/CIEB_CA.csv", "news_csv": "data/news10/CIEB_mubasher_augmented.csv", "sentiment_model": "CAMeL-Lab/bert-base-arabic-camelbert-msa-sentiment"}
    ]

    results = []
    metrics_to_show = ["mae", "rmse", "r2", "mape", "ic", "icir", "sharpe"]

    os.makedirs("outputs/models_fixed", exist_ok=True)

    for config in stocks_config:
        symbol = config["symbol"]
        sentiment_model_name = config["sentiment_model"]
        print("\n" + "#" * 80)
        print(f"PROCESSING {symbol} with Sentiment Model {sentiment_model_name}")
        print("#" * 80)

        dir_sentiment    = f"data/processed_with_sentiment_{symbol}"
        dir_no_sentiment = f"data/processed_no_sentiment_{symbol}"

        # Step 1: Data Preparation WITH Sentiment (Only needed once per stock)
        print("\n" + "=" * 60)
        print(f"STEP 1: Preparing Data WITH Sentiment - {symbol}")
        print("=" * 60)
        run_command([
            ".venv/bin/python3", "dataset_preparation.py",
            "--prices", config["prices_csv"], "--news", config["news_csv"],
            "--start-date", START_DATE, "--end-date", END_DATE,
            "--save-dir", dir_sentiment, "--sentiment-model", sentiment_model_name
        ])

        # Step 2: Data Preparation WITHOUT Sentiment (Only needed once per stock)
        print("\n" + "=" * 60)
        print(f"STEP 2: Preparing Data WITHOUT Sentiment - {symbol}")
        print("=" * 60)
        run_command([
            ".venv/bin/python3", "dataset_preparation.py",
            "--prices", config["prices_csv"], "--news", config["news_csv"],
            "--start-date", START_DATE, "--end-date", END_DATE,
            "--save-dir", dir_no_sentiment, "--no-sentiment"
        ])

        # Initialize lists to store metrics across seeds
        stock_sent_metrics = {m: [] for m in metrics_to_show}
        stock_no_sent_metrics = {m: [] for m in metrics_to_show}

        for seed in SEEDS:
            print("\n" + "*" * 60)
            print(f"--- RUNNING SEED {seed} FOR {symbol} ---")
            print("*" * 60)

            model_sentiment  = f"outputs/models_fixed/best_transformer_with_sentiment_{symbol}_{seed}.pth"
            model_no_sentiment = f"outputs/models_fixed/best_transformer_no_sentiment_{symbol}_{seed}.pth"
            
            # Step 3: Train WITH Sentiment
            out_sentiment = run_command([
                ".venv/bin/python3", "train_automl.py",
                "--trials", TRIALS, "--data-dir", dir_sentiment,
                "--save-model", model_sentiment,
                "--plot-prefix", f"outputs/models_fixed/{symbol}_sentiment_{seed}_",
                "--seed", seed, "--d_model", FIXED_D_MODEL, "--nhead", FIXED_NHEAD,
                "--num_layers", FIXED_NUM_LAYERS, "--dropout", FIXED_DROPOUT,
                "--lr", FIXED_LR, "--batch_size", FIXED_BATCH_SIZE
            ])
            with open(f"outputs/models_fixed/{symbol}_sentiment_{seed}_terminal_output.txt", "w") as f:
                f.write(out_sentiment)
            m_sent = extract_metrics(out_sentiment)
            for m in metrics_to_show:
                if m_sent[m] != "N/A": stock_sent_metrics[m].append(float(m_sent[m]))

            # Step 4: Train WITHOUT Sentiment
            out_no_sentiment = run_command([
                ".venv/bin/python3", "train_automl.py",
                "--trials", TRIALS, "--data-dir", dir_no_sentiment,
                "--save-model", model_no_sentiment,
                "--plot-prefix", f"outputs/models_fixed/{symbol}_no_sentiment_{seed}_",
                "--seed", seed, "--d_model", FIXED_D_MODEL, "--nhead", FIXED_NHEAD,
                "--num_layers", FIXED_NUM_LAYERS, "--dropout", FIXED_DROPOUT,
                "--lr", FIXED_LR, "--batch_size", FIXED_BATCH_SIZE
            ])
            with open(f"outputs/models_fixed/{symbol}_no_sentiment_{seed}_terminal_output.txt", "w") as f:
                f.write(out_no_sentiment)
            m_no_sent = extract_metrics(out_no_sentiment)
            for m in metrics_to_show:
                if m_no_sent[m] != "N/A": stock_no_sent_metrics[m].append(float(m_no_sent[m]))

            # Combine predictions plot for THIS SEED
            try:
                sent_csv = f"outputs/models_fixed/{symbol}_sentiment_{seed}_true_vs_pred.csv"
                no_sent_csv = f"outputs/models_fixed/{symbol}_no_sentiment_{seed}_true_vs_pred.csv"
                
                df_sent = pd.read_csv(sent_csv)
                df_no_sent = pd.read_csv(no_sent_csv)
                
                plt.figure(figsize=(14, 6))
                plt.plot(df_sent['True Close Price'], label='True Close Price', color='blue', alpha=0.6, linewidth=2)
                plt.plot(df_no_sent['Predicted Close Price'], label='Pred (No Sentiment)', color='orange', alpha=0.8, linestyle='--')
                plt.plot(df_sent['Predicted Close Price'], label='Pred (With Sentiment)', color='green', alpha=0.8, linestyle='--')
                plt.title(f'{symbol} (Seed {seed}): True vs Predicted Prices')
                plt.xlabel('Time Steps (Days)')
                plt.ylabel('Price')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(f"outputs/models_fixed/{symbol}_combined_{seed}_true_vs_pred.png")
                plt.close()
            except Exception as e:
                print(f"Could not create combined plot for {symbol} (seed {seed}): {e}")

        # Compute Aggregates
        avg_sent = {}
        avg_no_sent = {}
        for m in metrics_to_show:
            if stock_sent_metrics[m]:
                mean_val = np.mean(stock_sent_metrics[m])
                std_val = np.std(stock_sent_metrics[m])
                avg_sent[m] = f"{mean_val:.4f} ± {std_val:.4f}"
                avg_sent[f"{m}_mean"] = mean_val
            else:
                avg_sent[m] = "N/A"
                avg_sent[f"{m}_mean"] = None
                
            if stock_no_sent_metrics[m]:
                mean_val = np.mean(stock_no_sent_metrics[m])
                std_val = np.std(stock_no_sent_metrics[m])
                avg_no_sent[m] = f"{mean_val:.4f} ± {std_val:.4f}"
                avg_no_sent[f"{m}_mean"] = mean_val
            else:
                avg_no_sent[m] = "N/A"
                avg_no_sent[f"{m}_mean"] = None

        results.append({
            "Symbol": symbol,
            "Model": sentiment_model_name,
            "No_Sent": avg_no_sent,
            "With_Sent": avg_sent
        })

    # Step 5: Final Comparison Table
    print("\n" + "=" * 120)
    print("FINAL COMPARISON RESULTS (AVERAGED OVER SEEDS)")
    print("=" * 120)
    
    print(f"{'Stock':<8} | {'Metric':<8} | {'No Sentiment':<25} | {'With Sentiment':<25} | {'Mean Improvement'}")
    print("-" * 100)
    for r in results:
        for m in metrics_to_show:
            no_val = r["No_Sent"][m]
            si_val = r["With_Sent"][m]
            
            nv_mean = r["No_Sent"].get(f"{m}_mean")
            sv_mean = r["With_Sent"].get(f"{m}_mean")
            
            if nv_mean is not None and sv_mean is not None:
                imp = calc_imp(nv_mean, sv_mean, m)
                imp_str = f"{imp:+.2f}%" if imp is not None else "N/A"
            else:
                imp_str = "N/A"
                
            print(f"{r['Symbol']:<8} | {m.upper():<8} | {no_val:<25} | {si_val:<25} | {imp_str}")
        print("-" * 100)
    
    # Save CSV
    csv_file = "outputs/models_fixed/comparison_results_averaged.csv"
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Symbol", "Sentiment_Model", "Metric", "No_Sentiment (Mean±Std)", "With_Sentiment (Mean±Std)", "Mean_Improvement"])
        for r in results:
            for m in metrics_to_show:
                nv_mean = r["No_Sent"].get(f"{m}_mean")
                sv_mean = r["With_Sent"].get(f"{m}_mean")
                imp = calc_imp(nv_mean, sv_mean, m) if (nv_mean is not None and sv_mean is not None) else None
                imp_str = f"{imp:.4f}%" if imp is not None else "N/A"
                writer.writerow([r["Symbol"], r["Model"], m.upper(), r["No_Sent"][m], r["With_Sent"][m], imp_str])
    
    # HTML Report
    metric_labels = {"mae": "MAE ($)", "rmse": "RMSE ($)", "r2": "R² Score", "mape": "MAPE (%)", "ic": "IC", "icir": "ICIR", "sharpe": "Sharpe Ratio"}
    metrics_rows_html = ""
    for r in results:
        symbol = r["Symbol"]
        for i, m in enumerate(metrics_to_show):
            no_val = r["No_Sent"][m]
            si_val = r["With_Sent"][m]
            nv_mean = r["No_Sent"].get(f"{m}_mean")
            sv_mean = r["With_Sent"].get(f"{m}_mean")
            imp = calc_imp(nv_mean, sv_mean, m) if (nv_mean is not None and sv_mean is not None) else None
            
            row_class = "alt-row" if i % 2 == 0 else ""
            sym_cell = f'<td class="symbol-cell" rowspan="{len(metrics_to_show)}">{symbol}</td>' if i == 0 else ""
            metrics_rows_html += f"""
            <tr class="{row_class}">
                {sym_cell}
                <td><strong>{metric_labels.get(m, m.upper())}</strong></td>
                <td>{no_val}</td>
                {imp_cell(imp, si_val)}
            </tr>"""
        metrics_rows_html += '<tr class="spacer"><td colspan="4"></td></tr>'

    charts_html = ""
    for r in results:
        symbol = r["Symbol"]
        # Use the first seed's chart as representative
        rep_seed = SEEDS[0]
        img_path = f"outputs/models_fixed/{symbol}_combined_{rep_seed}_true_vs_pred.png"
        b64 = img_to_base64(img_path)
        if b64:
            charts_html += f"""
            <div class="chart-card">
                <h3>{symbol} — True vs Predicted (Representative Plot: Seed {rep_seed})</h3>
                <img src="{b64}" alt="{symbol} combined plot">
            </div>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Averaged Stock Sentiment Report</title>
<style>
  :root {{ --bg: #0f1117; --card: #1a1d27; --border: #2a2d3e; --text: #e2e8f0; --dim: #8892a4; --accent: #6366f1; --positive: #22c55e; --negative: #ef4444; --neutral: #f59e0b; }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text); font-family: 'Segoe UI', system-ui, sans-serif; padding: 32px; }}
  h1 {{ font-size: 2rem; font-weight: 700; text-align: center; margin-bottom: 6px; background: linear-gradient(90deg, #6366f1, #a855f7); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
  .subtitle {{ text-align: center; color: var(--dim); margin-bottom: 40px; font-size: 0.95rem; }}
  .section {{ background: var(--card); border: 1px solid var(--border); border-radius: 16px; padding: 28px; margin-bottom: 32px; }}
  .section h2 {{ font-size: 1.25rem; font-weight: 600; margin-bottom: 20px; padding-bottom: 12px; border-bottom: 1px solid var(--border); color: var(--accent); }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
  th {{ background: #23263a; color: var(--dim); font-weight: 600; text-transform: uppercase; font-size: 0.75rem; letter-spacing: 0.05em; padding: 12px 16px; text-align: left; }}
  td {{ padding: 11px 16px; border-bottom: 1px solid var(--border); vertical-align: middle; }}
  tr.alt-row td {{ background: rgba(255,255,255,0.02); }}
  tr.spacer td {{ padding: 4px; border: none; }}
  td.symbol-cell {{ font-weight: 700; font-size: 1rem; color: var(--accent); border-right: 1px solid var(--border); vertical-align: top; padding-top: 14px; }}
  td.positive {{ color: var(--positive); }} td.negative {{ color: var(--negative); }} td.neutral  {{ color: var(--text); }}
  .badge {{ display: inline-block; margin-left: 8px; font-size: 0.72rem; opacity: 0.8; font-weight: 600; }}
  .charts-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(600px, 1fr)); gap: 24px; }}
  .chart-card {{ background: var(--card); border: 1px solid var(--border); border-radius: 16px; padding: 20px; }}
  .chart-card h3 {{ font-size: 1rem; margin-bottom: 14px; color: var(--dim); }}
  .chart-card img {{ width: 100%; border-radius: 8px; }}
  .meta {{ text-align: center; color: var(--dim); font-size: 0.8rem; margin-top: 32px; }}
</style>
</head>
<body>
<h1>📈 Stock Prediction: Averaged Seed Report</h1>
<p class="subtitle">Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} &nbsp;|&nbsp; Seeds: {", ".join(SEEDS)} &nbsp;|&nbsp; Fixed HParams</p>

<div class="section">
  <h2>📊 Metrics Comparison (Mean ± Std)</h2>
  <table>
    <thead><tr>
      <th>Stock</th><th>Metric</th>
      <th>Without Sentiment</th>
      <th>With Sentiment ↔ Improvement (on Mean)</th>
    </tr></thead>
    <tbody>{metrics_rows_html}</tbody>
  </table>
</div>

<div class="section">
  <h2>📉 Representative Prediction Charts</h2>
  <div class="charts-grid">{charts_html}</div>
</div>

<p class="meta">Stock Sentiment Analysis Research · ASU · 2020–2025</p>
</body>
</html>"""

    html_file = "outputs/models_fixed/report_averaged.html"
    with open(html_file, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"[DONE] HTML report saved to: {html_file}")
    print("=" * 120)
