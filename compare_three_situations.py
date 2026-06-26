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
    m = {"mae": "N/A", "rmse": "N/A", "r2": "N/A", "mape": "N/A", "ic": "N/A", "icir": "N/A", "sharpe": "N/A"}
    for line in output_text.split('\n'):
        if "Final Test MAE" in line: m["mae"] = line.split(":")[-1].strip().replace("$", "")
        elif "Final Test RMSE" in line: m["rmse"] = line.split(":")[-1].strip().replace("$", "")
        elif "Final Test R2 Score" in line: m["r2"] = line.split(":")[-1].strip()
        elif "Final Test MAPE" in line: m["mape"] = line.split(":")[-1].strip().replace("%", "")
        elif "Final Test IC:" in line: m["ic"] = line.split(":")[-1].strip()
        elif "Final Test ICIR" in line: m["icir"] = line.split(":")[-1].strip()
        elif "Final Test Sharpe Ratio" in line: m["sharpe"] = line.split(":")[-1].strip()
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
    if imp is None: return f'<td class="neutral">{val_str}</td>'
    color = "positive" if imp > 0.5 else ("negative" if imp < -0.5 else "neutral")
    arrow = "▲" if imp > 0.5 else ("▼" if imp < -0.5 else "–")
    return f'<td class="{color}">{val_str}<span class="badge">{arrow} {abs(imp):.1f}%</span></td>'

if __name__ == "__main__":
    TRIALS = "0" 
    SEEDS = ["7100", "42", "1234"]

    FIXED_D_MODEL = "64"
    FIXED_NHEAD = "4"
    FIXED_NUM_LAYERS = "1"
    FIXED_DROPOUT = "0.15"
    FIXED_LR = "0.0003"
    FIXED_BATCH_SIZE = "16"

    situations = {
        "Original": {"start": "2020-01-01", "end": "2025-12-31", "suffix": ""},
        "Augmented": {"start": "2020-01-01", "end": "2025-12-31", "suffix": "_augmented"},
        "Filtered": {"start": "2020-01-01", "end": "2025-12-31", "suffix": "_intersected"}
    }

    stocks_config = [
        {"symbol": "JPM", "prices_csv": "data/prices/JPM.csv", "news_csv": "data/news_investing.com/jpm_news.csv", "sentiment_model": "ProsusAI/finbert"},
        {"symbol": "BAC", "prices_csv": "data/prices/BAC.csv", "news_csv": "data/news_investing.com/bac_news.csv", "sentiment_model": "ProsusAI/finbert"},
        {"symbol": "COMI", "prices_csv": "data/prices/COMI_CA.csv", "news_csv": "data/news10/COMI_mubasher.csv", "sentiment_model": "CAMeL-Lab/bert-base-arabic-camelbert-msa-sentiment"},
        {"symbol": "CIEB", "prices_csv": "data/prices/CIEB_CA.csv", "news_csv": "data/news10/CIEB_mubasher.csv", "sentiment_model": "CAMeL-Lab/bert-base-arabic-camelbert-msa-sentiment"}
    ]

    metrics_to_show = ["mae", "rmse", "r2", "mape", "ic", "icir", "sharpe"]
    results = []

    out_dir = "outputs/situations"
    os.makedirs(out_dir, exist_ok=True)

    for config in stocks_config:
        symbol = config["symbol"]
        sentiment_model_name = config["sentiment_model"]

        for sit_name, sit_conf in situations.items():
            print("\n" + "#" * 80)
            print(f"PROCESSING {symbol} - {sit_name.upper()} DATA")
            print("#" * 80)

            # Resolve news path
            base_news = config["news_csv"]
            if sit_conf["suffix"]:
                news_csv = base_news.replace(".csv", f"{sit_conf['suffix']}.csv")
            else:
                news_csv = base_news

            start_date = sit_conf["start"]
            end_date = sit_conf["end"]

            dir_sentiment    = f"data/processed_{symbol}_{sit_name}_sent"
            dir_no_sentiment = f"data/processed_{symbol}_{sit_name}_nosent"

            # Prep Data
            run_command([
                ".venv/bin/python3", "dataset_preparation.py",
                "--prices", config["prices_csv"], "--news", news_csv,
                "--start-date", start_date, "--end-date", end_date,
                "--save-dir", dir_sentiment, "--sentiment-model", sentiment_model_name
            ])
            run_command([
                ".venv/bin/python3", "dataset_preparation.py",
                "--prices", config["prices_csv"], "--news", news_csv,
                "--start-date", start_date, "--end-date", end_date,
                "--save-dir", dir_no_sentiment, "--no-sentiment"
            ])

            stock_sent_metrics = {m: [] for m in metrics_to_show}
            stock_no_sent_metrics = {m: [] for m in metrics_to_show}

            for seed in SEEDS:
                print("\n" + "*" * 60)
                print(f"--- RUNNING SEED {seed} FOR {symbol} ({sit_name}) ---")
                print("*" * 60)

                model_sentiment  = f"{out_dir}/model_{symbol}_{sit_name}_sent_{seed}.pth"
                model_no_sentiment = f"{out_dir}/model_{symbol}_{sit_name}_nosent_{seed}.pth"
                
                # Train WITH Sentiment
                out_sentiment = run_command([
                    ".venv/bin/python3", "train_automl.py",
                    "--trials", TRIALS, "--data-dir", dir_sentiment,
                    "--save-model", model_sentiment,
                    "--plot-prefix", f"{out_dir}/{symbol}_{sit_name}_sent_{seed}_",
                    "--seed", seed, "--d_model", FIXED_D_MODEL, "--nhead", FIXED_NHEAD,
                    "--num_layers", FIXED_NUM_LAYERS, "--dropout", FIXED_DROPOUT,
                    "--lr", FIXED_LR, "--batch_size", FIXED_BATCH_SIZE
                ])
                with open(f"{out_dir}/{symbol}_{sit_name}_sent_{seed}_term.txt", "w") as f:
                    f.write(out_sentiment)
                m_sent = extract_metrics(out_sentiment)
                for m in metrics_to_show:
                    if m_sent[m] != "N/A": stock_sent_metrics[m].append(float(m_sent[m]))

                # Train WITHOUT Sentiment
                out_no_sentiment = run_command([
                    ".venv/bin/python3", "train_automl.py",
                    "--trials", TRIALS, "--data-dir", dir_no_sentiment,
                    "--save-model", model_no_sentiment,
                    "--plot-prefix", f"{out_dir}/{symbol}_{sit_name}_nosent_{seed}_",
                    "--seed", seed, "--d_model", FIXED_D_MODEL, "--nhead", FIXED_NHEAD,
                    "--num_layers", FIXED_NUM_LAYERS, "--dropout", FIXED_DROPOUT,
                    "--lr", FIXED_LR, "--batch_size", FIXED_BATCH_SIZE
                ])
                with open(f"{out_dir}/{symbol}_{sit_name}_nosent_{seed}_term.txt", "w") as f:
                    f.write(out_no_sentiment)
                m_no_sent = extract_metrics(out_no_sentiment)
                for m in metrics_to_show:
                    if m_no_sent[m] != "N/A": stock_no_sent_metrics[m].append(float(m_no_sent[m]))

                # Combined Plot
                try:
                    df_sent = pd.read_csv(f"{out_dir}/{symbol}_{sit_name}_sent_{seed}_true_vs_pred.csv")
                    df_no_sent = pd.read_csv(f"{out_dir}/{symbol}_{sit_name}_nosent_{seed}_true_vs_pred.csv")
                    
                    plt.figure(figsize=(14, 6))
                    plt.plot(df_sent['True Close Price'], label='True Close Price', color='blue', alpha=0.6, linewidth=2)
                    plt.plot(df_no_sent['Predicted Close Price'], label='Pred (No Sentiment)', color='orange', alpha=0.8, linestyle='--')
                    plt.plot(df_sent['Predicted Close Price'], label='Pred (With Sentiment)', color='green', alpha=0.8, linestyle='--')
                    plt.title(f'{symbol} ({sit_name} | Seed {seed}) True vs Predicted')
                    plt.xlabel('Time Steps')
                    plt.ylabel('Price')
                    plt.legend()
                    plt.grid(True)
                    plt.tight_layout()
                    plt.savefig(f"{out_dir}/{symbol}_{sit_name}_combined_{seed}.png")
                    plt.close()
                except Exception as e:
                    print(f"Could not create combined plot for {symbol} ({sit_name}): {e}")

            # Compute Aggregates
            avg_sent, avg_no_sent = {}, {}
            for m in metrics_to_show:
                if stock_sent_metrics[m]:
                    avg_sent[m] = f"{np.mean(stock_sent_metrics[m]):.4f} ± {np.std(stock_sent_metrics[m]):.4f}"
                    avg_sent[f"{m}_mean"] = np.mean(stock_sent_metrics[m])
                else:
                    avg_sent[m], avg_sent[f"{m}_mean"] = "N/A", None
                    
                if stock_no_sent_metrics[m]:
                    avg_no_sent[m] = f"{np.mean(stock_no_sent_metrics[m]):.4f} ± {np.std(stock_no_sent_metrics[m]):.4f}"
                    avg_no_sent[f"{m}_mean"] = np.mean(stock_no_sent_metrics[m])
                else:
                    avg_no_sent[m], avg_no_sent[f"{m}_mean"] = "N/A", None

            results.append({
                "Symbol": symbol,
                "Situation": sit_name,
                "Model": sentiment_model_name,
                "No_Sent": avg_no_sent,
                "With_Sent": avg_sent
            })

    # Terminal Report
    print("\n" + "=" * 140)
    print("FINAL 3-WAY COMPARISON RESULTS (AVERAGED OVER SEEDS)")
    print("=" * 140)
    print(f"{'Stock':<8} | {'Situation':<12} | {'Metric':<8} | {'No Sentiment':<25} | {'With Sentiment':<25} | {'Mean Improvement'}")
    print("-" * 120)
    for r in results:
        for m in metrics_to_show:
            nv_mean = r["No_Sent"].get(f"{m}_mean")
            sv_mean = r["With_Sent"].get(f"{m}_mean")
            imp = calc_imp(nv_mean, sv_mean, m) if (nv_mean is not None and sv_mean is not None) else None
            imp_str = f"{imp:+.2f}%" if imp is not None else "N/A"
            print(f"{r['Symbol']:<8} | {r['Situation']:<12} | {m.upper():<8} | {r['No_Sent'][m]:<25} | {r['With_Sent'][m]:<25} | {imp_str}")
        print("-" * 120)

    # Save CSV
    csv_file = f"{out_dir}/situations_comparison.csv"
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Symbol", "Situation", "Metric", "No_Sentiment", "With_Sentiment", "Mean_Improvement"])
        for r in results:
            for m in metrics_to_show:
                nv_mean = r["No_Sent"].get(f"{m}_mean")
                sv_mean = r["With_Sent"].get(f"{m}_mean")
                imp = calc_imp(nv_mean, sv_mean, m) if (nv_mean is not None and sv_mean is not None) else None
                imp_str = f"{imp:.4f}%" if imp is not None else "N/A"
                writer.writerow([r["Symbol"], r["Situation"], m.upper(), r["No_Sent"][m], r["With_Sent"][m], imp_str])

    # HTML Report
    metric_labels = {"mae": "MAE ($)", "rmse": "RMSE ($)", "r2": "R² Score", "mape": "MAPE (%)", "ic": "IC", "icir": "ICIR", "sharpe": "Sharpe Ratio"}
    metrics_rows_html = ""
    
    # Group results by symbol to create nice blocks
    from itertools import groupby
    results_sorted = sorted(results, key=lambda x: x["Symbol"])
    for symbol, group in groupby(results_sorted, key=lambda x: x["Symbol"]):
        group_list = list(group)
        sym_rendered = False
        
        for r in group_list:
            sit_name = r["Situation"]
            sit_rendered = False
            for i, m in enumerate(metrics_to_show):
                nv_mean = r["No_Sent"].get(f"{m}_mean")
                sv_mean = r["With_Sent"].get(f"{m}_mean")
                imp = calc_imp(nv_mean, sv_mean, m) if (nv_mean is not None and sv_mean is not None) else None
                
                row_class = "alt-row" if i % 2 == 0 else ""
                
                sym_cell = ""
                if not sym_rendered:
                    sym_cell = f'<td class="symbol-cell" rowspan="{len(metrics_to_show) * len(situations)}">{symbol}</td>'
                    sym_rendered = True
                    
                sit_cell = ""
                if not sit_rendered:
                    sit_cell = f'<td class="situation-cell" rowspan="{len(metrics_to_show)}">{sit_name}</td>'
                    sit_rendered = True
                    
                metrics_rows_html += f"""
                <tr class="{row_class}">
                    {sym_cell}
                    {sit_cell}
                    <td><strong>{metric_labels.get(m, m.upper())}</strong></td>
                    <td>{r['No_Sent'][m]}</td>
                    {imp_cell(imp, r['With_Sent'][m])}
                </tr>"""
            metrics_rows_html += '<tr class="spacer"><td colspan="5"></td></tr>'

    charts_html = ""
    for r in results:
        symbol = r["Symbol"]
        sit_name = r["Situation"]
        rep_seed = SEEDS[0]
        img_path = f"{out_dir}/{symbol}_{sit_name}_combined_{rep_seed}.png"
        b64 = img_to_base64(img_path)
        if b64:
            charts_html += f"""
            <div class="chart-card">
                <h3>{symbol} — {sit_name} Data (Seed {rep_seed})</h3>
                <img src="{b64}" alt="{symbol} {sit_name} plot">
            </div>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>3-Way Data Situation Comparison</title>
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
  tr.spacer td {{ padding: 2px; border: none; }}
  td.symbol-cell {{ font-weight: 800; font-size: 1.2rem; color: var(--accent); border-right: 2px solid var(--border); vertical-align: middle; text-align: center; }}
  td.situation-cell {{ font-weight: 600; color: #a855f7; border-right: 1px solid var(--border); vertical-align: top; padding-top: 14px; }}
  td.positive {{ color: var(--positive); }} td.negative {{ color: var(--negative); }} td.neutral  {{ color: var(--text); }}
  .badge {{ display: inline-block; margin-left: 8px; font-size: 0.72rem; opacity: 0.8; font-weight: 600; }}
  .charts-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 24px; }}
  .chart-card {{ background: var(--card); border: 1px solid var(--border); border-radius: 16px; padding: 20px; }}
  .chart-card h3 {{ font-size: 1rem; margin-bottom: 14px; color: var(--dim); text-align: center; }}
  .chart-card img {{ width: 100%; border-radius: 8px; }}
  .meta {{ text-align: center; color: var(--dim); font-size: 0.8rem; margin-top: 32px; }}
</style>
</head>
<body>
<h1>📈 Data Quality Comparison Report</h1>
<p class="subtitle">Situations: Original vs Augmented vs Date-Filtered &nbsp;|&nbsp; Seeds: {", ".join(SEEDS)}</p>

<div class="section">
  <h2>📊 Aggregate Metrics Comparison</h2>
  <table>
    <thead><tr>
      <th>Stock</th><th>Situation</th><th>Metric</th>
      <th>Without Sentiment</th>
      <th>With Sentiment ↔ Improvement</th>
    </tr></thead>
    <tbody>{metrics_rows_html}</tbody>
  </table>
</div>

<div class="section">
  <h2>📉 Representative Charts by Situation</h2>
  <div class="charts-grid">{charts_html}</div>
</div>
</body>
</html>"""

    html_file = f"{out_dir}/report_situations.html"
    with open(html_file, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n[DONE] Saved comprehensive 3-way report to: {html_file}")
