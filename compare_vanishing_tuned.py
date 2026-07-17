import os
import subprocess
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import base64
import csv
from datetime import datetime
from itertools import groupby
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from dataset_preparation import prepare_data

# --- 1. CONFIGURATION ---
DECAY_RATE = 0.90  # Default decay rate for the Vanishing Sentiment scenario
SEEDS = [7100, 15732, 29530, 100,200,300,400]
EPOCHS = 100
BATCH_SIZE = 32

situations = {
    "No_Sentiment": {"include_sentiment": False, "decay_rate": 0.0},
    "Standard_Sentiment": {"include_sentiment": True, "decay_rate": 0.0},
    "Vanishing_Sentiment": {"include_sentiment": True, "decay_rate": DECAY_RATE}
}

stocks_config = [
    {"symbol": "JPM", "prices": "data/prices/JPM.csv", "news": "data/news_investing.com/jpm_news.csv", "sentiment_model": "ProsusAI/finbert"},
    {"symbol": "BAC", "prices": "data/prices/BAC.csv", "news": "data/news_investing.com/bac_news.csv", "sentiment_model": "ProsusAI/finbert"},
    {"symbol": "COMI", "prices": "data/prices/COMI_CA.csv", "news": "data/news10/COMI_mubasher.csv", "sentiment_model": "CAMeL-Lab/bert-base-arabic-camelbert-msa-sentiment"},
    {"symbol": "CIEB", "prices": "data/prices/CIEB_CA.csv", "news": "data/news10/CIEB_mubasher.csv", "sentiment_model": "CAMeL-Lab/bert-base-arabic-camelbert-msa-sentiment"}
]

out_dir = "outputs/vanishing_tuned"
os.makedirs(out_dir, exist_ok=True)

def run_command(cmd):
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return result.stdout

def extract_metrics(output):
    m = {"mae": "N/A", "rmse": "N/A", "r2": "N/A", "mape": "N/A", "ic": "N/A", "icir": "N/A", "sharpe": "N/A"}
    for line in output.split('\n'):
        if "Final Test MAE" in line: m["mae"] = line.split(":")[-1].strip().replace("$", "")
        elif "Final Test RMSE" in line: m["rmse"] = line.split(":")[-1].strip().replace("$", "")
        elif "Final Test R2 Score" in line: m["r2"] = line.split(":")[-1].strip()
        elif "Final Test MAPE" in line: m["mape"] = line.split(":")[-1].strip().replace("%", "")
        elif "Final Test IC:" in line: m["ic"] = line.split(":")[-1].strip()
        elif "Final Test ICIR" in line: m["icir"] = line.split(":")[-1].strip()
        elif "Final Test Sharpe Ratio" in line: m["sharpe"] = line.split(":")[-1].strip()
    return {k: float(v) if v != "N/A" else "N/A" for k, v in m.items()}

def compute_switch_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
    
    if len(y_true) > 1:
        actual_returns = np.diff(y_true) / (y_true[:-1] + 1e-10)
        pred_returns = (y_pred[1:] - y_true[:-1]) / (y_true[:-1] + 1e-10)
        
        ic = np.corrcoef(pred_returns, actual_returns)[0, 1]
        if np.isnan(ic): ic = 0.0
        
        window = 20
        if len(actual_returns) > window:
            rolling_ics = []
            for i in range(len(actual_returns) - window + 1):
                w_ic = np.corrcoef(pred_returns[i:i+window], actual_returns[i:i+window])[0, 1]
                rolling_ics.append(w_ic)
            rolling_ics = np.array(rolling_ics)
            rolling_ics = rolling_ics[~np.isnan(rolling_ics)]
            icir = np.mean(rolling_ics) / (np.std(rolling_ics) + 1e-10) if len(rolling_ics) > 0 else 0.0
        else:
            icir = 0.0
            
        strat_returns = np.where(pred_returns > 0, actual_returns, 0)
        sharpe = (np.mean(strat_returns) / (np.std(strat_returns) + 1e-10)) * np.sqrt(252)
    else:
        ic, icir, sharpe = 0.0, 0.0, 0.0
        
    return {
        "mae": mae, "rmse": rmse, "r2": r2, "mape": mape,
        "ic": ic, "icir": icir, "sharpe": sharpe
    }

# --- 2. TRAIN FUNCTION ---
def run_training(symbol, sit_name, seed):
    processed_dir = f"data/processed_{symbol}_{sit_name}"
    model_path = f"{out_dir}/model_{symbol}_{sit_name}_{seed}.pth"
    plot_prefix = f"{out_dir}/{symbol}_{sit_name}_{seed}_"
    term_file = f"{out_dir}/{symbol}_{sit_name}_{seed}_term.txt"
    
    if os.path.exists(term_file):
        print(f"      [Skipping training, found cached {term_file}]")
        with open(term_file, "r") as f:
            out = f.read()
    else:
        out = run_command([
            ".venv/bin/python3", "train_automl.py",
            "--trials", "20", "--data-dir", processed_dir,
            "--save-model", model_path,
            "--plot-prefix", plot_prefix,
            "--seed", str(seed), "--epochs", "100"
        ])
        with open(term_file, "w") as f:
            f.write(out)
        
    return extract_metrics(out)

# --- 3. HELPER FUNCTIONS ---
def calc_imp(baseline, current, metric):
    if baseline is None or current is None: return None
    imp = ((baseline - current) / baseline) * 100
    if metric in ['r2', 'ic', 'icir', 'sharpe']:
        imp = -imp  # Higher is better
    return imp

def img_to_base64(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return "data:image/png;base64," + base64.b64encode(f.read()).decode()
    return ""

def format_improvement(imp_str):
    if imp_str == "N/A" or pd.isna(imp_str):
        return '<span class="neutral badge">–</span>'
    try:
        imp = float(imp_str.replace('%', ''))
        color = "positive" if imp > 0.5 else ("negative" if imp < -0.5 else "neutral")
        arrow = "▲" if imp > 0.5 else ("▼" if imp < -0.5 else "–")
        return f'<span class="{color} badge">{arrow} {abs(imp):.1f}%</span>'
    except:
        return '<span class="neutral badge">–</span>'

def plot_true_vs_preds(stock, true_vals, preds_dict, out_path, seed):
    plt.figure(figsize=(14, 7))
    plt.style.use('dark_background')
    
    colors = {
        'No_Sentiment': '#a855f7',       # purple
        'Standard_Sentiment': '#3b82f6', # blue
        'Vanishing_Sentiment': '#22c55e',# green
        'Switching_Ensemble': '#f59e0b'  # amber
    }
    
    plt.plot(range(len(true_vals)), true_vals, color='white', label='True Price', linewidth=2.5, alpha=0.9)
    
    for sit, preds in preds_dict.items():
        plt.plot(range(len(preds)), preds, color=colors.get(sit, '#ef4444'), label=f'Pred ({sit})', linewidth=1.5, alpha=0.8)

    plt.title(f"{stock} Price Prediction — Vanishing Sentiment Comparison (Seed {seed})", fontsize=16, pad=15)
    plt.xlabel('Time Step (Test Set)', fontsize=12)
    plt.ylabel('Stock Price', fontsize=12)
    plt.legend(loc='best', fontsize=11)
    plt.grid(True, alpha=0.2, color='gray', linestyle='--')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#1a1d27')
    plt.close()

# --- 4. MAIN PIPELINE ---
if __name__ == "__main__":
    results = []
    metrics_to_show = ["mae", "rmse", "r2", "mape", "ic", "icir", "sharpe"]

    for stock_info in stocks_config:
        symbol = stock_info["symbol"]
        print(f"\n{'='*80}\nPROCESSING STOCK: {symbol}\n{'='*80}")
        
        # Dictionary to store predictions for the combined plot (we'll save the first seed)
        first_seed_preds = {}
        first_seed_true = None

        stock_results = {}
        for sit_name in situations.keys():
            stock_results[sit_name] = {m: [] for m in metrics_to_show}

        for sit_name, config in situations.items():
            print(f"\n--- Situation: {sit_name} ---")
            processed_dir = f"data/processed_{symbol}_{sit_name}"
            
            print(f"Preparing dataset for {sit_name}...")
            prepare_data(
                prices_path=stock_info["prices"],
                news_path=stock_info["news"],
                start_date="2020-01-01",
                end_date="2025-12-31",
                save_dir=processed_dir,
                include_sentiment=config["include_sentiment"],
                sentiment_model=stock_info["sentiment_model"],
                decay_rate=config["decay_rate"]
            )
            
            for seed in SEEDS:
                print(f"  -> Running Training (Seed {seed})...")
                metrics = run_training(symbol, sit_name, seed)
                for m in metrics_to_show:
                    stock_results[sit_name][m].append(metrics[m])
                    
                # Save predictions for charting if it's the first seed
                if seed == SEEDS[0]:
                    csv_path = f"{out_dir}/{symbol}_{sit_name}_{seed}_true_vs_pred.csv"
                    if os.path.exists(csv_path):
                        df = pd.read_csv(csv_path)
                        first_seed_preds[sit_name] = df['Predicted Close Price'].values
                        if first_seed_true is None:
                            first_seed_true = df['True Close Price'].values

        # Compute Switching Ensemble Metrics
        sit_name = "Switching_Ensemble"
        print(f"\n--- Situation: {sit_name} ---")
        stock_results[sit_name] = {m: [] for m in metrics_to_show}
        
        # Load X_test from Standard_Sentiment to get the news mask
        x_test_path = f"data/processed_{symbol}_Standard_Sentiment/X_test.npy"
        if os.path.exists(x_test_path):
            X_test = np.load(x_test_path)
            # Sentiment score is at index 1 of the features. Check the last day of the window (index -1)
            news_mask = (X_test[:, -1, 1] != 0.0)
            
            for seed in SEEDS:
                no_sent_csv = f"{out_dir}/{symbol}_No_Sentiment_{seed}_true_vs_pred.csv"
                std_sent_csv = f"{out_dir}/{symbol}_Standard_Sentiment_{seed}_true_vs_pred.csv"
                
                if os.path.exists(no_sent_csv) and os.path.exists(std_sent_csv):
                    df_no = pd.read_csv(no_sent_csv)
                    df_std = pd.read_csv(std_sent_csv)
                    
                    preds_no = df_no['Predicted Close Price'].values
                    preds_std = df_std['Predicted Close Price'].values
                    true_vals = df_no['True Close Price'].values
                    
                    # Routing logic: If news_mask is True, use Standard_Sentiment. Else use No_Sentiment.
                    preds_switch = np.where(news_mask, preds_std, preds_no)
                    
                    # Compute metrics
                    switch_metrics = compute_switch_metrics(true_vals, preds_switch)
                    for m in metrics_to_show:
                        stock_results[sit_name][m].append(switch_metrics[m])
                        
                    if seed == SEEDS[0]:
                        first_seed_preds[sit_name] = preds_switch
        
        # Aggregate Results for this stock
        avg_results = {}
        for sit in list(situations.keys()) + ["Switching_Ensemble"]:
            avg_results[sit] = {}
            for m in metrics_to_show:
                if len(stock_results[sit][m]) > 0:
                    vals = stock_results[sit][m]
                    avg_results[sit][m] = f"{np.mean(vals):.4f} ± {np.std(vals):.4f}"
                    avg_results[sit][f"{m}_mean"] = np.mean(vals)
                else:
                    avg_results[sit][m] = "N/A"
                    avg_results[sit][f"{m}_mean"] = None
                
        results.append({
            "Symbol": symbol,
            "Metrics": avg_results
        })
        
        # Generate combined chart for this stock
        if first_seed_true is not None:
            chart_path = f"{out_dir}/{symbol}_all_situations.png"
            plot_true_vs_preds(symbol, first_seed_true, first_seed_preds, chart_path, SEEDS[0])
            print(f"Saved combined chart to {chart_path}")

    # --- 5. TERMINAL REPORT ---
    print("\n" + "=" * 120)
    print("FINAL COMPARISON RESULTS (AVERAGED OVER SEEDS)")
    print("=" * 120)
    for r in results:
        print(f"\n--- {r['Symbol']} ---")
        for m in metrics_to_show:
            no_sent = r["Metrics"]["No_Sentiment"][m]
            std_sent = r["Metrics"]["Standard_Sentiment"][m]
            van_sent = r["Metrics"]["Vanishing_Sentiment"][m]
            sw_sent = r["Metrics"]["Switching_Ensemble"][m]
            print(f"{m.upper():<8} | Baseline: {no_sent:<18} | Std: {std_sent:<18} | Vanishing: {van_sent:<18} | Switching: {sw_sent:<18}")

    # --- 6. HTML REPORT GENERATION ---
    html_rows = ""
    for r in results:
        symbol = r["Symbol"]
        html_rows += f'<tr class="stock-header"><td colspan="5">{symbol}</td></tr>'
        for metric in metrics_to_show:
            row_html = f'<tr><td class="metric-cell">{metric.upper()}</td>'
            
            # Baseline (No Sentiment)
            baseline_val_str = r["Metrics"]["No_Sentiment"][metric]
            baseline_mean = r["Metrics"]["No_Sentiment"].get(f"{metric}_mean")
            baseline_display = baseline_val_str.split(" ± ")[0] if baseline_val_str != "N/A" else "-"
            row_html += f'<td class="baseline-cell">{baseline_display}</td>'
            
            # Helper for sentiment cells
            def build_cell(sit_name):
                val_str = r["Metrics"][sit_name][metric]
                mean_val = r["Metrics"][sit_name].get(f"{metric}_mean")
                display_val = val_str.split(" ± ")[0] if val_str != "N/A" else "-"
                imp = calc_imp(baseline_mean, mean_val, metric) if (baseline_mean is not None and mean_val is not None) else None
                imp_html = format_improvement(f"{imp:.2f}%") if imp is not None else '<span class="neutral badge">–</span>'
                return f'<td><div class="stat-group"><div>{display_val}</div>{imp_html}</div></td>'

            row_html += build_cell("Standard_Sentiment")
            row_html += build_cell("Vanishing_Sentiment")
            row_html += build_cell("Switching_Ensemble")
            
            row_html += "</tr>"
            html_rows += row_html

    charts_html = ""
    for r in results:
        symbol = r["Symbol"]
        img_path = f"{out_dir}/{symbol}_all_situations.png"
        b64 = img_to_base64(img_path)
        if b64:
            charts_html += f"""
            <div class="chart-card">
                <h3>{symbol} Price Prediction — Vanishing Decay Comparison</h3>
                <img src="{b64}" alt="{symbol} all situations plot">
            </div>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Vanishing Sentiment Comparison</title>
<style>
  :root {{ --bg: #0f1117; --card: #1a1d27; --border: #2a2d3e; --text: #e2e8f0; --dim: #8892a4; --accent: #6366f1; --positive: #22c55e; --negative: #ef4444; --neutral: #f59e0b; }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text); font-family: 'Segoe UI', system-ui, sans-serif; padding: 32px; }}
  h1 {{ font-size: 2rem; font-weight: 700; text-align: center; margin-bottom: 6px; background: linear-gradient(90deg, #6366f1, #a855f7); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
  .subtitle {{ text-align: center; color: var(--dim); margin-bottom: 40px; font-size: 0.95rem; }}
  .section {{ background: var(--card); border: 1px solid var(--border); border-radius: 16px; padding: 28px; margin-bottom: 32px; overflow-x: auto; }}
  .section h2 {{ font-size: 1.25rem; font-weight: 600; margin-bottom: 20px; padding-bottom: 12px; border-bottom: 1px solid var(--border); color: var(--accent); }}
  
  table {{ width: 100%; min-width: 800px; border-collapse: collapse; font-size: 0.95rem; text-align: center; }}
  th, td {{ padding: 14px 16px; border: 1px solid var(--border); vertical-align: middle; }}
  th {{ background: #23263a; color: var(--dim); font-weight: 600; text-transform: uppercase; font-size: 0.8rem; letter-spacing: 0.05em; }}
  tr.stock-header td {{ background: rgba(99, 102, 241, 0.1); color: #a855f7; font-weight: 700; font-size: 1.1rem; }}
  td.metric-cell {{ font-weight: 600; background: rgba(255,255,255,0.02); width: 120px; text-align: left; }}
  td.baseline-cell {{ font-weight: 600; color: #cbd5e1; font-size: 1.05rem; background: rgba(255,255,255,0.01); }}
  
  .stat-group {{ display: flex; flex-direction: column; gap: 4px; align-items: center; justify-content: center; }}
  .stat-group div {{ font-size: 1.05rem; font-weight: 500; }}
  
  .positive {{ color: var(--positive); }} .negative {{ color: var(--negative); }} .neutral  {{ color: var(--neutral); }}
  .badge {{ display: inline-block; font-size: 0.85rem; font-weight: 700; padding: 4px 8px; border-radius: 6px; background: rgba(0,0,0,0.2); width: fit-content; margin-top: 4px; }}
  
  .charts-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 24px; }}
  .chart-card {{ background: var(--card); border: 1px solid var(--border); border-radius: 16px; padding: 20px; }}
  .chart-card h3 {{ font-size: 1rem; margin-bottom: 14px; color: var(--dim); text-align: center; }}
  .chart-card img {{ width: 100%; border-radius: 8px; }}
</style>
</head>
<body>
<h1>📉 Vanishing Sentiment Comparison</h1>
<p class="subtitle">Decay Rate: {DECAY_RATE} | Improvements are relative to the Price-Only Baseline</p>

<div class="section">
  <h2>📊 Aggregate Metrics Comparison</h2>
  <table>
    <thead><tr>
      <th>Metric</th>
      <th>No Sentiment<br><span style="font-size:0.7rem;font-weight:normal;opacity:0.7">(Price Only Baseline)</span></th>
      <th>Standard Sentiment<br><span style="font-size:0.7rem;font-weight:normal;opacity:0.7">(Zero-Padding)</span></th>
      <th>Vanishing Sentiment<br><span style="font-size:0.7rem;font-weight:normal;opacity:0.7">(Exponential Decay α={DECAY_RATE})</span></th>
      <th>Switching Ensemble<br><span style="font-size:0.7rem;font-weight:normal;opacity:0.7">(Dynamic Routing)</span></th>
    </tr></thead>
    <tbody>{html_rows}</tbody>
  </table>
</div>

<div class="section">
  <h2>📉 Representative Overlaid Charts</h2>
  <div class="charts-grid">{charts_html}</div>
</div>
</body>
</html>"""

    report_path = f"{out_dir}/report_vanishing_tuned.html"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n[DONE] Saved comprehensive Vanishing Sentiment report to: {report_path}")
