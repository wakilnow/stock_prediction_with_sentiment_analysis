import os
import subprocess
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import base64
import csv
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from dataset_preparation import prepare_data

# --- 1. CONFIGURATION ---
DECAY_RATE = 0.90
TRIALS = 1000
EPOCHS = 100

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

out_dir = "outputs/vanishing_autotuned_seed"
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

# --- 2. TRAIN FUNCTION WITH AUTO-TUNED SEED ---
def run_training(symbol, sit_name):
    processed_dir = f"data/processed_{symbol}_{sit_name}"
    model_path = f"{out_dir}/model_{symbol}_{sit_name}.pth"
    plot_prefix = f"{out_dir}/{symbol}_{sit_name}_"
    term_file = f"{out_dir}/{symbol}_{sit_name}_term.txt"
    
    if os.path.exists(term_file):
        print(f"      [Skipping training, found cached {term_file}]")
        with open(term_file, "r") as f:
            out = f.read()
    else:
        out = run_command([
            ".venv/bin/python3", "train_automl.py",
            "--trials", str(TRIALS),
            "--data-dir", processed_dir,
            "--save-model", model_path,
            "--plot-prefix", plot_prefix,
            "--epochs", str(EPOCHS)
        ])
        with open(term_file, "w") as f:
            f.write(out)
            
    metrics = extract_metrics(out)
    
    # Load optimal seed and hparams if saved
    best_params_file = f"{plot_prefix}best_params.json"
    hparams = {}
    if os.path.exists(best_params_file):
        with open(best_params_file, "r") as jf:
            hparams = json.load(jf)
            
    return metrics, hparams

def calc_imp(baseline, current, metric):
    if baseline is None or current is None: return None
    try:
        baseline, current = float(baseline), float(current)
        if abs(baseline) < 1e-8:
            return 0.0
        if metric in ['r2', 'ic', 'icir', 'sharpe']:
            return ((current - baseline) / abs(baseline)) * 100.0
        else:
            return ((baseline - current) / abs(baseline)) * 100.0
    except:
        return None

def img_to_base64(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return "data:image/png;base64," + base64.b64encode(f.read()).decode()
    return ""

def format_improvement(imp):
    if imp is None or np.isnan(imp):
        return '<span class="neutral badge">–</span>'
    color = "positive" if imp > 0.5 else ("negative" if imp < -0.5 else "neutral")
    arrow = "▲" if imp > 0.5 else ("▼" if imp < -0.5 else "–")
    return f'<span class="{color} badge">{arrow} {abs(imp):.1f}%</span>'

def plot_true_vs_preds(stock, true_vals, preds_dict, out_path):
    plt.figure(figsize=(14, 7))
    plt.style.use('dark_background')
    
    colors = {
        'No_Sentiment': '#94a3b8',
        'Standard_Sentiment': '#38bdf8',
        'Vanishing_Sentiment': '#ec4899',
        'Switching_Ensemble': '#f59e0b'
    }
    
    plt.plot(true_vals, label='True Price', color='#ffffff', linewidth=2.5, zorder=5)
    
    labels = {
        'No_Sentiment': 'Price Only Baseline',
        'Standard_Sentiment': 'Standard Sentiment (Zero-Pad)',
        'Vanishing_Sentiment': f'Vanishing Sentiment (α={DECAY_RATE})',
        'Switching_Ensemble': 'Switching Ensemble (Dynamic)'
    }
    
    for sit, preds in preds_dict.items():
        if preds is not None:
            plt.plot(preds, label=labels.get(sit, sit), color=colors.get(sit, '#a855f7'), linestyle='--', alpha=0.85, linewidth=1.8)
            
    plt.title(f"{stock} Price Prediction: Auto-Tuned Hyperparameters & Seed Comparison", fontsize=14, fontweight='bold', pad=12, color='#f8fafc')
    plt.xlabel("Test Set Time Steps (Days)", fontsize=11, color='#94a3b8')
    plt.ylabel("Close Price ($ / EGP)", fontsize=11, color='#94a3b8')
    plt.grid(True, linestyle=':', alpha=0.3, color='#475569')
    plt.legend(frameon=True, facecolor='#1e293b', edgecolor='#334155', loc='upper left', fontsize=10)
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
        
        preds_dict = {}
        true_vals_dict = None
        stock_results = {}
        hparams_dict = {}

        # 1. Train 3 core situations
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
            
            print(f"  -> Running Optuna Tuning (Tuning Seed + Hyperparameters)...")
            metrics, hparams = run_training(symbol, sit_name)
            stock_results[sit_name] = metrics
            hparams_dict[sit_name] = hparams
            
            csv_path = f"{out_dir}/{symbol}_{sit_name}_true_vs_pred.csv"
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                preds_dict[sit_name] = df['Predicted Close Price'].values
                if true_vals_dict is None:
                    true_vals_dict = df['True Close Price'].values

        # 2. Dynamic Switching Ensemble
        sit_name = "Switching_Ensemble"
        print(f"\n--- Situation: {sit_name} ---")
        x_test_path = f"data/processed_{symbol}_Standard_Sentiment/X_test.npy"
        if os.path.exists(x_test_path) and "No_Sentiment" in preds_dict and "Standard_Sentiment" in preds_dict:
            X_test = np.load(x_test_path)
            news_mask = (X_test[:, -1, 1] != 0.0)
            
            preds_no = preds_dict["No_Sentiment"]
            preds_std = preds_dict["Standard_Sentiment"]
            true_vals = true_vals_dict
            
            preds_switch = np.where(news_mask, preds_std, preds_no)
            switch_metrics = compute_switch_metrics(true_vals, preds_switch)
            stock_results[sit_name] = switch_metrics
            preds_dict[sit_name] = preds_switch
            hparams_dict[sit_name] = {
                "Strategy": "Dynamic Routing",
                "Rule": "Standard_Sentiment on news days, No_Sentiment on quiet days"
            }

        # 3. Generate combined chart
        combined_chart_path = f"{out_dir}/{symbol}_all_situations.png"
        plot_true_vs_preds(symbol, true_vals_dict, preds_dict, combined_chart_path)
        print(f"Saved combined chart to {combined_chart_path}")

        results.append({
            "Symbol": symbol,
            "Metrics": stock_results,
            "Hparams": hparams_dict,
            "Chart": combined_chart_path
        })

    # --- 5. REPORT GENERATION ---
    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Auto-Tuned Seed & Hyperparameters Comparison Report</title>
<style>
  :root { --bg: #0f1117; --card: #1a1d27; --border: #2a2d3e; --text: #e2e8f0; --dim: #8892a4; --accent: #6366f1; --positive: #22c55e; --negative: #ef4444; --neutral: #f59e0b; }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: 'Segoe UI', system-ui, sans-serif; padding: 32px; }
  h1 { font-size: 2rem; font-weight: 700; text-align: center; margin-bottom: 6px; background: linear-gradient(90deg, #6366f1, #a855f7); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
  .subtitle { text-align: center; color: var(--dim); margin-bottom: 40px; font-size: 0.95rem; }
  .section { background: var(--card); border: 1px solid var(--border); border-radius: 16px; padding: 28px; margin-bottom: 32px; overflow-x: auto; }
  .section h2 { font-size: 1.25rem; font-weight: 600; margin-bottom: 20px; padding-bottom: 12px; border-bottom: 1px solid var(--border); color: var(--accent); }
  table { width: 100%; min-width: 850px; border-collapse: collapse; font-size: 0.95rem; text-align: center; }
  th, td { padding: 14px 16px; border: 1px solid var(--border); vertical-align: middle; }
  th { background: #23263a; color: var(--dim); font-weight: 600; text-transform: uppercase; font-size: 0.8rem; letter-spacing: 0.05em; }
  tr.stock-header td { background: rgba(99, 102, 241, 0.1); color: #a855f7; font-weight: 700; font-size: 1.1rem; }
  td.metric-cell { font-weight: 600; background: rgba(255,255,255,0.02); width: 120px; text-align: left; }
  td.baseline-cell { font-weight: 600; color: #cbd5e1; font-size: 1.05rem; background: rgba(255,255,255,0.01); }
  .stat-group { display: flex; flex-direction: column; gap: 4px; align-items: center; justify-content: center; }
  .stat-group div { font-size: 1.05rem; font-weight: 500; }
  .positive { color: var(--positive); } .negative { color: var(--negative); } .neutral  { color: var(--neutral); }
  .badge { display: inline-block; font-size: 0.85rem; font-weight: 700; padding: 4px 8px; border-radius: 6px; background: rgba(0,0,0,0.2); width: fit-content; margin-top: 4px; }
  .seed-tag { font-size: 0.75rem; color: #a855f7; background: rgba(168,85,247,0.1); padding: 2px 6px; border-radius: 4px; margin-top: 2px; }
  .charts-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 24px; }
  .chart-card { background: var(--card); border: 1px solid var(--border); border-radius: 16px; padding: 20px; }
  .chart-card h3 { font-size: 1rem; margin-bottom: 14px; color: var(--dim); text-align: center; }
  .chart-card img { width: 100%; border-radius: 8px; }
</style>
</head>
<body>
<h1>🎯 Auto-Tuned Seed & Hyperparameters Model Comparison</h1>
<p class="subtitle">Decay Rate: 0.90 | Optuna Automated Seed & Architecture Optimization</p>

<div class="section">
  <h2>📊 Model Performance Matrix</h2>
  <table>
    <thead><tr>
      <th>Metric</th>
      <th>No Sentiment<br><span style="font-size:0.7rem;opacity:0.7">(Baseline)</span></th>
      <th>Standard Sentiment<br><span style="font-size:0.7rem;opacity:0.7">(Zero-Pad)</span></th>
      <th>Vanishing Sentiment<br><span style="font-size:0.7rem;opacity:0.7">(Decay α=0.90)</span></th>
      <th>Switching Ensemble<br><span style="font-size:0.7rem;opacity:0.7">(Dynamic)</span></th>
    </tr></thead>
    <tbody>"""

    for r in results:
        sym = r["Symbol"]
        html += f'<tr class="stock-header"><td colspan="5">{sym}</td></tr>'
        
        for m in metrics_to_show:
            base_val = r["Metrics"]["No_Sentiment"].get(m)
            base_str = f"{base_val:.4f}" if isinstance(base_val, float) else "N/A"
            no_seed = r["Hparams"]["No_Sentiment"].get("seed", "")
            
            html += f'<tr><td class="metric-cell">{m.upper()}</td>'
            html += f'<td class="baseline-cell"><div class="stat-group"><div>{base_str}</div><span class="seed-tag">Seed: {no_seed}</span></div></td>'
            
            for sit in ["Standard_Sentiment", "Vanishing_Sentiment", "Switching_Ensemble"]:
                val = r["Metrics"][sit].get(m)
                val_str = f"{val:.4f}" if isinstance(val, float) else "N/A"
                imp = calc_imp(base_val, val, m) if isinstance(base_val, float) and isinstance(val, float) else None
                seed_info = f'<span class="seed-tag">Seed: {r["Hparams"][sit].get("seed", "")}</span>' if "seed" in r["Hparams"].get(sit, {}) else ''
                
                html += f'<td><div class="stat-group"><div>{val_str}</div>{format_improvement(imp)}{seed_info}</div></td>'
            html += '</tr>'

    html += """</tbody>
  </table>
</div>

<div class="section">
  <h2>📉 True vs Predicted Overlaid Forecasts</h2>
  <div class="charts-grid">"""

    for r in results:
        chart_b64 = img_to_base64(r["Chart"])
        if chart_b64:
            html += f"""
            <div class="chart-card">
                <h3>{r['Symbol']} Multi-Model Forecast Overlay</h3>
                <img src="{chart_b64}" alt="{r['Symbol']}">
            </div>"""

    html += """
  </div>
</div>
</body>
</html>"""

    report_path = f"{out_dir}/report_vanishing_autotuned_seed.html"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n[DONE] Saved Auto-Tuned Seed report to: {report_path}")
