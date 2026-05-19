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

if __name__ == "__main__":
    # Base configuration
    START_DATE = "2020-01-01"
    END_DATE = "2025-12-31"
    TRIALS = "50" # Optuna trials per model
    SEED = "7100"

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
            "news_csv": "data/news_investing.com/bac_news.csv",
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
        model_sentiment  = f"outputs/models/best_transformer_with_sentiment_{symbol}.pth"
        
        dir_no_sentiment = f"data/processed_no_sentiment_{symbol}"
        model_no_sentiment = f"outputs/models/best_transformer_no_sentiment_{symbol}.pth"

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
            "--plot-prefix", f"outputs/models/{symbol}_sentiment_",
            "--seed", SEED
        ])
        with open(f"outputs/models/{symbol}_sentiment_terminal_output.txt", "w") as f:
            f.write(out_sentiment)
        m_sent = extract_metrics(out_sentiment)

        # Step 4: Train Model WITHOUT Sentiment
        print("\n" + "=" * 60)
        print(f"STEP 4: Training Model WITHOUT Sentiment - {symbol}")
        print("=" * 60)
        out_no_sentiment = run_command([
            ".venv/bin/python3", "train_automl.py",
            "--trials", TRIALS,
            "--data-dir", dir_no_sentiment,
            "--save-model", model_no_sentiment,
            "--plot-prefix", f"outputs/models/{symbol}_no_sentiment_",
            "--seed", SEED
        ])
        with open(f"outputs/models/{symbol}_no_sentiment_terminal_output.txt", "w") as f:
            f.write(out_no_sentiment)
        m_no_sent = extract_metrics(out_no_sentiment)

        results.append({
            "Symbol": symbol,
            "Model": sentiment_model_name,
            "No_Sent": m_no_sent,
            "With_Sent": m_sent
        })

        # Combine predictions plot
        print(f"\nCreating combined plot for {symbol}...")
        try:
            import pandas as pd
            import matplotlib.pyplot as plt
            
            sent_csv = f"outputs/models/{symbol}_sentiment_true_vs_pred.csv"
            no_sent_csv = f"outputs/models/{symbol}_no_sentiment_true_vs_pred.csv"
            
            df_sent = pd.read_csv(sent_csv)
            df_no_sent = pd.read_csv(no_sent_csv)
            
            plt.figure(figsize=(14, 6))
            # True values are the same in both, so we can just use one
            plt.plot(df_sent['True Close Price'], label='True Close Price', color='blue', alpha=0.6, linewidth=2)
            plt.plot(df_no_sent['Predicted Close Price'], label='Pred (No Sentiment)', color='orange', alpha=0.8, linestyle='--')
            plt.plot(df_sent['Predicted Close Price'], label='Pred (With Sentiment)', color='green', alpha=0.8, linestyle='--')
            
            plt.title(f'{symbol}: True vs Predicted Prices (Sentiment vs No Sentiment)')
            plt.xlabel('Time Steps (Days)')
            plt.ylabel('Price')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"outputs/models/{symbol}_combined_true_vs_pred.png")
            plt.close()
            print(f"Saved combined plot to outputs/models/{symbol}_combined_true_vs_pred.png")
        except Exception as e:
            print(f"Could not create combined plot for {symbol}: {e}")

    # Step 5: Final Comparison Table
    print("\n" + "=" * 120)
    print("FINAL COMPARISON RESULTS (2020 - 2025)")
    print("=" * 120)
    
    metrics_to_show = ["mae", "rmse", "r2", "mape", "ic", "icir", "sharpe"]
    
    # 1. Unified Table for Terminal
    print(f"{'Stock':<8} | {'Metric':<8} | {'No Sentiment':<15} | {'With Sentiment':<15} | {'Improvement'}")
    print("-" * 80)
    for r in results:
        for m in metrics_to_show:
            no_val = r["No_Sent"][m]
            si_val = r["With_Sent"][m]
            
            # Calculate improvement percentage
            try:
                nv = float(no_val)
                sv = float(si_val)
                if nv != 0:
                    # For error metrics (MAE, RMSE, MAPE), lower is better
                    if m in ["mae", "rmse", "mape"]:
                        imp = (nv - sv) / nv * 100
                    else:
                        imp = (sv - nv) / abs(nv) * 100
                    imp_str = f"{imp:+.2f}%"
                else:
                    imp_str = "N/A"
            except:
                imp_str = "N/A"
                
            print(f"{r['Symbol']:<8} | {m.upper():<8} | {no_val:<15} | {si_val:<15} | {imp_str}")
        print("-" * 80)
    
    # 2. Save to CSV
    import csv
    csv_file = "outputs/models/comparison_results.csv"
    os.makedirs("outputs/models", exist_ok=True)
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Symbol", "Sentiment_Model", "Metric", "No_Sentiment", "With_Sentiment", "Improvement"])
        for r in results:
            for m in metrics_to_show:
                no_val = r["No_Sent"][m]
                si_val = r["With_Sent"][m]
                
                try:
                    nv, sv = float(no_val), float(si_val)
                    if nv != 0:
                        if m in ["mae", "rmse", "mape"]: imp = (nv - sv) / nv * 100
                        else: imp = (sv - nv) / abs(nv) * 100
                        imp_str = f"{imp:.4f}%"
                    else: imp_str = "0%"
                except: imp_str = "N/A"
                
                writer.writerow([r["Symbol"], r["Model"], m.upper(), no_val, si_val, imp_str])
    
    print(f"\n[DONE] Consolidated results saved to: {csv_file}")
    
    # 3. Generate HTML Report
    import json
    import base64
    from datetime import datetime

    def img_to_base64(path):
        """Convert an image file to a base64 data URI for embedding in HTML."""
        if os.path.exists(path):
            with open(path, "rb") as f:
                return "data:image/png;base64," + base64.b64encode(f.read()).decode()
        return ""

    def calc_imp(no_val, si_val, metric):
        try:
            nv, sv = float(no_val), float(si_val)
            if nv == 0: return 0.0
            return (nv - sv) / nv * 100 if metric in ["mae","rmse","mape"] else (sv - nv) / abs(nv) * 100
        except:
            return None

    def imp_cell(imp, val):
        """Returns styled <td> HTML for an improvement value."""
        if imp is None:
            return f'<td class="neutral">{val}</td>'
        color = "positive" if imp > 0.5 else ("negative" if imp < -0.5 else "neutral")
        arrow = "▲" if imp > 0.5 else ("▼" if imp < -0.5 else "–")
        return f'<td class="{color}">{val}<span class="badge">{arrow} {abs(imp):.1f}%</span></td>'

    metric_labels = {
        "mae": "MAE ($)", "rmse": "RMSE ($)", "r2": "R² Score",
        "mape": "MAPE (%)", "ic": "IC", "icir": "ICIR", "sharpe": "Sharpe Ratio"
    }
    param_keys = ["d_model", "nhead", "num_layers", "dropout", "lr", "batch_size", "best_val_loss"]

    # Build metrics rows
    metrics_rows_html = ""
    for r in results:
        symbol = r["Symbol"]
        for i, m in enumerate(metrics_to_show):
            no_val = r["No_Sent"][m]
            si_val = r["With_Sent"][m]
            imp = calc_imp(no_val, si_val, m)
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

    # Build hyperparameters rows
    hparam_rows_html = ""
    for r in results:
        symbol = r["Symbol"]
        sent_path = f"outputs/models/{symbol}_sentiment_best_params.json"
        no_sent_path = f"outputs/models/{symbol}_no_sentiment_best_params.json"
        sent_params = json.load(open(sent_path)) if os.path.exists(sent_path) else {}
        no_sent_params = json.load(open(no_sent_path)) if os.path.exists(no_sent_path) else {}

        for i, pk in enumerate(param_keys):
            sv = sent_params.get(pk, "N/A")
            nv = no_sent_params.get(pk, "N/A")
            # Format floats nicely
            sv_str = f"{sv:.6f}" if isinstance(sv, float) else str(sv)
            nv_str = f"{nv:.6f}" if isinstance(nv, float) else str(nv)
            row_class = "alt-row" if i % 2 == 0 else ""
            sym_cell = f'<td class="symbol-cell" rowspan="{len(param_keys)}">{symbol}</td>' if i == 0 else ""
            hparam_rows_html += f"""
            <tr class="{row_class}">
                {sym_cell}
                <td><strong>{pk}</strong></td>
                <td>{nv_str}</td>
                <td>{sv_str}</td>
            </tr>"""
        hparam_rows_html += '<tr class="spacer"><td colspan="4"></td></tr>'

    # Build combined chart images section
    charts_html = ""
    for r in results:
        symbol = r["Symbol"]
        img_path = f"outputs/models/{symbol}_combined_true_vs_pred.png"
        b64 = img_to_base64(img_path)
        if b64:
            charts_html += f"""
            <div class="chart-card">
                <h3>{symbol} — True vs Predicted (With & Without Sentiment)</h3>
                <img src="{b64}" alt="{symbol} combined plot">
            </div>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Stock Prediction Sentiment Analysis Report</title>
<style>
  :root {{
    --bg: #0f1117; --card: #1a1d27; --border: #2a2d3e;
    --text: #e2e8f0; --dim: #8892a4; --accent: #6366f1;
    --positive: #22c55e; --negative: #ef4444; --neutral: #f59e0b;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text); font-family: 'Segoe UI', system-ui, sans-serif; padding: 32px; }}
  h1 {{ font-size: 2rem; font-weight: 700; text-align: center; margin-bottom: 6px;
       background: linear-gradient(90deg, #6366f1, #a855f7); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
  .subtitle {{ text-align: center; color: var(--dim); margin-bottom: 40px; font-size: 0.95rem; }}
  .section {{ background: var(--card); border: 1px solid var(--border); border-radius: 16px; padding: 28px; margin-bottom: 32px; }}
  .section h2 {{ font-size: 1.25rem; font-weight: 600; margin-bottom: 20px; padding-bottom: 12px;
                 border-bottom: 1px solid var(--border); color: var(--accent); }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
  th {{ background: #23263a; color: var(--dim); font-weight: 600; text-transform: uppercase;
        font-size: 0.75rem; letter-spacing: 0.05em; padding: 12px 16px; text-align: left; }}
  td {{ padding: 11px 16px; border-bottom: 1px solid var(--border); vertical-align: middle; }}
  tr.alt-row td {{ background: rgba(255,255,255,0.02); }}
  tr.spacer td {{ padding: 4px; border: none; }}
  td.symbol-cell {{ font-weight: 700; font-size: 1rem; color: var(--accent); border-right: 1px solid var(--border); vertical-align: top; padding-top: 14px; }}
  td.positive {{ color: var(--positive); }}
  td.negative {{ color: var(--negative); }}
  td.neutral  {{ color: var(--text); }}
  .badge {{ display: inline-block; margin-left: 8px; font-size: 0.72rem; opacity: 0.8; font-weight: 600; }}
  .charts-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(600px, 1fr)); gap: 24px; }}
  .chart-card {{ background: var(--card); border: 1px solid var(--border); border-radius: 16px; padding: 20px; }}
  .chart-card h3 {{ font-size: 1rem; margin-bottom: 14px; color: var(--dim); }}
  .chart-card img {{ width: 100%; border-radius: 8px; }}
  .meta {{ text-align: center; color: var(--dim); font-size: 0.8rem; margin-top: 32px; }}
</style>
</head>
<body>
<h1>📈 Stock Prediction: Sentiment Analysis Report</h1>
<p class="subtitle">Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} &nbsp;|&nbsp; Seed: 7100 &nbsp;|&nbsp; Trials: 50 &nbsp;|&nbsp; Optimizer: Bayesian GP</p>

<div class="section">
  <h2>📊 Metrics Comparison — With vs Without Sentiment</h2>
  <table>
    <thead><tr>
      <th>Stock</th><th>Metric</th>
      <th>Without Sentiment</th>
      <th>With Sentiment ↔ Improvement</th>
    </tr></thead>
    <tbody>{metrics_rows_html}</tbody>
  </table>
</div>

<div class="section">
  <h2>⚙️ Best Hyperparameters — Bayesian Optimization Results</h2>
  <table>
    <thead><tr>
      <th>Stock</th><th>Hyperparameter</th>
      <th>Without Sentiment</th>
      <th>With Sentiment</th>
    </tr></thead>
    <tbody>{hparam_rows_html}</tbody>
  </table>
</div>

<div class="section">
  <h2>📉 Prediction Charts</h2>
  <div class="charts-grid">{charts_html}</div>
</div>

<p class="meta">Stock Sentiment Analysis Research · ASU · 2020–2025</p>
</body>
</html>"""

    html_file = "outputs/models/report.html"
    with open(html_file, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"[DONE] HTML report saved to: {html_file}")
    print("=" * 120)
