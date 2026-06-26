import pandas as pd
import base64
import os

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

df = pd.read_csv("outputs/situations/situations_comparison.csv")

metrics = ["MAE", "RMSE", "R2", "MAPE", "IC", "ICIR", "SHARPE"]

html_rows = ""

for symbol, group in df.groupby("Symbol"):
    html_rows += f'<tr class="stock-header"><td colspan="5">{symbol}</td></tr>'
    
    for metric in metrics:
        metric_group = group[group["Metric"] == metric]
        row_html = f'<tr><td class="metric-cell">{metric}</td>'
        
        # 1. No Sentiment (Baseline) - Taken from Original Situation
        orig_row = metric_group[metric_group["Situation"] == "Original"]
        if len(orig_row) > 0:
            no_sent = orig_row.iloc[0]["No_Sentiment"]
            no_sent_display = no_sent.split(" ± ")[0] if pd.notna(no_sent) else "-"
            row_html += f'<td class="baseline-cell">{no_sent_display}</td>'
        else:
            row_html += '<td class="baseline-cell">-</td>'
            
        # Helper to build standard cell
        def build_sent_cell(sit_name):
            sit_row = metric_group[metric_group["Situation"] == sit_name]
            if len(sit_row) > 0:
                row = sit_row.iloc[0]
                with_sent = row["With_Sentiment"]
                with_sent_display = with_sent.split(" ± ")[0] if pd.notna(with_sent) else "-"
                imp = row["Mean_Improvement"]
                return f'<td><div class="stat-group"><div>{with_sent_display}</div>{format_improvement(imp)}</div></td>'
            return "<td>-</td>"

        # 2. With Sentiment (Original)
        row_html += build_sent_cell("Original")
        # 3. Augmented
        row_html += build_sent_cell("Augmented")
        # 4. Filtered
        row_html += build_sent_cell("Filtered")
        
        row_html += "</tr>"
        html_rows += row_html

# Charts
charts_html = ""
for symbol in df["Symbol"].unique():
    img_path = f"outputs/situations/{symbol}_all_situations.png"
    b64 = img_to_base64(img_path)
    if b64:
        charts_html += f"""
        <div class="chart-card">
            <h3>{symbol} Price Prediction — All Situations Overlaid</h3>
            <img src="{b64}" alt="{symbol} all situations plot">
        </div>"""

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>3-Way Data Situation Comparison (Pivoted)</title>
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
<h1>📈 Data Quality Comparison Report</h1>
<p class="subtitle">Model Performances Relative to No-Sentiment Baseline</p>

<div class="section">
  <h2>📊 Aggregate Metrics Comparison</h2>
  <table>
    <thead><tr>
      <th>Metric</th>
      <th>No Sentiment<br><span style="font-size:0.7rem;font-weight:normal;opacity:0.7">(Price Only Baseline)</span></th>
      <th>With Sentiment<br><span style="font-size:0.7rem;font-weight:normal;opacity:0.7">(Original News)</span></th>
      <th>Augmented<br><span style="font-size:0.7rem;font-weight:normal;opacity:0.7">(NLP Expanded News)</span></th>
      <th>Filtered<br><span style="font-size:0.7rem;font-weight:normal;opacity:0.7">(Intersected News)</span></th>
    </tr></thead>
    <tbody>{html_rows}</tbody>
  </table>
</div>

<div class="section">
  <h2>📉 Representative Charts</h2>
  <div class="charts-grid">{charts_html}</div>
</div>
</body>
</html>"""

out_path = "outputs/situations/report_situations_pivoted.html"
with open(out_path, "w", encoding="utf-8") as f:
    f.write(html)
print(f"Generated pivoted HTML at: {out_path}")
