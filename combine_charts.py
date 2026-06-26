import pandas as pd
import matplotlib.pyplot as plt
import os

stocks = ['JPM', 'BAC', 'COMI', 'CIEB']
situations = ['Original', 'Augmented', 'Filtered']
seed = 7100

colors = {
    'Original': '#ef4444',   # red
    'Augmented': '#3b82f6',  # blue
    'Filtered': '#22c55e'    # green
}

for stock in stocks:
    plt.figure(figsize=(14, 7))
    plt.style.use('dark_background')
    
    true_plotted = False
    
    for sit in situations:
        csv_file = f"outputs/situations/{stock}_{sit}_sent_{seed}_true_vs_pred.csv"
        if not os.path.exists(csv_file):
            continue
            
        df = pd.read_csv(csv_file)
        
        # Plot true line just once
        if not true_plotted:
            plt.plot(df['Time Step'], df['True Close Price'], color='white', label='True Price', linewidth=2.5, alpha=0.9)
            true_plotted = True
            
        # Plot predicted line for this situation
        plt.plot(df['Time Step'], df['Predicted Close Price'], color=colors[sit], label=f'Pred ({sit})', linewidth=1.5, alpha=0.8)

    plt.title(f"{stock} Price Prediction — All Situations Comparison (With Sentiment)", fontsize=16, pad=15)
    plt.xlabel('Time Step (Test Set)', fontsize=12)
    plt.ylabel('Stock Price', fontsize=12)
    plt.legend(loc='best', fontsize=11)
    plt.grid(True, alpha=0.2, color='gray', linestyle='--')
    plt.tight_layout()
    
    out_file = f"outputs/situations/{stock}_all_situations.png"
    plt.savefig(out_file, dpi=150, bbox_inches='tight', facecolor='#1a1d27')
    plt.close()
    print(f"Generated {out_file}")
