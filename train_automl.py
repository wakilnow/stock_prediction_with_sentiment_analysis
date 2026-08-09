# train_automl.py
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import random

from model import TimeSeriesDataset, StockLSTM

def set_seed(seed=42):
    """Set all random seeds for reproducible results."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_data(data_dir="data/processed"):
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    X_test  = np.load(os.path.join(data_dir, "X_test.npy"))
    y_test  = np.load(os.path.join(data_dir, "y_test.npy"))
    return X_train, y_train, X_test, y_test

def train_and_evaluate(args, X_train, y_train, X_valid, y_valid, X_test=None, y_test=None, plot_prefix=None, epochs=25):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    
    num_features = X_train.shape[2]
    seq_length = X_train.shape[1]
    
    model = StockLSTM(
        num_features=num_features,
        hidden_dim=args['d_model'],
        num_layers=args['num_layers'],
        dropout=args['dropout'],
        seq_length=seq_length
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args['lr'])
    
    train_dataset = TimeSeriesDataset(X_train, y_train)
    valid_dataset = TimeSeriesDataset(X_valid, y_valid)
    
    if X_test is not None and y_test is not None:
        test_dataset = TimeSeriesDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=args['batch_size'], shuffle=False)
    else:
        test_loader = None
    
    train_loader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args['batch_size'], shuffle=False)
    
    best_val_loss = float('inf')
    early_stopping_patience = 5
    patience_counter = 0
    
    train_losses = []
    val_losses = []
    test_losses = []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_x.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in valid_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_x.size(0)
                
        val_loss /= len(valid_loader.dataset)
        
        if test_loader is not None:
            test_loss = 0.0
            with torch.no_grad():
                for batch_x, batch_y in test_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    test_loss += loss.item() * batch_x.size(0)
            test_loss /= len(test_loader.dataset)
            test_losses.append(test_loss)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            if args.get('save_path'):
                torch.save(model.state_dict(), args['save_path'])
        else:
            patience_counter += 1
            
        if patience_counter >= early_stopping_patience:
            break
            
    if plot_prefix:
        os.makedirs(os.path.dirname(plot_prefix) or '.', exist_ok=True)
        
        # Save to CSV
        csv_data = {
            'Epoch': range(1, len(train_losses) + 1),
            'Train Loss': train_losses,
            'Validation Loss': val_losses
        }
        if test_losses:
            csv_data['Test Loss'] = test_losses
            
        df_metrics = pd.DataFrame(csv_data)
        df_metrics.to_csv(f"{plot_prefix}training_curve.csv", index=False)
        
        # Plot
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
        plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
        plt.title('Training and Validation Loss Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Loss (MSE)')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{plot_prefix}training_curve.png")
        plt.close()
            
    return best_val_loss

def objective(trial):
    # Search Space
    d_model = trial.suggest_categorical('d_model', [16, 32, 64])
    # Ensure d_model is divisible by nhead
    nhead_choices = [2, 4, 8]
    valid_nheads = [h for h in nhead_choices if d_model % h == 0]
    nhead = trial.suggest_categorical('nhead', valid_nheads)
    
    num_layers = trial.suggest_int('num_layers', 1, 3)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    lr = trial.suggest_float('lr', 1e-4, 5e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32])
    
    args = {
        'd_model': d_model,
        'nhead': nhead,
        'num_layers': num_layers,
        'dropout': dropout,
        'lr': lr,
        'batch_size': batch_size
    }
    
    # Load Data
    global DATA_DIR, cmd_args
    X_train, y_train, X_test, y_test = load_data(data_dir=DATA_DIR)
    
    # Chronological split for validation out of Train set
    train_size = int(len(X_train) * 0.8)
    X_tr, y_tr = X_train[:train_size], y_train[:train_size]
    X_va, y_va = X_train[train_size:], y_train[train_size:]
    
    val_loss = train_and_evaluate(args, X_tr, y_tr, X_va, y_va, epochs=cmd_args.epochs)
    return val_loss

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=5, help='Number of optuna trials (set to 0 for fixed hyperparameters)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--data-dir', type=str, default='data/processed', help='Directory containing the processed numpy arrays')
    parser.add_argument('--save-model', type=str, default='models/best_transformer.pth', help='Path to save the best model')
    parser.add_argument('--plot-prefix', type=str, default=None, help='Prefix for saved plots (e.g., models/sentiment_)')
    
    # Fixed hyperparameter arguments
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.15)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=25, help='Number of training epochs')
    
    cmd_args = parser.parse_args()

    # Pass data_dir into global scope for objective function
    DATA_DIR = cmd_args.data_dir

    print(f"Starting on dataset: {DATA_DIR}")
    set_seed(cmd_args.seed)
    
    if cmd_args.trials > 0:
        print(f"Running Optuna optimization with {cmd_args.trials} trials...")
        sampler = optuna.samplers.TPESampler(seed=cmd_args.seed)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(objective, n_trials=cmd_args.trials)
        
        print("\nBest hyperparameters data:")
        print(study.best_params)
        best_args = study.best_params
    else:
        print("\nUsing fixed hyperparameters (skipping Optuna)...")
        best_args = {
            'd_model': cmd_args.d_model,
            'nhead': cmd_args.nhead,
            'num_layers': cmd_args.num_layers,
            'dropout': cmd_args.dropout,
            'lr': cmd_args.lr,
            'batch_size': cmd_args.batch_size
        }
        print(best_args)
        
    print("\nTraining final model with hparams (using chronological validation split)...")
    best_args['save_path'] = cmd_args.save_model
    os.makedirs(os.path.dirname(cmd_args.save_model) or '.', exist_ok=True)
    
    X_train, y_train, X_test, y_test = load_data(data_dir=DATA_DIR)
    
    # Chronological split for validation out of Train set
    train_size = int(len(X_train) * 0.8)
    X_tr, y_tr = X_train[:train_size], y_train[:train_size]
    X_va, y_va = X_train[train_size:], y_train[train_size:]
    
    best_val_loss = train_and_evaluate(best_args, X_tr, y_tr, X_va, y_va, X_test=X_test, y_test=y_test, plot_prefix=cmd_args.plot_prefix, epochs=cmd_args.epochs)
    
    print(f"Validation loss (MSE Scaled) for best model: {best_val_loss:.6f}")
    
    # Compute MAE & RMSE on Original Scale
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    model = StockLSTM(
        num_features=X_train.shape[2],
        hidden_dim=best_args['d_model'],
        num_layers=best_args['num_layers'],
        dropout=best_args['dropout'],
        seq_length=X_train.shape[1]
    ).to(device)
    
    model.load_state_dict(torch.load(cmd_args.save_model))
    model.eval()
    
    test_dataset = TimeSeriesDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            preds = model(batch_x).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(batch_y.numpy())
            
    all_preds = np.array(all_preds).reshape(-1, 1)
    all_targets = np.array(all_targets).reshape(-1, 1)
    
    # Inverse transform
    scaler = joblib.load(os.path.join(DATA_DIR, 'scaler.save'))
    all_preds_inv = scaler.inverse_transform(all_preds)
    all_targets_inv = scaler.inverse_transform(all_targets)
    
    # Compute Metrics
    y_true = all_targets_inv.flatten()
    y_pred = all_preds_inv.flatten()
    
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    
    # R2
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - (ss_res / (ss_tot + 1e-10))
    
    # MAPE
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
    
    # Returns-based metrics (IC, ICIR, Sharpe)
    if len(y_true) > 1:
        # Simple returns
        actual_returns = np.diff(y_true) / (y_true[:-1] + 1e-10)
        # Predicted returns (predicted next-day price vs today's actual price)
        pred_returns = (y_pred[1:] - y_true[:-1]) / (y_true[:-1] + 1e-10)
        
        # IC (Pearson Correlation)
        ic = np.corrcoef(pred_returns, actual_returns)[0, 1]
        if np.isnan(ic): ic = 0.0
        
        # ICIR (Mean IC / Std IC)
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
            
        # Sharpe (Annualized, simple strategy)
        # Strategy: Long if pred_return > 0, else stay out
        strat_returns = np.where(pred_returns > 0, actual_returns, 0)
        sharpe = (np.mean(strat_returns) / (np.std(strat_returns) + 1e-10)) * np.sqrt(252)
    else:
        ic, icir, sharpe = 0.0, 0.0, 0.0

    print(f"Final Test MAE (Original Price Scale): ${mae:.2f}")
    print(f"Final Test RMSE (Original Price Scale): ${rmse:.2f}")
    print(f"Final Test R2 Score: {r2:.4f}")
    print(f"Final Test MAPE: {mape:.2f}%")
    print(f"Final Test IC: {ic:.4f}")
    print(f"Final Test ICIR: {icir:.4f}")
    print(f"Final Test Sharpe Ratio: {sharpe:.4f}")

    if cmd_args.plot_prefix:
        os.makedirs(os.path.dirname(cmd_args.plot_prefix) or '.', exist_ok=True)
        
        # Save to CSV
        preds_flat = all_preds_inv.flatten()
        targets_flat = all_targets_inv.flatten()
        df_preds = pd.DataFrame({
            "Time Step": range(1, len(preds_flat) + 1),
            "True Close Price": targets_flat,
            "Predicted Close Price": preds_flat
        })
        df_preds.to_csv(f"{cmd_args.plot_prefix}true_vs_pred.csv", index=False)
        
        # Save metrics to CSV
        df_metrics = pd.DataFrame({
            "Metric": ["MAE", "RMSE", "R2", "MAPE", "IC", "ICIR", "Sharpe Ratio"],
            "Value": [mae, rmse, r2, mape, ic, icir, sharpe]
        })
        df_metrics.to_csv(f"{cmd_args.plot_prefix}metrics.csv", index=False)
        
        # Save best hyperparameters to JSON
        import json
        params_to_save = {k: v for k, v in best_args.items() if k != 'save_path'}
        params_to_save['best_val_loss'] = float(best_val_loss)
        with open(f"{cmd_args.plot_prefix}best_params.json", "w") as jf:
            json.dump(params_to_save, jf, indent=2)
        
        # Plot
        plt.figure(figsize=(14, 6))
        plt.plot(all_targets_inv, label='True Close Price', color='blue', alpha=0.7)
        plt.plot(all_preds_inv, label='Predicted Close Price', color='red', alpha=0.7)
        plt.xlabel('Time Steps (Days)')
        plt.ylabel('Price (USD)')
        plt.title('True vs Predicted Stock Prices on Test Set')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{cmd_args.plot_prefix}true_vs_pred.png")
        plt.close()
