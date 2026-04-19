# train_automl.py
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt

from model import TimeSeriesDataset, MultimodalStockTransformer

def load_data(data_dir="data/processed"):
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    X_test  = np.load(os.path.join(data_dir, "X_test.npy"))
    y_test  = np.load(os.path.join(data_dir, "y_test.npy"))
    return X_train, y_train, X_test, y_test

def train_and_evaluate(args, X_train, y_train, X_valid, y_valid, X_test=None, y_test=None, plot_prefix=None):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    
    num_features = X_train.shape[2]
    seq_length = X_train.shape[1]
    
    model = MultimodalStockTransformer(
        num_features=num_features,
        d_model=args['d_model'],
        nhead=args['nhead'],
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
    epochs = 25
    
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
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
        plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
        if test_losses:
            plt.plot(range(1, len(test_losses) + 1), test_losses, label='Test Loss')
            plt.title('Training, Validation, and Test Loss Curve')
        else:
            plt.title('Training and Validation Loss Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Loss (MSE)')
        plt.legend()
        plt.grid(True)
        os.makedirs(os.path.dirname(plot_prefix) or '.', exist_ok=True)
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
    global DATA_DIR
    X_train, y_train, X_test, y_test = load_data(data_dir=DATA_DIR)
    
    # Chronological split for validation out of Train set
    train_size = int(len(X_train) * 0.8)
    X_tr, y_tr = X_train[:train_size], y_train[:train_size]
    X_va, y_va = X_train[train_size:], y_train[train_size:]
    
    val_loss = train_and_evaluate(args, X_tr, y_tr, X_va, y_va)
    return val_loss

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=5, help='Number of optuna trials')
    parser.add_argument('--data-dir', type=str, default='data/processed', help='Directory containing the processed numpy arrays')
    parser.add_argument('--save-model', type=str, default='models/best_transformer.pth', help='Path to save the best model')
    parser.add_argument('--plot-prefix', type=str, default=None, help='Prefix for saved plots (e.g., models/sentiment_)')
    cmd_args = parser.parse_args()

    # Pass data_dir into global scope for objective function
    DATA_DIR = cmd_args.data_dir

    print(f"Starting Optuna optimization on dataset: {DATA_DIR}")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=cmd_args.trials)
    
    print("\nBest hyperparameters data:")
    print(study.best_params)
    
    print("\nTraining final model with best hparams (using chronological validation split)...")
    best_args = study.best_params
    best_args['save_path'] = cmd_args.save_model
    os.makedirs(os.path.dirname(cmd_args.save_model) or '.', exist_ok=True)
    
    X_train, y_train, X_test, y_test = load_data(data_dir=DATA_DIR)
    
    # Chronological split for validation out of Train set
    train_size = int(len(X_train) * 0.8)
    X_tr, y_tr = X_train[:train_size], y_train[:train_size]
    X_va, y_va = X_train[train_size:], y_train[train_size:]
    
    best_val_loss = train_and_evaluate(best_args, X_tr, y_tr, X_va, y_va, X_test=X_test, y_test=y_test, plot_prefix=cmd_args.plot_prefix)
    
    print(f"Validation loss (MSE Scaled) for best model: {best_val_loss:.6f}")
    
    # Compute MAE & RMSE on Original Scale
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    model = MultimodalStockTransformer(
        num_features=X_train.shape[2],
        d_model=best_args['d_model'],
        nhead=best_args['nhead'],
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
    
    mae = np.mean(np.abs(all_preds_inv - all_targets_inv))
    rmse = np.sqrt(np.mean((all_preds_inv - all_targets_inv)**2))
    
    print(f"Final Test MAE (Original Price Scale): ${mae:.2f}")
    print(f"Final Test RMSE (Original Price Scale): ${rmse:.2f}")

    if cmd_args.plot_prefix:
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
