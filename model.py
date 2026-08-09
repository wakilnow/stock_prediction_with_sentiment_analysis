# model.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        """
        X: numpy array of shape (N, seq_length, num_features)
        y: numpy array of shape (N,)
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class StockLSTM(nn.Module):
    def __init__(self, num_features, hidden_dim, num_layers=1, dropout=0.1, seq_length=30):
        super(StockLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layer
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True
        )
        
        # Regression head to predict next Close price
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_length, num_features)
        lstm_out, _ = self.lstm(x) # (batch_size, seq_length, hidden_dim)
        
        # Use output of last time step
        last_step = lstm_out[:, -1, :] # (batch_size, hidden_dim)
        
        # Output prediction
        output = self.fc(last_step) # (batch_size, 1)
        return output.squeeze(-1) # (batch_size,)

# Backward compatibility alias
MultimodalStockTransformer = StockLSTM
