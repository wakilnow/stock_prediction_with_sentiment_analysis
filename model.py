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

class MultimodalStockTransformer(nn.Module):
    def __init__(self, num_features, d_model, nhead, num_layers, dropout=0.1, seq_length=30):
        super(MultimodalStockTransformer, self).__init__()
        
        # Linear projection to transform input features to d_model dimensions
        self.input_projection = nn.Linear(num_features, d_model)
        
        # Positional encoding param
        # A simple learnable parameter for positional sequence embeddings
        self.positional_encoding = nn.Parameter(torch.randn(1, seq_length, d_model))
        
        # Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Final fully connected layer to predict the next Close Price
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_length, num_features)
        batch_size, seq_length, _ = x.shape
        
        # 1. Project input features
        x = self.input_projection(x) # (batch_size, seq_length, d_model)
        
        # 2. Add positional encoding
        x = x + self.positional_encoding[:, :seq_length, :]
        
        # 3. Pass through Transformer Encoder
        x = self.transformer_encoder(x) # (batch_size, seq_length, d_model)
        
        # 4. Use the output of the last time step for prediction
        last_time_step = x[:, -1, :] # (batch_size, d_model)
        
        # 5. Regression head
        output = self.fc(last_time_step) # (batch_size, 1)
        
        return output.squeeze(-1) # (batch_size,)
