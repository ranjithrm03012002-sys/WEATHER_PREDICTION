import yfinance as yf
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# -----------------------------
# 1. DATA LOADING & PREPROCESSING
# -----------------------------
ticker = "AAPL"
data = yf.download(ticker, period="3y", interval="1d")
data = data[["Open", "High", "Low", "Close", "Volume"]].dropna()

scaler = MinMaxScaler()
scaled = scaler.fit_transform(data)

sequence_length = 60   # 60 days input window

def create_sequences(data, window):
    xs, ys = [], []
    for i in range(len(data) - window):
        xs.append(data[i:i+window])
        ys.append(data[i+window, 3])  # predict Close value
    return np.array(xs), np.array(ys)

X, y = create_sequences(scaled, sequence_length)
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

X_train, X_test = torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32)
y_train, y_test = torch.tensor(y_train, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)

train_loader = DataLoader(list(zip(X_train, y_train)), batch_size=32, shuffle=True)

# -----------------------------
# 2. TRANSFORMER MODEL
# -----------------------------
class TransformerModel(nn.Module):
    def __init__(self, feature_size=5, num_layers=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(feature_size, 64)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64, nhead=4, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(64, 1)
        self.attention_weights = None

    def forward(self, x):
        x = self.embedding(x)
        att = self.transformer.layers[0].self_attn
        self.attention_weights = att
        x = self.transformer(x)
        x = x[:, -1, :]
        return self.fc_out(x)

model = TransformerModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# -----------------------------
# 3. TRAINING LOOP
# -----------------------------
for epoch in range(30):
    model.train()
    epoch_loss = 0
    for seq, target in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(seq), target.unsqueeze(1))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1} Loss: {epoch_loss / len(train_loader):.6f}")

# -----------------------------
# 4. EVALUATION
# -----------------------------
model.eval()
with torch.no_grad():
    preds = model(X_test).squeeze().numpy()
    true = y_test.numpy()

rmse = np.sqrt(mean_squared_error(true, preds))
mae = mean_absolute_error(true, preds)
mape = np.mean(np.abs((true - preds) / true)) * 100

print("RMSE:", rmse, "MAE:", mae, "MAPE:", mape)
attention_map = model.attention_weights attn_output_weights
