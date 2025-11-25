"""
Advanced Time Series Forecasting with Deep Learning and Attention (Single-file implementation)

What this file provides:
- Synthetic multivariate time series generator with non-stationarity + multiple seasonalities
- LSTM with additive (Bahdanau-style) attention (object-oriented, PyTorch)
- MLP baseline (PyTorch)
- SARIMA baseline (statsmodels)
- Hyperparameter search (Optuna if available, otherwise randomized search)
- Training, evaluation (MAE, RMSE, MAPE), and attention visualization
- Full reproducibility seed and easy-to-run CLI-style section at bottom

Usage:
- Install requirements (recommended): 
    pip install numpy pandas scipy matplotlib scikit-learn torch statsmodels optuna
  (optuna is optional; if missing the script will fall back to randomized search)
- Run:
    python timeseries_attention_project.py

This file is intended to be run as a script. It is self-contained and documented.

Author: ChatGPT (code-only deliverable requested)
Date: 2025-11-25
"""

import os
import math
import time
import random
from typing import Tuple, Dict, Any, List

import numpy as np
import pandas as pd
from scipy import signal
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# SARIMA baseline
import statsmodels.api as sm

# Try to import optuna (optional). If not present, we'll fallback to randomized search.
try:
    import optuna
    OPTUNA_AVAILABLE = True
except Exception:
    OPTUNA_AVAILABLE = False

# ---------------------------
# Utilities and config
# ---------------------------
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

set_seed(SEED)

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mape(y_true, y_pred):
    # avoid divide-by-zero by adding small epsilon
    eps = 1e-8
    return np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100.0

# ---------------------------
# 1) Dataset generator
# ---------------------------
class SyntheticMultivariateSeries:
    """
    Generates a multivariate time series with:
     - long-term trend (non-stationary)
     - multiple seasonal components (different frequencies)
     - an exogenous slowly varying signal
     - heteroscedastic noise (to make it more realistic)
    Produces a pandas DataFrame with columns: ['feature_0', 'feature_1', 'feature_2']
    """
    def __init__(self, n_steps: int = 5000, seed: int = SEED):
        self.n_steps = n_steps
        self.seed = seed
        np.random.seed(seed)

    def generate(self) -> pd.DataFrame:
        t = np.arange(self.n_steps)

        # Trend: quadratic / piecewise to ensure non-stationarity
        trend = 0.0002 * (t ** 2) / (self.n_steps) + 0.02 * t

        # Seasonality 1: yearly-ish (slow)
        s1 = 10 * np.sin(2 * np.pi * t / 200)  # period 200

        # Seasonality 2: weekly-ish (faster)
        s2 = 3.5 * np.sin(2 * np.pi * t / 30)  # period 30

        # Intermittent events: spikes / structural breaks
        spikes = np.zeros_like(t, dtype=float)
        for loc in [int(self.n_steps*0.2), int(self.n_steps*0.5), int(self.n_steps*0.8)]:
            spikes[loc:loc+5] += np.linspace(5, 0, 5)

        # Exogenous slow-moving feature (e.g., temperature-like)
        exog = 2.0 * np.sin(2 * np.pi * t / 365) + 0.01 * t + np.random.normal(0, 0.3, size=self.n_steps)

        # Feature 0: main target-like signal (trend + seasons + noise)
        noise0 = np.random.normal(0, 1.0 + 0.002 * t, size=self.n_steps)  # heteroscedastic noise
        feature_0 = trend + s1 + 0.6 * s2 + spikes + noise0

        # Feature 1: correlated with feature_0 but shifted lags and scaled
        noise1 = np.random.normal(0, 0.8 + 0.0015 * t, size=self.n_steps)
        feature_1 = 0.5 * trend + 0.8 * np.roll(s1, 3) + 0.4 * s2 + 0.3 * exog + noise1

        # Feature 2: exogenous + interactions
        noise2 = np.random.normal(0, 0.6 + 0.001 * t, size=self.n_steps)
        feature_2 = 0.2 * trend - 0.3 * s1 + 1.2 * np.roll(s2, -2) + 0.7 * exog + noise2

        df = pd.DataFrame({
            'feature_0': feature_0,
            'feature_1': feature_1,
            'feature_2': feature_2,
            'exog': exog
        })
        # keep only three features as per requirements; we will include exog optionally
        return df[['feature_0', 'feature_1', 'feature_2']]

# ---------------------------
# 2) Dataset wrapper for PyTorch
# ---------------------------
class TimeSeriesDataset(Dataset):
    """
    Prepares sliding windows for seq->one forecasting.
    Each sample: input_seq (seq_len x n_features) -> predict next target (scalar or multi-step)
    """
    def __init__(self, data: np.ndarray, seq_len: int = 60, horizon: int = 1, target_col: int = 0, scaler: StandardScaler = None):
        """
        data: numpy array shape (n_steps, n_features)
        seq_len: length of input history
        horizon: forecast horizon (1 = next step)
        target_col: which column to predict
        scaler: optional StandardScaler fitted on train data for scaling inputs
        """
        self.data = data
        self.seq_len = seq_len
        self.horizon = horizon
        self.target_col = target_col
        self.n_steps = data.shape[0]
        self.n_features = data.shape[1]
        self.indices = []
        # build valid start indices
        for start in range(0, self.n_steps - seq_len - horizon + 1):
            self.indices.append(start)
        self.scaler = scaler

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start = self.indices[idx]
        x = self.data[start: start + self.seq_len]
        y = self.data[start + self.seq_len + self.horizon - 1, self.target_col]
        if self.scaler is not None:
            # apply scaler to input (assume scaler fits features)
            x = self.scaler.transform(x)
            # target scaling not applied here; will be handled outside if needed
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        return x, y

# ---------------------------
# 3) Model: LSTM with additive attention (Bahdanau-style)
# ---------------------------
class BahdanauAttention(nn.Module):
    """Additive (Bahdanau) attention over encoder outputs with a query vector (e.g., last hidden state)."""
    def __init__(self, enc_hidden_dim, dec_hidden_dim, attn_dim):
        super().__init__()
        self.W_enc = nn.Linear(enc_hidden_dim, attn_dim, bias=False)
        self.W_dec = nn.Linear(dec_hidden_dim, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, encoder_outputs, decoder_hidden):
        """
        encoder_outputs: (batch, seq_len, enc_hidden_dim)
        decoder_hidden: (batch, dec_hidden_dim) -- usually last hidden state
        returns: context (batch, enc_hidden_dim), attn_weights (batch, seq_len)
        """
        # Apply linear layers
        # shape transforms:
        # W_enc(encoder_outputs) -> (batch, seq_len, attn_dim)
        # W_dec(decoder_hidden) -> (batch, 1, attn_dim)
        enc_proj = self.W_enc(encoder_outputs)  # (B, S, A)
        dec_proj = self.W_dec(decoder_hidden).unsqueeze(1)  # (B, 1, A)
        score = self.v(torch.tanh(enc_proj + dec_proj)).squeeze(-1)  # (B, S)
        attn_weights = torch.softmax(score, dim=1)  # (B, S)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)  # (B, enc_hidden_dim)
        return context, attn_weights

class LSTMWithAttention(nn.Module):
    def __init__(self, n_features: int, enc_hidden: int = 64, n_layers: int = 1, dropout: float = 0.1, attn_dim: int = 32, fc_hidden: int = 64):
        """
        Encoder-only LSTM with attention pooling for one-step forecast.
        We use encoder LSTM to produce outputs; use last hidden state as 'query' to attention,
        compute context vector, then combine and pass through FC to produce prediction.
        """
        super().__init__()
        self.n_features = n_features
        self.enc_hidden = enc_hidden
        self.n_layers = n_layers
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=enc_hidden, num_layers=n_layers, batch_first=True, dropout=dropout if n_layers>1 else 0.0, bidirectional=False)
        self.attention = BahdanauAttention(enc_hidden_dim=enc_hidden, dec_hidden_dim=enc_hidden, attn_dim=attn_dim)
        self.fc = nn.Sequential(
            nn.Linear(enc_hidden + enc_hidden, fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, 1)
        )

    def forward(self, x):
        """
        x: (batch, seq_len, n_features)
        returns: pred (batch,), attn_weights (batch, seq_len)
        """
        outputs, (h_n, c_n) = self.lstm(x)  # outputs: (B, S, enc_hidden), h_n: (num_layers, B, enc_hidden)
        # use last layer's hidden state as decoder_hidden
        decoder_hidden = h_n[-1]  # (B, enc_hidden)
        context, attn_weights = self.attention(outputs, decoder_hidden)  # (B, enc_hidden), (B, S)
        # combine context and decoder hidden
        combined = torch.cat([context, decoder_hidden], dim=1)  # (B, 2*enc_hidden)
        out = self.fc(combined).squeeze(-1)  # (B,)
        return out, attn_weights

# ---------------------------
# 4) Baseline models
# ---------------------------
class MLPBaseline(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: List[int] = [128, 64], dropout: float = 0.1):
        super().__init__()
        layers = []
        in_dim = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x shape (batch, seq_len, features) -> flatten
        x = x.view(x.size(0), -1)
        return self.net(x).squeeze(-1)

def sarima_forecast(train_series: np.ndarray, test_series: np.ndarray, order=(1,1,1), seasonal_order=(0,1,1,30)):
    """
    Train SARIMA on the single target series (univariate) and forecast test length.
    Returns predictions aligned to test_series length.
    """
    # We fit on train_series and forecast len(test_series)
    model = sm.tsa.statespace.SARIMAX(train_series, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)
    preds = res.forecast(steps=len(test_series))
    return np.array(preds)

# ---------------------------
# 5) Trainer & Evaluator
# ---------------------------
class Trainer:
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, criterion, device=DEVICE):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train_epoch(self, loader: DataLoader):
        self.model.train()
        total_loss = 0.0
        n = 0
        for x, y in loader:
            x = x.to(self.device)
            y = y.to(self.device)
            self.optimizer.zero_grad()
            pred, _ = self.model(x)
            loss = self.criterion(pred, y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * x.size(0)
            n += x.size(0)
        return total_loss / n

    def validate(self, loader: DataLoader):
        self.model.eval()
        total_loss = 0.0
        n = 0
        preds = []
        trues = []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)
                pred, _ = self.model(x)
                loss = self.criterion(pred, y)
                total_loss += loss.item() * x.size(0)
                n += x.size(0)
                preds.append(pred.cpu().numpy())
                trues.append(y.cpu().numpy())
        preds = np.concatenate(preds)
        trues = np.concatenate(trues)
        return total_loss / n, preds, trues

    def predict_with_attention(self, loader: DataLoader):
        """
        Returns predictions, true values, and attention weights for each batch.
        attn weights concatenated in order.
        """
        self.model.eval()
        preds = []
        trues = []
        attn_all = []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)
                pred, attn = self.model(x)
                preds.append(pred.cpu().numpy())
                trues.append(y.cpu().numpy())
                attn_all.append(attn.cpu().numpy())
        return np.concatenate(preds), np.concatenate(trues), np.concatenate(attn_all)

# ---------------------------
# 6) Hyperparameter search (Optuna optional) and training pipeline
# ---------------------------
def train_and_evaluate(params: Dict[str, Any], data_train: np.ndarray, data_val: np.ndarray, seq_len: int, horizon: int, target_col: int, scaler: StandardScaler, num_epochs: int = 50, batch_size: int = 64, patience: int = 8):
    """
    Trains an LSTMWithAttention model with the provided hyperparameters and returns validation RMSE (lower is better)
    Also returns the trained model.
    params keys: enc_hidden, n_layers, dropout, attn_dim, fc_hidden, lr
    """
    set_seed(SEED)
    # Build datasets
    train_ds = TimeSeriesDataset(data=data_train, seq_len=seq_len, horizon=horizon, target_col=target_col, scaler=scaler)
    val_ds = TimeSeriesDataset(data=data_val, seq_len=seq_len, horizon=horizon, target_col=target_col, scaler=scaler)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = LSTMWithAttention(n_features=data_train.shape[1],
                              enc_hidden=params['enc_hidden'],
                              n_layers=params['n_layers'],
                              dropout=params['dropout'],
                              attn_dim=params['attn_dim'],
                              fc_hidden=params['fc_hidden']).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
    criterion = nn.MSELoss()
    trainer = Trainer(model, optimizer, criterion, device=DEVICE)

    best_val = float('inf')
    best_state = None
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        train_loss = trainer.train_epoch(train_loader)
        val_loss, _, _ = trainer.validate(val_loader)
        val_rmse = math.sqrt(val_loss)
        # simple early stopping on val_loss
        if val_rmse < best_val - 1e-6:
            best_val = val_rmse
            best_state = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    # load best state
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_val, model

def hp_search_random(data_train, data_val, seq_len, horizon, target_col, scaler, n_trials=20):
    """
    Randomized search (fallback) for hyperparameters.
    """
    param_space = {
        'enc_hidden': [32, 64, 128],
        'n_layers': [1, 2],
        'dropout': [0.0, 0.1, 0.2, 0.3],
        'attn_dim': [16, 32, 64],
        'fc_hidden': [32, 64, 128],
        'lr': [1e-3, 5e-4, 1e-4]
    }
    def sample():
        return {k: random.choice(v) for k, v in param_space.items()}

    best = (float('inf'), None, None)
    for i in range(n_trials):
        params = sample()
        print(f"[RandomSearch] Trial {i+1}/{n_trials} params: {params}")
        val_rmse, model = train_and_evaluate(params, data_train, data_val, seq_len, horizon, target_col, scaler, num_epochs=60, batch_size=64, patience=8)
        print(f" -> val_rmse: {val_rmse:.4f}")
        if val_rmse < best[0]:
            best = (val_rmse, params, model)
    return best  # (best_rmse, best_params, best_model)

def hp_search_optuna(data_train, data_val, seq_len, horizon, target_col, scaler, n_trials=30):
    """
    Bayesian optimization with Optuna (if available).
    """
    def objective(trial):
        params = {
            'enc_hidden': trial.suggest_categorical('enc_hidden', [32, 64, 128]),
            'n_layers': trial.suggest_categorical('n_layers', [1, 2]),
            'dropout': trial.suggest_float('dropout', 0.0, 0.4, step=0.1),
            'attn_dim': trial.suggest_categorical('attn_dim', [16, 32, 64]),
            'fc_hidden': trial.suggest_categorical('fc_hidden', [32, 64, 128]),
            'lr': trial.suggest_loguniform('lr', 1e-4, 1e-2)
        }
        val_rmse, _ = train_and_evaluate(params, data_train, data_val, seq_len, horizon, target_col, scaler, num_epochs=50, batch_size=64, patience=8)
        return val_rmse

    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials)
    best_params = study.best_trial.params
    # convert float choices to int for certain keys
    best_params['enc_hidden'] = int(best_params['enc_hidden'])
    best_params['n_layers'] = int(best_params['n_layers'])
    best_params['attn_dim'] = int(best_params['attn_dim'])
    best_params['fc_hidden'] = int(best_params['fc_hidden'])
    # train final model with best params
    best_rmse, best_model = train_and_evaluate(best_params, data_train, data_val, seq_len, horizon, target_col, scaler, num_epochs=100, batch_size=64, patience=12)
    return best_rmse, best_params, best_model

# ---------------------------
# 7) Full pipeline: prepare splits, scalers, run search, evaluate baselines
# ---------------------------
def prepare_data(df: pd.DataFrame, seq_len=60, horizon=1, test_size=0.2, val_size=0.1, target_col=0, scale=True, random_state=SEED):
    """
    Splits the dataframe into train/val/test chronologically (no leakage).
    Returns numpy arrays for train/val/test, fitted scaler for features,
    and raw target arrays for baseline (for SARIMA).
    """
    n = len(df)
    test_n = int(n * test_size)
    val_n = int(n * val_size)
    train_n = n - test_n - val_n
    train_df = df.iloc[:train_n]
    val_df = df.iloc[train_n:train_n+val_n]
    test_df = df.iloc[train_n+val_n:]

    scaler = None
    if scale:
        scaler = StandardScaler()
        scaler.fit(train_df.values)

    return train_df.values, val_df.values, test_df.values, scaler

def run_full_experiment(seed=SEED, seq_len=60, horizon=1, target_col=0, use_optuna=False):
    set_seed(seed)
    # 1) data generation
    synth = SyntheticMultivariateSeries(n_steps=4000, seed=seed)
    df = synth.generate()
    print("Generated data shape:", df.shape)
    # 2) split and scaler
    data_train, data_val, data_test, scaler = prepare_data(df, seq_len=seq_len, horizon=horizon, test_size=0.2, val_size=0.1, target_col=target_col, scale=True, random_state=seed)
    # We'll also prepare full arrays for dataset creation (train contains both train+val when training final model if desired)
    print(f"Train/Val/Test sizes: {len(data_train)}/{len(data_val)}/{len(data_test)}")
    # 3) hyperparameter search
    if use_optuna and OPTUNA_AVAILABLE:
        print("Running Optuna hyperparameter search...")
        best_rmse, best_params, best_model = hp_search_optuna(data_train, data_val, seq_len, horizon, target_col, scaler, n_trials=30)
    else:
        print("Running randomized hyperparameter search...")
        best_rmse, best_params, best_model = hp_search_random(data_train, data_val, seq_len, horizon, target_col, scaler, n_trials=20)

    print("Best RMSE (val):", best_rmse)
    print("Best params:", best_params)

    # 4) Evaluate on test set and collect attention
    # create DataLoaders for test
    test_ds = TimeSeriesDataset(data_test, seq_len=seq_len, horizon=horizon, target_col=target_col, scaler=scaler)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    trainer = Trainer(best_model, optimizer=None, criterion=None, device=DEVICE)  # optimizer not needed for predict
    preds, trues, attn_weights = trainer.predict_with_attention(test_loader)

    # Because we scaled inputs only, the target is still in original scale (we didn't scale target), so predictions are in same scale.
    test_mae = mean_absolute_error(trues, preds)
    test_rmse = rmse(trues, preds)
    test_mape = mape(trues, preds)
    print(f"Test metrics - MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}, MAPE: {test_mape:.2f}%")

    results = {
        'model': best_model,
        'best_params': best_params,
        'test_metrics': {'mae': test_mae, 'rmse': test_rmse, 'mape': test_mape},
        'preds': preds,
        'trues': trues,
        'attn_weights': attn_weights,
        'test_df': data_test
    }

    # 5) Baselines
    # SARIMA: fit on training+validation concatenated target series, forecast for test length (univariate)
    # For SARIMA we must build the univariate series (target_col)
    full_series = np.vstack([data_train, data_val, data_test])[:, target_col]
    train_val_series = full_series[:len(data_train) + len(data_val)]
    test_series = full_series[len(data_train) + len(data_val):]

    # naive SARIMA orders (user may tune further). We'll choose (1,1,1)x(0,1,1,30) to capture seasonality ~30.
    try:
        sarima_preds = sarima_forecast(train_val_series, test_series, order=(1,1,1), seasonal_order=(0,1,1,30))
        sarima_rmse = rmse(test_series, sarima_preds)
        sarima_mae = mean_absolute_error(test_series, sarima_preds)
        sarima_mape = mape(test_series, sarima_preds)
    except Exception as e:
        print("SARIMA failed:", e)
        sarima_preds = np.full_like(test_series, np.nan)
        sarima_rmse = sarima_mae = sarima_mape = float('nan')

    print(f"SARIMA metrics - MAE: {sarima_mae:.4f}, RMSE: {sarima_rmse:.4f}, MAPE: {sarima_mape:.2f}%")

    # MLP baseline: train simple MLP with flattened input
    # Reuse datasets but with MLPBaseline
    mlp = MLPBaseline(input_size=seq_len * data_train.shape[1], hidden_sizes=[128, 64], dropout=0.1).to(DEVICE)
    criterion = nn.MSELoss()
    opt = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    train_ds = TimeSeriesDataset(data_train, seq_len=seq_len, horizon=horizon, target_col=target_col, scaler=scaler)
    val_ds = TimeSeriesDataset(data_val, seq_len=seq_len, horizon=horizon, target_col=target_col, scaler=scaler)
    test_ds = TimeSeriesDataset(data_test, seq_len=seq_len, horizon=horizon, target_col=target_col, scaler=scaler)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    best_val = float('inf')
    best_state = None
    epochs_no_improve = 0
    for epoch in range(60):
        # train
        mlp.train()
        for x, y in train_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            opt.zero_grad()
            pred = mlp(x)
            loss = criterion(pred, y)
            loss.backward()
            opt.step()
        # validate
        mlp.eval()
        vals = []
        trues_val = []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                pred = mlp(x)
                vals.append(((pred - y) ** 2).cpu().numpy())
                trues_val.append(y.cpu().numpy())
        val_rmse = np.sqrt(np.mean(np.concatenate(vals)))
        if val_rmse < best_val - 1e-6:
            best_val = val_rmse
            best_state = {k: v.cpu() for k, v in mlp.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= 8:
                break
    if best_state is not None:
        mlp.load_state_dict(best_state)
    # test mlp
    preds_mlp = []
    trues_mlp = []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)
            pred = mlp(x).cpu().numpy()
            preds_mlp.append(pred)
            trues_mlp.append(y.numpy())
    preds_mlp = np.concatenate(preds_mlp)
    trues_mlp = np.concatenate(trues_mlp)
    mlp_rmse = rmse(trues_mlp, preds_mlp)
    mlp_mae = mean_absolute_error(trues_mlp, preds_mlp)
    mlp_mape = mape(trues_mlp, preds_mlp)
    print(f"MLP baseline metrics - MAE: {mlp_mae:.4f}, RMSE: {mlp_rmse:.4f}, MAPE: {mlp_mape:.2f}%")

    results['baselines'] = {
        'sarima': {'preds': sarima_preds, 'mae': sarima_mae, 'rmse': sarima_rmse, 'mape': sarima_mape},
        'mlp': {'preds': preds_mlp, 'mae': mlp_mae, 'rmse': mlp_rmse, 'mape': mlp_mape}
    }

    return results, df

# ---------------------------
# 8) Visualization helpers
# ---------------------------
def plot_predictions(trues, preds, title="Predictions vs True", n_plot=200, savepath=None):
    n = min(n_plot, len(trues))
    t = np.arange(n)
    plt.figure(figsize=(12,4))
    plt.plot(t, trues[:n], label="True")
    plt.plot(t, preds[:n], label="Predicted")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath)
    else:
        plt.show()
    plt.close()

def visualize_attention_for_sample(input_sequence: np.ndarray, attn_weights: np.ndarray, feature_names: List[str] = None, savepath=None, title=None):
    """
    input_sequence: (seq_len, n_features) original (unscaled) or scaled values for a single sample
    attn_weights: (seq_len,) attention weights (must sum to 1)
    """
    seq_len, n_features = input_sequence.shape
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(n_features)]
    # Plot attention as a heatmap per feature over time: multiply each feature time series by attention weights (broadcast)
    weighted = input_sequence * attn_weights[:, None]  # (S, F)
    # create figure showing attention weights bar and weighted signals overlay
    fig, axs = plt.subplots(2, 1, figsize=(12, 6), gridspec_kw={'height_ratios':[1,2]})
    axs[0].bar(np.arange(seq_len), attn_weights)
    axs[0].set_title('Attention weights over input time steps')
    axs[0].set_xlabel('time step (relative, most recent at right)')
    axs[0].set_ylabel('weight')

    for i in range(n_features):
        axs[1].plot(np.arange(seq_len), input_sequence[:, i], label=feature_names[i], alpha=0.8)
    axs[1].set_title('Input features (original scale)')
    axs[1].legend(ncol=min(n_features, 4))
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath)
        plt.close()
    else:
        plt.show()

# ---------------------------
# 9) Main: run experiment and produce artifacts
# ---------------------------
if __name__ == "__main__":
    # Parameters (user can modify these)
    SEED = 42
    seq_len = 120          # history window length
    horizon = 1            # one-step ahead forecasting
    target_col = 0
    use_optuna = OPTUNA_AVAILABLE  # set False to force randomized search
    set_seed(SEED)

    # Run experiment
    results, df = run_full_experiment(seed=SEED, seq_len=seq_len, horizon=horizon, target_col=target_col, use_optuna=use_optuna)

    # Print final report (summary)
    print("\n=== EXPERIMENT SUMMARY ===")
    print("Best model params:", results['best_params'])
    print("Test metrics (LSTM+Attn):", results['test_metrics'])
    print("MLP baseline metrics:", results['baselines']['mlp']['mae'], results['baselines']['mlp']['rmse'], results['baselines']['mlp']['mape'])
    print("SARIMA baseline metrics:", results['baselines']['sarima']['mae'], results['baselines']['sarima']['rmse'], results['baselines']['sarima']['mape'])

    # Save some artifacts locally
    os.makedirs("artifacts", exist_ok=True)
    # 1) Plot sample predictions
    plot_predictions(results['trues'], results['preds'], title="LSTM-Attn Predictions vs True (test subset)", n_plot=400, savepath="artifacts/predictions_lstm_attn.png")
    plot_predictions(results['trues'], results['baselines']['mlp']['preds'], title="MLP Predictions vs True (test subset)", n_plot=400, savepath="artifacts/predictions_mlp.png")
    if not np.all(np.isnan(results['baselines']['sarima']['preds'])):
        plot_predictions(results['trues'], results['baselines']['sarima']['preds'], title="SARIMA Predictions vs True (test subset)", n_plot=400, savepath="artifacts/predictions_sarima.png")

    # 2) Attention visualization for a few test samples
    # Note: attn_weights array shape (num_test_samples, seq_len)
    attn = results['attn_weights']
    # We also want the corresponding input sequences (unscaled) for the test set to plot real values
    # Build test dataset without scaler to access original raw input sequences (for few examples)
    synth = SyntheticMultivariateSeries(n_steps=4000, seed=SEED)
    full_df = synth.generate()
    # We must index into the test portion where test_ds indices start after train+val
    n = len(full_df)
    test_n = int(n * 0.2)
    val_n = int(n * 0.1)
    train_n = n - test_n - val_n
    raw_test_df = full_df.iloc[train_n+val_n:].values  # shape: (test_n, n_features)
    # For each sample in test DataLoader, the corresponding input sequence uses seq_len rows prior to the target index.
    # We'll visualize attention for the first 3 test samples
    num_samples_to_plot = min(3, attn.shape[0])
    example_dir = "artifacts/attention_examples"
    os.makedirs(example_dir, exist_ok=True)
    for i in range(num_samples_to_plot):
        # The ith sample in the dataset corresponds to input rows [i : i+seq_len] of raw_test_df
        # because TimeSeriesDataset indices start at 0 for the test partition
        input_seq = raw_test_df[i: i+seq_len]  # (seq_len, n_features)
        # attn weights for sample i:
        attn_i = attn[i]  # (seq_len,)
        # Normalize (should already sum to 1)
        attn_i = attn_i / (attn_i.sum() + 1e-12)
        plot_path = os.path.join(example_dir, f"attn_sample_{i}.png")
        visualize_attention_for_sample(input_seq, attn_i, feature_names=list(full_df.columns), savepath=plot_path)
        print(f"Saved attention visualization for sample {i} to {plot_path}")

    print("Artifacts saved in ./artifacts (plots, attention visualizations).")
    print("End of script.")
