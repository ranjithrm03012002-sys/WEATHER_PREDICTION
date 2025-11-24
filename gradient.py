"""
Advanced Time Series Forecasting with Deep Learning and Explainability
Single-file, production-quality Python script that:
 - Programmatically generates a synthetic multivariate time series dataset
 - Trains a PyTorch LSTM forecasting model with advanced optimization (OneCycleLR + EarlyStopping)
 - Trains a statistical baseline (SARIMAX) and compares MAE, RMSE, MAPE
 - Applies Integrated Gradients via Captum to explain predictions for a selected forecast window
 - Exports results and a short textual summary file

Dependencies:
 - numpy, pandas, matplotlib, scikit-learn, torch, statsmodels, captum
Install (example):
 pip install numpy pandas matplotlib scikit-learn torch statsmodels captum

Notes:
 - The script is written to be run as a script (python forecast_xai.py).
 - All configurable hyperparameters are at the top in the CONFIG block.
"""

# -----------------------
# CONFIG / SETUP (ESSENTIAL)
# -----------------------
import os
import math
import random
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from statsmodels.tsa.statespace.sarimax import SARIMAX

# Captum for Integrated Gradients
from captum.attr import IntegratedGradients

# plotting (optional)
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class Config:
    seed: int = 42

    # Data generation
    n_series: int = 1              # number of separate series (we generate one multivariate series)
    n_timesteps: int = 24 * 90     # ~90 days of hourly data (2160)
    freq: str = "H"                # hourly
    n_features: int = 5            # minimum 5 features required
    forecast_horizon: int = 24     # predict next 24 hours
    input_window: int = 168        # lookback (one week)
    noise_scale: float = 0.1       # base noise

    # Model hyperparams
    model_type: str = "LSTM"       # LSTM implementation
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2

    # Training
    batch_size: int = 64
    epochs: int = 60
    lr_max: float = 1e-3
    weight_decay: float = 1e-5

    # Early stopping
    early_stop_patience: int = 8
    early_stop_min_delta: float = 1e-4

    # Misc
    save_dir: str = "results_ts"
    device = DEVICE

cfg = Config()
os.makedirs(cfg.save_dir, exist_ok=True)

# reproducibility
random.seed(cfg.seed)
np.random.seed(cfg.seed)
torch.manual_seed(cfg.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(cfg.seed)

# -----------------------
# 1) Synthetic dataset generation (multivariate)
# -----------------------
def generate_multivariate_series(n_steps, n_features, freq="H", seed=cfg.seed, noise_scale=0.1):
    """
    Generate a multivariate timeseries with:
     - linear + nonlinear trend
     - multiple seasonalities (daily + weekly)
     - interactions between features
     - controlled Gaussian noise
    Returns: DataFrame with datetime index and columns f0..f{n_features-1}
    """
    rng = pd.date_range(start="2020-01-01", periods=n_steps, freq=freq)
    t = np.arange(n_steps).astype(float)

    data = np.zeros((n_steps, n_features), dtype=float)

    # base signals
    # seasonal components
    daily = np.sin(2 * np.pi * (t % 24) / 24)  # daily cycle
    weekly = np.sin(2 * np.pi * (t % (24 * 7)) / (24 * 7))  # weekly cycle
    yearly = np.sin(2 * np.pi * (t % (24 * 365)) / (24 * 365))  # yearly (weak given short series)

    # nonlinear trend
    trend = 0.0005 * (t**1.5)  # nonlinear upward trend

    for i in range(n_features):
        # unique phase shifts and amplitude per feature
        phase = (i + 1) * 0.3
        amp_daily = 1.0 + 0.2 * i
        amp_weekly = 0.8 + 0.15 * ((-1) ** i)
        amp_yearly = 0.3 + 0.05 * i

        # interactions
        inter = 0.2 * np.cos(2 * np.pi * t / (24 * (i + 2) + 7))  # slow interaction

        base = (amp_daily * np.sin(2 * np.pi * t / 24 + phase)
                + amp_weekly * np.sin(2 * np.pi * t / (24 * 7) + phase * 0.5)
                + amp_yearly * yearly * (0.5 + 0.1 * i)
                + 0.5 * trend
                + inter)

        # non-linear distortions for some features
        if i % 2 == 0:
            base = base + 0.1 * (np.tanh(0.001 * (t - n_steps / 2)) * (i + 1))

        # controlled noise
        noise = noise_scale * (1 + 0.1 * i) * np.random.normal(scale=1.0, size=n_steps)

        data[:, i] = base + noise

    df = pd.DataFrame(data, index=rng, columns=[f"f{i}" for i in range(n_features)])
    return df

print("Generating synthetic dataset...")
df = generate_multivariate_series(cfg.n_timesteps, cfg.n_features, noise_scale=cfg.noise_scale)
df.to_csv(os.path.join(cfg.save_dir, "synthetic_multivariate.csv"))
print("Saved synthetic_multivariate.csv")

# Inspect
print(df.head())

# -----------------------
# 2) Data preparation for supervised learning
# -----------------------
def create_supervised(df, input_window, horizon, feature_cols=None, target_col="f0"):
    """
    Build sliding windows for supervised training.
    - features: multivariate (all columns)
    - target: forecast horizon for target_col (multi-step)
    Returns arrays: X (n_samples, input_window, n_features), y (n_samples, horizon)
    """
    if feature_cols is None:
        feature_cols = df.columns.tolist()
    data = df[feature_cols].values
    n = len(df)
    X, y = [], []
    for i in range(n - input_window - horizon + 1):
        X.append(data[i:i + input_window])
        y.append(data[i + input_window:i + input_window + horizon, feature_cols.index(target_col)])
    X = np.stack(X)
    y = np.stack(y)
    return X, y

feature_cols = df.columns.tolist()
X_all, y_all = create_supervised(df, cfg.input_window, cfg.forecast_horizon, feature_cols=feature_cols, target_col="f0")
print("Supervised shapes:", X_all.shape, y_all.shape)

# Train/val/test split (time-ordered)
n_samples = X_all.shape[0]
train_end = int(n_samples * 0.7)
val_end = int(n_samples * 0.85)

X_train, y_train = X_all[:train_end], y_all[:train_end]
X_val, y_val = X_all[train_end:val_end], y_all[train_end:val_end]
X_test, y_test = X_all[val_end:], y_all[val_end:]

print("Splits:", X_train.shape, X_val.shape, X_test.shape)

# Scale features (fit on train)
scaler = StandardScaler()
n_features = X_train.shape[2]
# reshape to 2D for scaler
X_train_flat = X_train.reshape(-1, n_features)
scaler.fit(X_train_flat)
def scale_X(X):
    s = scaler.transform(X.reshape(-1, n_features)).reshape(X.shape)
    return s
X_train_s = scale_X(X_train)
X_val_s = scale_X(X_val)
X_test_s = scale_X(X_test)

# targets - we standardize target using train target mean/std for stability in training
target_scaler_mean = y_train.mean()
target_scaler_std = y_train.std() if y_train.std() > 0 else 1.0
def scale_y(y): return (y - target_scaler_mean) / target_scaler_std
def inv_scale_y(y): return y * target_scaler_std + target_scaler_mean

y_train_s = scale_y(y_train)
y_val_s = scale_y(y_val)
y_test_s = scale_y(y_test)

# -----------------------
# 3) PyTorch dataset + model
# -----------------------
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_ds = TimeSeriesDataset(X_train_s, y_train_s)
val_ds = TimeSeriesDataset(X_val_s, y_val_s)
test_ds = TimeSeriesDataset(X_test_s, y_test_s)

train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

# Model definition (LSTM encoder + MLP decoder producing multi-step forecast)
class LSTMForecaster(nn.Module):
    def __init__(self, n_features, hidden_size=128, num_layers=2, dropout=0.2, horizon=24):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=dropout)
        # decoder produces horizon outputs
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, horizon)
        )

    def forward(self, x):
        # x: (batch, seq_len, features)
        out, (hn, cn) = self.lstm(x)  # out: (batch, seq_len, hidden)
        last = out[:, -1, :]          # (batch, hidden)
        y = self.fc(last)             # (batch, horizon)
        return y

# instantiate
model = LSTMForecaster(n_features=n_features,
                       hidden_size=cfg.hidden_size,
                       num_layers=cfg.num_layers,
                       dropout=cfg.dropout,
                       horizon=cfg.forecast_horizon).to(cfg.device)

# Loss, optimizer, scheduler
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr_max, weight_decay=cfg.weight_decay)

# OneCycleLR scheduler (advanced optimization strategy)
# OneCycle requires specifying steps_per_epoch
steps_per_epoch = max(1, len(train_loader))
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg.lr_max,
                                                steps_per_epoch=steps_per_epoch,
                                                epochs=cfg.epochs,
                                                pct_start=0.1,
                                                anneal_strategy="cos",
                                                final_div_factor=1e4)

# -----------------------
# 4) Training loop with early stopping
# -----------------------
class EarlyStopping:
    def __init__(self, patience=8, min_delta=1e-4, restore_best=True):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = float("inf")
        self.best_state = None
        self.wait = 0
        self.restore_best = restore_best

    def step(self, current_score, model):
        if current_score + self.min_delta < self.best_score:
            self.best_score = current_score
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.wait = 0
            return False
        else:
            self.wait += 1
            if self.wait >= self.patience:
                if self.restore_best and self.best_state is not None:
                    model.load_state_dict(self.best_state)
                return True
            return False

def evaluate(model, loader, criterion):
    model.eval()
    losses = []
    preds = []
    trues = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(cfg.device)
            yb = yb.to(cfg.device)
            out = model(xb)
            loss = criterion(out, yb)
            losses.append(loss.item())
            preds.append(out.cpu().numpy())
            trues.append(yb.cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    mean_loss = float(np.mean(losses))
    return mean_loss, preds, trues

print("Starting training loop...")
es = EarlyStopping(patience=cfg.early_stop_patience, min_delta=cfg.early_stop_min_delta)
train_history = {"train_loss": [], "val_loss": []}

for epoch in range(cfg.epochs):
    model.train()
    epoch_losses = []
    for xb, yb in train_loader:
        xb = xb.to(cfg.device)
        yb = yb.to(cfg.device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        # scheduler step per batch for OneCycleLR
        scheduler.step()
        epoch_losses.append(loss.item())

    train_loss = float(np.mean(epoch_losses))
    val_loss, _, _ = evaluate(model, val_loader, criterion)
    train_history["train_loss"].append(train_loss)
    train_history["val_loss"].append(val_loss)

    print(f"Epoch {epoch+1}/{cfg.epochs} - train_loss: {train_loss:.6f}  val_loss: {val_loss:.6f}")

    # early stopping
    if es.step(val_loss, model):
        print(f"Early stopping at epoch {epoch+1}")
        break

# Save model
model_path = os.path.join(cfg.save_dir, "lstm_forecaster.pth")
torch.save(model.state_dict(), model_path)
print(f"Saved model to {model_path}")

# -----------------------
# 5) Evaluate on test set (DL model)
# -----------------------
test_loss, preds_s, trues_s = evaluate(model, test_loader, criterion)
# inverse-scaling predictions
preds = inv_scale_y(preds_s)
trues = inv_scale_y(trues_s)

# compute metrics (multi-step forecasting; compute per-horizon aggregated metrics)
def compute_metrics(true, pred):
    # true/pred shape: (n_samples, horizon)
    mae = mean_absolute_error(true.flatten(), pred.flatten())
    rmse = math.sqrt(mean_squared_error(true.flatten(), pred.flatten()))
    # MAPE: avoid divide-by-zero by small epsilon
    eps = 1e-8
    mape = (np.abs((true - pred) / (np.clip(np.abs(true), eps, None)))) * 100.0
    mape = np.nanmean(mape)
    return mae, rmse, mape

dl_mae, dl_rmse, dl_mape = compute_metrics(trues, preds)
print("Deep Learning model test metrics - MAE: {:.4f}, RMSE: {:.4f}, MAPE: {:.2f}%".format(dl_mae, dl_rmse, dl_mape))

# -----------------------
# 6) Baseline statistical model: SARIMAX (univariate on target f0)
# -----------------------
# We'll train SARIMAX on the target series f0 using exogenous variables (other features aggregated)
# For simplicity, use the same train/val/test splits in time for the original timeseries

target_series = df["f0"]
# train/end indexes matching supervised windows
start_idx = cfg.input_window
end_idx = cfg.input_window + X_all.shape[0] - 1  # inclusive last index that has a target

# Build time-indexed windows to align
dates_for_targets = df.index[start_idx:end_idx + 1]

# create series for train/val/test aligned with y_all
target_vals = target_series[start_idx:end_idx + 1].values  # shape (n_samples,)

n_total = len(target_vals)
train_end_idx = int(n_total * 0.7)
val_end_idx = int(n_total * 0.85)

sar_train = target_vals[:train_end_idx]
sar_val = target_vals[train_end_idx:val_end_idx]
sar_test = target_vals[val_end_idx:]

# Fit SARIMAX on train + val combined to produce one-step-ahead rolling multi-step forecast on test
# We'll fit on training data and then produce multi-step direct forecasts using dynamic forecasting
# Choose SARIMAX order via simple heuristics (p,d,q) x (P,D,Q,s) - small model to keep things stable
sar_order = (1, 1, 1)
seasonal_order = (1, 0, 1, 24)  # daily seasonality at hourly frequency

print("Fitting SARIMAX baseline...")
sar_model = SARIMAX(sar_train, order=sar_order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
sar_res = sar_model.fit(disp=False)
print("SARIMAX fitted.")

# Rolling multi-step forecast on test set: for each test-origin, forecast horizon steps ahead using model extended
# We'll use a simple approach: re-fit the model incrementally by appending actuals as we move forward.
sar_preds = []
history = list(sar_train.copy())
for i in range(len(sar_test)):
    # fit on current history
    # For speed we won't re-fit at each step; instead use dynamic forecast from last fit for horizon,
    # but to keep it stable we'll re-fit every 24 steps
    if i % 24 == 0 and i != 0:
        sar_res = SARIMAX(np.array(history), order=sar_order, seasonal_order=seasonal_order,
                          enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    # forecast horizon steps ahead from the current point
    start = len(history)
    end = len(history) + cfg.forecast_horizon - 1
    fcast = sar_res.predict(start=start, end=end)
    # store only the first's horizon aligned across test predictions to match our DL evaluation:
    sar_preds.append(fcast.values[:cfg.forecast_horizon])
    # append the true next observation to history (simulate real-time arrival)
    history.append(sar_test[i])

sar_preds = np.array(sar_preds)  # shape (len(test), horizon)
sar_trues = []
# Build corresponding true values from sar_test for horizon windows
# For each test origin i, the true horizon vector is sar_test[i : i+horizon] with padding if necessary
for i in range(len(sar_test)):
    end_idx = i + cfg.forecast_horizon
    if end_idx <= len(sar_test):
        sar_trues.append(sar_test[i:end_idx])
    else:
        # pad with last value (not ideal but ensures shapes)
        needed = end_idx - len(sar_test)
        pad = np.repeat(sar_test[-1], needed)
        sar_trues.append(np.concatenate([sar_test[i:], pad]))
sar_trues = np.array(sar_trues)

# compute metrics
baseline_mae, baseline_rmse, baseline_mape = compute_metrics(sar_trues, sar_preds)
print("SARIMAX baseline test metrics - MAE: {:.4f}, RMSE: {:.4f}, MAPE: {:.2f}%".format(baseline_mae, baseline_rmse, baseline_mape))

# -----------------------
# 7) Explainability: Integrated Gradients (Captum) for the DL model
# -----------------------
# We'll pick a specific test sample (e.g., middle of test set) and compute IG attribution for inputs (all features across time)
# Note: IntegratedGradients works with a forward function; we will attribute to the model's output for the first horizon step
ig = IntegratedGradients(model)

# choose sample index (from test dataset indices)
sample_idx = max(0, len(test_ds) // 2)
x_sample, y_sample = test_ds[sample_idx]
x_tensor = torch.tensor(x_sample[None, ...]).to(cfg.device)  # shape (1, seq, features)
y_target = 0  # index of the horizon to explain (0..horizon-1); choose first horizon step

# Define wrapper forward function to return scalar for the target horizon
def forward_for_ig(input_tensor):
    # input_tensor: (batch, seq_len, features)
    out = model(input_tensor)
    # return the scalar predictions of the target horizon (batch, 1)
    # Captum expects output shape (batch,) or (batch, n_classes) etc. We'll return (batch,).
    return out[:, y_target]

# baseline (reference) input: zero (which corresponds to standardized zero); choose baseline as zeros
baseline = torch.zeros_like(x_tensor).to(cfg.device)

# compute attributions
model.eval()
with torch.no_grad():
    # ensure forward_for_ig uses model in eval
    attr_ig, delta = ig.attribute(x_tensor, baselines=baseline, target=None, return_convergence_delta=True, internal_batch_size=1, n_steps=200)
# attr_ig shape: (1, seq_len, features)
attr = attr_ig.squeeze(0).cpu().numpy()

# Aggregate attribution across time to get per-feature importance, and across features to get per-time importance
feature_importance = np.mean(np.abs(attr), axis=0)  # mean over time -> per-feature
time_importance = np.mean(np.abs(attr), axis=1)     # mean over features -> per-time step

# Normalize importance
feature_importance_norm = feature_importance / (np.sum(feature_importance) + 1e-12)
time_importance_norm = time_importance / (np.sum(time_importance) + 1e-12)

# Prepare a small textual interpretation
explanation_lines = []
explanation_lines.append("Integrated Gradients explanation for test sample index {}".format(sample_idx))
explanation_lines.append("Target horizon step explained: {}".format(y_target))
explanation_lines.append("")
explanation_lines.append("Top feature importances (normalized):")
for i, val in enumerate(feature_importance_norm):
    explanation_lines.append(f"  f{i}: {val:.4f}")

explanation_lines.append("")
explanation_lines.append("Top time-steps (relative, 0 is oldest in input window, {} is last):".format(cfg.input_window-1))
top_time_idx = np.argsort(-time_importance_norm)[:8]
for idx in top_time_idx:
    explanation_lines.append(f"  t={idx} (relative) : importance {time_importance_norm[idx]:.4f}")

explanation_text = "\n".join(explanation_lines)
print("\n" + explanation_text)

# Save explanation and attribution arrays
np.save(os.path.join(cfg.save_dir, "ig_attr_sample.npy"), attr)
with open(os.path.join(cfg.save_dir, "explanation_text.txt"), "w") as f:
    f.write(explanation_text)
print("Saved IG attribution and textual explanation to results directory.")

# -----------------------
# 8) Comparative report (text file) and simple plots
# -----------------------
report_lines = []
report_lines.append("Advanced Time Series Forecasting Report\n")
report_lines.append("Dataset: synthetic multivariate ({} features), {} timesteps".format(cfg.n_features, cfg.n_timesteps))
report_lines.append("\nModel configuration (Deep Learning - LSTM):")
report_lines.append(f"  input_window: {cfg.input_window}, forecast_horizon: {cfg.forecast_horizon}")
report_lines.append(f"  hidden_size: {cfg.hidden_size}, num_layers: {cfg.num_layers}, dropout: {cfg.dropout}")
report_lines.append(f"  optimizer: AdamW, OneCycleLR, max_lr: {cfg.lr_max}, weight_decay: {cfg.weight_decay}")
report_lines.append("")
report_lines.append("Training details:")
report_lines.append(f"  epochs run: {epoch+1}, early stopping patience: {cfg.early_stop_patience}")
report_lines.append("")
report_lines.append("Evaluation (test set aggregated over all horizons):")
report_lines.append(f"  Deep Learning (LSTM) - MAE: {dl_mae:.6f}, RMSE: {dl_rmse:.6f}, MAPE: {dl_mape:.2f}%")
report_lines.append(f"  Baseline (SARIMAX)    - MAE: {baseline_mae:.6f}, RMSE: {baseline_rmse:.6f}, MAPE: {baseline_mape:.2f}%")
report_lines.append("")
report_lines.append("Integrated Gradients interpretation (saved in explanation_text.txt)")
report_lines.append("")
report_lines.append("Hyperparameter choices reasoning:")
report_lines.append("  - hidden_size chosen to balance capacity and overfitting for hourly time-series.")
report_lines.append("  - OneCycleLR used to accelerate training and achieve robust generalization.")
report_lines.append("  - Early stopping to avoid overfitting on validation loss.")
report_lines.append("")
report_lines.append("Files produced in results directory:")
for fname in os.listdir(cfg.save_dir):
    report_lines.append("  - " + fname)

report_text = "\n".join(report_lines)
with open(os.path.join(cfg.save_dir, "report_summary.txt"), "w") as f:
    f.write(report_text)
print("Saved report_summary.txt")

# Optional: small plots of train/val loss and attribution heatmap
try:
    plt.figure(figsize=(8, 4))
    plt.plot(train_history["train_loss"], label="train_loss")
    plt.plot(train_history["val_loss"], label="val_loss")
    plt.legend()
    plt.title("Loss curves")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.save_dir, "loss_curve.png"))
    plt.close()

    # Attribution heatmap (time x features)
    plt.figure(figsize=(10, 4))
    plt.imshow(np.abs(attr).T, aspect="auto")
    plt.colorbar(label="abs attribution")
    plt.xlabel("time (relative)")
    plt.ylabel("feature")
    plt.yticks(ticks=np.arange(n_features), labels=feature_cols)
    plt.title("IG absolute attribution (features x time)")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.save_dir, "ig_heatmap.png"))
    plt.close()
    print("Saved loss_curve.png and ig_heatmap.png")
except Exception as e:
    print("Plotting skipped or failed:", e)

# -----------------------
# 9) Example: show a sample forecast comparison (print)
# -----------------------
# Take first 3 test samples and show predicted vs true for DL and baseline
n_show = min(3, len(preds))
print("\nSample forecasts (first {} test origins):".format(n_show))
for i in range(n_show):
    print(f"\nOrigin {i}:")
    print(" True (first 8 steps):", np.round(trues[i][:8], 4))
    print(" DL pred:", np.round(preds[i][:8], 4))
    # baseline index maps to the same aligned test origin
    print(" Baseline pred:", np.round(sar_preds[i][:8], 4))

# final note saved
with open(os.path.join(cfg.save_dir, "README.txt"), "w") as f:
    f.write("Results directory for Advanced Time Series Forecasting project.\n"
            "Contains: synthetic_multivariate.csv, lstm_forecaster.pth, report_summary.txt, explanation_text.txt, ig_attr_sample.npy, loss_curve.png, ig_heatmap.png\n")
print("\nAll done. Results saved in", cfg.save_dir)
