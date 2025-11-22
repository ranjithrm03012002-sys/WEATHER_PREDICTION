#!/usr/bin/env python3
"""
Advanced Time Series Forecasting with Prophet + Hyperparameter Optimization
Complete, self-contained script that:
 - Generates a 4+ year daily synthetic time series with trend, weekly & yearly seasonality, and anomalies
 - Trains a baseline Prophet model (default params)
 - Performs a randomized hyperparameter search adapted to time-series CV (TimeSeriesSplit)
 - Refits best model on full train and evaluates on a hold-out test set
 - Prints and saves metrics and CSVs for forecasts and CV results

Requirements:
  python>=3.8
  pip install prophet pandas numpy scikit-learn tqdm matplotlib
"""
import math
import itertools
import warnings
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm
import argparse
import json
import os

# -------------------------
# Utilities & Config
# -------------------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

@dataclass
class Results:
    baseline_mae: float = float("nan")
    baseline_rmse: float = float("nan")
    baseline_mape: float = float("nan")
    opt_mae: float = float("nan")
    opt_rmse: float = float("nan")
    opt_mape: float = float("nan")
    best_params: Dict[str, Any] = None

def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    mask = y_true_arr != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true_arr[mask] - y_pred_arr[mask]) / y_true_arr[mask])) * 100.0)

# -------------------------
# Data generation
# -------------------------
def generate_synthetic_series(start: str = "2016-01-01", years: int = 4, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Generate daily synthetic time series:
     - linear trend
     - yearly seasonality (complex)
     - weekly seasonality
     - Gaussian noise
     - injected short anomalies (spikes/dips)
    Returns DataFrame with columns ['ds','y'].
    """
    np.random.seed(seed)
    periods = years * 365 + int(years / 4)  # crude allowance for leap days
    rng = pd.date_range(start=start, periods=periods, freq="D")
    n = len(rng)
    t = np.arange(n, dtype=float)

    # Components
    trend = 0.002 * t  # slightly stronger trend than example so effect visible
    yearly = 3.0 * np.sin(2 * np.pi * t / 365.25) + 1.0 * np.cos(2 * np.pi * t / 365.25 * 2)
    weekly = 1.6 * np.sin(2 * np.pi * t / 7)
    noise = np.random.normal(scale=0.7, size=n)

    y = 12.0 + trend + yearly + weekly + noise

    # Inject anomalies: 10 random events of 1-3 day duration with positive or negative shock
    rng_indices = np.random.choice(np.arange(30, n - 30), size=10, replace=False)
    for idx in rng_indices:
        dur = np.random.choice([1, 2, 3])
        shock = np.random.choice([6.0, -6.0]) * (0.4 + np.random.rand() * 1.2)
        end = min(n, idx + dur)
        y[idx:end] += shock

    df = pd.DataFrame({"ds": rng, "y": y})
    return df

# -------------------------
# Baseline Prophet model
# -------------------------
def fit_prophet_and_forecast(train_df: pd.DataFrame, periods: int, params: Dict[str, Any] = None, extra_seasonalities: List[Dict] = None) -> pd.DataFrame:
    """
    Fit Prophet with optional parameters and extra seasonalities, return forecast dataframe for train+future periods
    """
    if params is None:
        params = {}
    model = Prophet(**params)
    # Add standard extra seasonalities if requested
    if extra_seasonalities:
        for s in extra_seasonalities:
            # each s: {'name':str, 'period':float, 'fourier_order':int}
            model.add_seasonality(name=s["name"], period=s["period"], fourier_order=s["fourier_order"])
    model.fit(train_df)
    future = model.make_future_dataframe(periods=periods, freq="D")
    forecast = model.predict(future)
    return forecast, model

# -------------------------
# Cross-validated randomized search for Prophet hyperparameters
# -------------------------
def randomized_search_prophet(train_df: pd.DataFrame,
                              param_grid: Dict[str, List[Any]],
                              n_trials: int = 36,
                              n_splits: int = 4,
                              extra_seasonalities: List[Dict] = None,
                              random_state: int = RANDOM_SEED) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Randomly sample combos from the cartesian product of param_grid (without replacement if possible),
    use TimeSeriesSplit to evaluate each candidate by mean CV MAE.
    Returns results_df sorted by cv_mae and best_params dict.
    """
    # Build full product of params
    keys = list(param_grid.keys())
    all_values = [param_grid[k] for k in keys]
    all_combinations = list(itertools.product(*all_values))
    rng = np.random.default_rng(random_state)
    rng.shuffle(all_combinations)
    candidates = all_combinations[:min(n_trials, len(all_combinations))]

    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = []
    for combo in tqdm(candidates, desc="Randomized search", unit="trial"):
        candidate_params = dict(zip(keys, combo))
        fold_maes = []
        # time-series-aware CV
        for train_idx, val_idx in tscv.split(train_df):
            train_cv = train_df.iloc[train_idx].reset_index(drop=True)
            val_cv = train_df.iloc[val_idx].reset_index(drop=True)

            try:
                forecast_cv, _ = fit_prophet_and_forecast(train_cv,
                                                          periods=len(val_cv),
                                                          params=candidate_params,
                                                          extra_seasonalities=extra_seasonalities)
            except Exception as e:
                # If model fails to fit for some parameters, penalize heavily
                fold_maes.append(float("inf"))
                continue

            yhat = forecast_cv[["ds", "yhat"]].iloc[-len(val_cv):]["yhat"].values
            fold_maes.append(mean_absolute_error(val_cv["y"].values, yhat))

        mean_cv_mae = float(np.mean(fold_maes))
        entry = {**candidate_params, "cv_mae": mean_cv_mae}
        results.append(entry)

    results_df = pd.DataFrame(results).sort_values("cv_mae").reset_index(drop=True)
    best_row = results_df.iloc[0].to_dict()
    best_params = {k: best_row[k] for k in keys}
    return results_df, best_params

# -------------------------
# Main routine
# -------------------------
def main(output_dir: str = "prophet_output",
         years: int = 4,
         holdout_days: int = 180,
         n_trials: int = 36,
         n_splits: int = 4):
    os.makedirs(output_dir, exist_ok=True)

    # 1) Data
    df = generate_synthetic_series(years=years)
    df.to_csv(os.path.join(output_dir, "synthetic_series.csv"), index=False)

    train_df = df.iloc[:-holdout_days].reset_index(drop=True)
    test_df = df.iloc[-holdout_days:].reset_index(drop=True)

    # Extra seasonalities to explicitly add: weekly and yearly (give tuning control to Prophet)
    extra_seasonalities = [
        {"name": "weekly", "period": 7, "fourier_order": 3},
        {"name": "yearly", "period": 365.25, "fourier_order": 10},
    ]

    results = Results()

    # 2) Baseline model (default Prophet)
    baseline_forecast, baseline_model = fit_prophet_and_forecast(train_df, periods=holdout_days, params=None, extra_seasonalities=extra_seasonalities)
    baseline_fc_holdout = baseline_forecast[["ds", "yhat"]].iloc[-holdout_days:].reset_index(drop=True)

    results.baseline_mae = float(mean_absolute_error(test_df["y"], baseline_fc_holdout["yhat"]))
    results.baseline_rmse = float(math.sqrt(mean_squared_error(test_df["y"], baseline_fc_holdout["yhat"])))
    results.baseline_mape = float(mape(test_df["y"].values, baseline_fc_holdout["yhat"].values))

    # Save baseline forecast
    baseline_forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_csv(os.path.join(output_dir, "baseline_forecast.csv"), index=False)
    # Save baseline model changepoints + params (best-effort)
    try:
        baseline_config = {
            "model": "prophet_baseline",
            "params": {"default": True},
        }
        with open(os.path.join(output_dir, "baseline_info.json"), "w") as f:
            json.dump(baseline_config, f, indent=2)
    except Exception:
        pass

    # 3) Hyperparameter search space
    param_grid = {
        "changepoint_prior_scale": [0.001, 0.01, 0.05, 0.1, 0.5],
        "seasonality_mode": ["additive", "multiplicative"],
        "seasonality_prior_scale": [1.0, 5.0, 10.0],
        "holidays_prior_scale": [0.1, 1.0, 5.0],
    }

    # 4) Randomized search with TimeSeriesSplit
    results_df, best_params = randomized_search_prophet(train_df=train_df,
                                                        param_grid=param_grid,
                                                        n_trials=n_trials,
                                                        n_splits=n_splits,
                                                        extra_seasonalities=extra_seasonalities,
                                                        random_state=RANDOM_SEED)

    results_df.to_csv(os.path.join(output_dir, "prophet_cv_results.csv"), index=False)
    results.best_params = best_params

    # 5) Refit best model on full train and evaluate on holdout
    best_forecast, best_model = fit_prophet_and_forecast(train_df,
                                                         periods=holdout_days,
                                                         params=best_params,
                                                         extra_seasonalities=extra_seasonalities)
    best_fc_holdout = best_forecast[["ds", "yhat"]].iloc[-holdout_days:].reset_index(drop=True)

    results.opt_mae = float(mean_absolute_error(test_df["y"], best_fc_holdout["yhat"]))
    results.opt_rmse = float(math.sqrt(mean_squared_error(test_df["y"], best_fc_holdout["yhat"])))
    results.opt_mape = float(mape(test_df["y"].values, best_fc_holdout["yhat"].values))

    # Save optimized forecast and best params
    best_forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_csv(os.path.join(output_dir, "optimized_forecast.csv"), index=False)
    with open(os.path.join(output_dir, "best_params.json"), "w") as f:
        json.dump(best_params, f, indent=2)

    # 6) Comparison table
    comparison = pd.DataFrame({
        "metric": ["MAE", "RMSE", "MAPE(%)"],
        "baseline": [results.baseline_mae, results.baseline_rmse, results.baseline_mape],
        "optimized": [results.opt_mae, results.opt_rmse, results.opt_mape]
    })
    comparison.to_csv(os.path.join(output_dir, "metrics_comparison.csv"), index=False)

    # 7) Print summary to stdout (machine- and human-readable)
    print("\n=== Holdout Metrics Comparison ===")
    print(comparison.to_string(index=False))

    print("\n=== Best Hyperparameters (by CV MAE) ===")
    print(json.dumps(best_params, indent=2))

    print("\nTop 10 CV candidates (lowest cv_mae first):")
    print(results_df.head(10).to_string(index=False))

    # 8) Save a small JSON summary
    summary = asdict(results)
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nAll outputs saved to directory: {os.path.abspath(output_dir)}\n")
    return summary

# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prophet hyperparameter tuning script (randomized search + TimeSeriesSplit).")
    parser.add_argument("--output-dir", type=str, default="prophet_output", help="Directory to save outputs (CSV/JSON).")
    parser.add_argument("--years", type=int, default=4, help="Number of years for synthetic data.")
    parser.add_argument("--holdout-days", type=int, default=180, help="Number of days as hold-out test set.")
    parser.add_argument("--n-trials", type=int, default=36, help="Number of randomized trials to run.")
    parser.add_argument("--n-splits", type=int, default=4, help="Number of TimeSeriesSplit folds for CV.")
    args = parser.parse_args()

    main(output_dir=args.output_dir,
         years=args.years,
         holdout_days=args.holdout_days,
         n_trials=args.n_trials,
         n_splits=args.n_splits)
