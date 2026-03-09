#!/usr/bin/env python3
"""
Forward-facing forecast evaluation pipeline.

Train on all data through Mar 1 2026, forecast Mar 2-31 (30d) and Mar 2 - May 15 (75d).
Compare our forecasts vs team's forecasts against actuals.

Usage:
    python3 forward_eval.py

Output:
    results/ directory with per-model forecasts and leaderboards.

Requirements:
    pip install -r requirements.txt

Environment:
    CUDA_VISIBLE_DEVICES is set to "" by default (CPU-only) to avoid
    LightGBM/cupy conflicts. To use GPU for neural models, modify the
    CUDA_VISIBLE_DEVICES logic below.
"""

import os
import sys
import warnings
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU only for LightGBM compatibility
os.environ["NIXTLA_ID_AS_COL"] = "1"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Paths (relative to this script) ───────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
OUT_DIR = SCRIPT_DIR / "results"
FORECAST_DIR = OUT_DIR / "forecasts"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FORECAST_DIR.mkdir(parents=True, exist_ok=True)

DAILY_CSV = DATA_DIR / "sp_daily_history.csv"
FORECASTS_30D = DATA_DIR / "forecasts_30d.csv"
FORECASTS_75D = DATA_DIR / "forecasts_75d.csv"
ACTUALS_30D = DATA_DIR / "sp_actuals_30d.csv"

# ── Constants ──────────────────────────────────────────────────────────────────
TRAIN_CUTOFF = pd.Timestamp("2026-03-01")
FC_START = pd.Timestamp("2026-03-02")
FC_END_30 = pd.Timestamp("2026-03-31")
FC_END_75 = pd.Timestamp("2026-05-15")
ACTUALS_END = pd.Timestamp("2026-03-08")  # last date with daily ground truth

H30 = 30
H75 = 75
INPUT_SIZE = 180   # lookback window for neural models (days)
RANDOM_SEED = 42


# ── Data loading ───────────────────────────────────────────────────────────────
def load_data():
    """
    Load and prepare all datasets.
    
    Key preprocessing: the raw daily data only contains rows where sales > 0.
    We create a complete daily panel and zero-fill missing days so that
    time series models see the true demand pattern (many zeros).
    """
    log.info("Loading data...")
    
    # Validate data files exist
    for f in [DAILY_CSV, FORECASTS_30D, FORECASTS_75D, ACTUALS_30D]:
        if not f.exists():
            log.error(f"Missing data file: {f}")
            log.error(f"Expected data files in: {DATA_DIR}")
            sys.exit(1)
    
    raw = pd.read_csv(DAILY_CSV, parse_dates=["date"])
    actuals_30d = pd.read_csv(ACTUALS_30D)
    team_30d = pd.read_csv(FORECASTS_30D)
    team_75d = pd.read_csv(FORECASTS_75D)

    asins = sorted(raw["asin"].unique())
    log.info(f"  {len(asins)} ASINs, date range {raw['date'].min().date()} to {raw['date'].max().date()}")

    # Build complete daily panel (zero-fill missing days)
    # This is critical: the raw data only has rows where units > 0
    # For time series models, we need explicit zeros
    all_dates = pd.date_range(raw["date"].min(), TRAIN_CUTOFF, freq="D")
    idx = pd.MultiIndex.from_product([asins, all_dates], names=["asin", "date"])
    panel = pd.DataFrame(index=idx).reset_index()
    panel = panel.merge(raw[["asin", "date", "units"]], on=["asin", "date"], how="left")
    panel["units"] = panel["units"].fillna(0).astype(float)
    
    log.info(f"  Panel: {len(panel)} rows ({len(asins)} ASINs × {len(all_dates)} dates, zero-filled)")

    # Also get actuals for Mar 2-8 from raw data (for daily evaluation)
    daily_actuals = raw[raw["date"].between(FC_START, ACTUALS_END)][["asin", "date", "units"]].copy()
    log.info(f"  Daily actuals (Mar 2-8): {len(daily_actuals)} rows")

    return panel, actuals_30d, team_30d, team_75d, daily_actuals, asins


def melt_team_forecasts(team_df, horizon_label):
    """
    Convert team forecasts from wide format (one column per date) to long format.
    Wide: asin | 2026-03-02 | 2026-03-02_p10 | 2026-03-02_p90 | 2026-03-03 | ...
    Long: asin | date | forecast
    
    Only keeps point forecasts (drops _p10 and _p90 percentile columns).
    """
    date_cols = [c for c in team_df.columns if c.startswith("2026") and "_p" not in c and len(c) == 10]
    melted = team_df[["asin"] + date_cols].melt(id_vars="asin", var_name="date", value_name="forecast")
    melted["date"] = pd.to_datetime(melted["date"])
    return melted.sort_values(["asin", "date"]).reset_index(drop=True)


# ── NeuralForecast models ─────────────────────────────────────────────────────
def run_neural_models(panel, asins, horizon):
    """
    Run PatchTST, LSTM, NHITS via NeuralForecast.
    
    All three are global models: trained on all ASINs simultaneously.
    They use direct multi-step forecasting (output all h values at once).
    
    Architecture details:
    - PatchTST: Transformer-based, segments input into overlapping patches
    - LSTM: 2-layer recurrent, MLP decoder for multi-step output
    - NHITS: Multi-scale hierarchical, 3 stacks at different resolutions
    
    All use:
    - input_size=180 (180 days of history)
    - max_steps=150 (max training iterations)
    - early stopping with patience=5 (checks every 25 steps)
    - MAE loss function
    - Standard (z-score) per-series normalization
    """
    from neuralforecast import NeuralForecast
    from neuralforecast.models import PatchTST, LSTM, NHITS

    log.info(f"Running neural models h={horizon}...")

    # NeuralForecast expects columns: unique_id, ds, y
    nf_df = panel.rename(columns={"asin": "unique_id", "date": "ds", "units": "y"})

    models = [
        PatchTST(h=horizon, input_size=INPUT_SIZE, max_steps=150,
                 val_check_steps=25, early_stop_patience_steps=5,
                 batch_size=64, scaler_type="standard", random_seed=RANDOM_SEED),
        LSTM(h=horizon, input_size=INPUT_SIZE, max_steps=150,
             val_check_steps=25, early_stop_patience_steps=5,
             batch_size=64, scaler_type="standard", random_seed=RANDOM_SEED),
        NHITS(h=horizon, input_size=INPUT_SIZE, max_steps=150,
              val_check_steps=25, early_stop_patience_steps=5,
              batch_size=64, scaler_type="standard", random_seed=RANDOM_SEED),
    ]

    nf = NeuralForecast(models=models, freq="D")
    # val_size=horizon: holds out the last h days of training data for validation/early stopping
    nf.fit(df=nf_df, val_size=horizon)
    preds = nf.predict()
    preds = preds.reset_index()

    results = {}
    for model_name in ["PatchTST", "LSTM", "NHITS"]:
        col = model_name
        df = preds[["unique_id", "ds", col]].copy()
        df.columns = ["asin", "date", "forecast"]
        df["forecast"] = df["forecast"].clip(lower=0)  # no negative forecasts
        # Trim to correct horizon
        end_date = FC_END_30 if horizon == H30 else FC_END_75
        df = df[df["date"].between(FC_START, end_date)]
        label = f"{model_name}_h{horizon}"
        results[label] = df
        df.to_csv(FORECAST_DIR / f"{label}.csv", index=False)
        log.info(f"  {label}: {len(df)} rows saved")

    return results


# ── Boosting models ────────────────────────────────────────────────────────────
def build_boosting_features(panel, asins):
    """
    Build 22 features for boosting models from the daily panel.
    
    Feature groups:
    1. Lag features (9): lag_1 through lag_60 — raw sales values shifted by N days
    2. Rolling statistics (12): mean/std/median over windows of 7/14/30/60 days
    3. Calendar features (4): day of week, month, weekend flag, day of month
    4. Derived features (2): 30-day trend slope, fraction of zero-sale days
    
    All lag/rolling features are shifted by 1 day to prevent leakage.
    """
    log.info("Building boosting features...")
    dfs = []
    for asin in asins:
        sub = panel[panel["asin"] == asin].copy().sort_values("date")

        # Lags (shifted by 1 to prevent leakage)
        for lag in [1, 2, 3, 5, 7, 14, 21, 30, 60]:
            sub[f"lag_{lag}"] = sub["units"].shift(lag)

        # Rolling stats (shifted by 1)
        for w in [7, 14, 30, 60]:
            sub[f"roll_mean_{w}"] = sub["units"].shift(1).rolling(w, min_periods=1).mean()
            sub[f"roll_std_{w}"] = sub["units"].shift(1).rolling(w, min_periods=1).std().fillna(0)
            sub[f"roll_median_{w}"] = sub["units"].shift(1).rolling(w, min_periods=1).median()

        # Calendar features
        sub["dow"] = sub["date"].dt.dayofweek
        sub["month"] = sub["date"].dt.month
        sub["is_weekend"] = (sub["dow"] >= 5).astype(int)
        sub["day_of_month"] = sub["date"].dt.day

        # Trend: slope of linear fit on last 30 days (shifted by 1)
        sub["trend_slope_30"] = sub["units"].shift(1).rolling(30, min_periods=7).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0, raw=True
        ).fillna(0)

        # Fraction of zero-sale days in last 30 days (shifted by 1)
        sub["frac_zero_30"] = sub["units"].shift(1).rolling(30, min_periods=1).apply(
            lambda x: (x == 0).mean(), raw=True
        ).fillna(0)

        dfs.append(sub)

    full = pd.concat(dfs, ignore_index=True)
    return full


def run_boosting_models(panel, asins, horizon):
    """
    Run XGBoost, LightGBM, NGBoost with recursive multi-step forecasting.
    
    Approach:
    1. Train a GLOBAL model on all ASINs (pooled training data)
    2. For each ASIN, recursively forecast one day at a time:
       - Compute features from history (actual + predicted values)
       - Predict next day
       - Append prediction to history
       - Repeat for full horizon
    
    This means errors compound: day-30's prediction quality depends on
    all 29 prior predictions used to compute its features.
    
    All models use:
    - n_estimators=500 (XGBoost/LightGBM) or 200 (NGBoost)
    - max_depth=6 (XGBoost/LightGBM) or 4 (NGBoost base learner)
    - learning_rate=0.03 (XGBoost/LightGBM) or 0.05 (NGBoost)
    - random_state=42
    """
    import xgboost as xgb
    import lightgbm as lgb
    from ngboost import NGBoost
    from sklearn.tree import DecisionTreeRegressor

    log.info(f"Running boosting models h={horizon}...")
    end_date = FC_END_30 if horizon == H30 else FC_END_75
    forecast_dates = pd.date_range(FC_START, end_date, freq="D")

    # Build features on training data
    full = build_boosting_features(panel, asins)
    feat_cols = [c for c in full.columns if c not in ["asin", "date", "units", "revenue"]]

    # Training data: Oct 2024+ (need 60-day lag history to be meaningful)
    train = full[(full["date"] <= TRAIN_CUTOFF) & (full["date"] >= "2024-10-01")].dropna(subset=feat_cols)
    X_train = train[feat_cols].values
    y_train = train["units"].values

    log.info(f"  Training set: {len(train)} rows, {len(feat_cols)} features")
    log.info(f"  Features: {feat_cols}")

    model_configs = {
        f"XGBoost_h{horizon}": xgb.XGBRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.03,
            objective="reg:squarederror", n_jobs=1, random_state=RANDOM_SEED, verbosity=0
        ),
        f"LightGBM_h{horizon}": lgb.LGBMRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.03,
            n_jobs=1, random_state=RANDOM_SEED, verbose=-1
        ),
        f"NGBoost_h{horizon}": NGBoost(
            n_estimators=200,
            Base=DecisionTreeRegressor(max_depth=4),
            learning_rate=0.05,
            random_state=RANDOM_SEED,
            verbose=False,
        ),
    }

    results = {}
    for model_name, model in model_configs.items():
        log.info(f"  Training {model_name}...")
        t0 = time.time()
        model.fit(X_train, y_train)
        log.info(f"    Trained in {time.time()-t0:.1f}s")

        # Recursive forecasting per ASIN
        all_preds = []
        for asin in asins:
            sub = full[full["asin"] == asin].copy().sort_values("date")
            
            # Build a buffer of recent history for recursive features
            history = sub[sub["date"] <= TRAIN_CUTOFF]["units"].values.tolist()

            for fd in forecast_dates:
                # Compute features from history (includes prior predictions)
                feats = {}
                h = history
                n = len(h)
                for lag in [1, 2, 3, 5, 7, 14, 21, 30, 60]:
                    feats[f"lag_{lag}"] = h[-lag] if n >= lag else 0
                for w in [7, 14, 30, 60]:
                    window = h[-w:] if n >= w else h
                    feats[f"roll_mean_{w}"] = np.mean(window)
                    feats[f"roll_std_{w}"] = np.std(window) if len(window) > 1 else 0
                    feats[f"roll_median_{w}"] = np.median(window)
                feats["dow"] = fd.dayofweek
                feats["month"] = fd.month
                feats["is_weekend"] = int(fd.dayofweek >= 5)
                feats["day_of_month"] = fd.day
                recent30 = h[-30:] if n >= 30 else h[-7:]
                feats["trend_slope_30"] = np.polyfit(range(len(recent30)), recent30, 1)[0] if len(recent30) > 1 else 0
                feats["frac_zero_30"] = np.mean([1 if x == 0 else 0 for x in (h[-30:] if n >= 30 else h)])

                X_pred = np.array([[feats[c] for c in feat_cols]])
                pred = max(0, float(model.predict(X_pred)[0]))
                all_preds.append({"asin": asin, "date": fd, "forecast": pred})
                history.append(pred)  # recursive: feed prediction back as history

        df = pd.DataFrame(all_preds)
        results[model_name] = df
        df.to_csv(FORECAST_DIR / f"{model_name}.csv", index=False)
        log.info(f"  {model_name}: {len(df)} rows, {time.time()-t0:.1f}s total")

    return results


# ── Statistical / baseline models ─────────────────────────────────────────────
def run_stat_baselines(panel, asins, horizon):
    """
    Run simple statistical baselines via statsforecast.
    
    Models:
    - AutoCES: Complex Exponential Smoothing (auto-selects best variant)
    - SeasonalNaive: Repeats the value from 7 days ago
    - WindowAverage (MA7): Simple 7-day rolling average
    
    All are LOCAL models (fitted independently per ASIN, no cross-series learning).
    """
    from statsforecast import StatsForecast
    from statsforecast.models import AutoCES, SeasonalNaive, WindowAverage

    log.info(f"Running statistical models h={horizon}...")
    end_date = FC_END_30 if horizon == H30 else FC_END_75

    nf_df = panel.rename(columns={"asin": "unique_id", "date": "ds", "units": "y"})

    models = [
        AutoCES(season_length=7),
        SeasonalNaive(season_length=7),
        WindowAverage(window_size=7),
    ]

    sf = StatsForecast(models=models, freq="D", n_jobs=1)
    sf.fit(df=nf_df)
    preds = sf.predict(h=horizon)
    preds = preds.reset_index()

    results = {}
    model_map = {
        "CES": "CES",
        "SeasonalNaive": "SeasonalNaive",
        "WindowAverage": "MA7",
    }

    for col, label in model_map.items():
        df = preds[["unique_id", "ds", col]].copy()
        df.columns = ["asin", "date", "forecast"]
        df["forecast"] = df["forecast"].clip(lower=0)
        df = df[df["date"].between(FC_START, end_date)]
        full_label = f"{label}_h{horizon}"
        results[full_label] = df
        df.to_csv(FORECAST_DIR / f"{full_label}.csv", index=False)
        log.info(f"  {full_label}: {len(df)} rows saved")

    return results


# ── Ensemble ───────────────────────────────────────────────────────────────────
def build_ensemble(all_results, horizon):
    """
    Simple equally-weighted ensemble of all non-baseline models.
    
    Includes: CES, PatchTST, LSTM, NHITS, XGBoost, LightGBM, NGBoost
    Excludes: MA7, SeasonalNaive (too simple to contribute to ensemble)
    
    No learned weights or stacking — just arithmetic mean of predictions.
    Potential improvement: use holdout performance to weight models,
    or use per-ASIN model selection (oracle MAE shows 46% improvement potential).
    """
    log.info(f"Building ensemble h={horizon}...")

    ensemble_models = [k for k in all_results 
                       if f"h{horizon}" in k 
                       and "MA7" not in k 
                       and "SeasonalNaive" not in k]
    if not ensemble_models:
        log.warning("  No models for ensemble!")
        return {}

    log.info(f"  Ensemble from: {ensemble_models}")

    # Merge all on (asin, date)
    merged = None
    for m in ensemble_models:
        df = all_results[m][["asin", "date", "forecast"]].rename(columns={"forecast": m})
        if merged is None:
            merged = df
        else:
            merged = merged.merge(df, on=["asin", "date"], how="outer")

    model_cols = ensemble_models
    merged["forecast"] = merged[model_cols].mean(axis=1).clip(lower=0)
    ens = merged[["asin", "date", "forecast"]].copy()

    label = f"Ensemble_h{horizon}"
    ens.to_csv(FORECAST_DIR / f"{label}.csv", index=False)
    log.info(f"  {label}: {len(ens)} rows saved")
    return {label: ens}


# ── Evaluation ─────────────────────────────────────────────────────────────────
def evaluate_daily(model_preds, daily_actuals, label):
    """
    Evaluate daily forecasts against Mar 2-8 actuals.
    
    Only 7 days of daily ground truth are available (training data ends Mar 8).
    n_obs will be < 211*7 because the raw data only has rows where units > 0.
    """
    merged = model_preds.merge(daily_actuals, on=["asin", "date"], how="inner")
    if len(merged) == 0:
        return {"model": label, "daily_mae": np.nan, "daily_rmse": np.nan, "n_obs": 0}
    err = (merged["forecast"] - merged["units"]).abs()
    return {
        "model": label,
        "daily_mae": err.mean(),
        "daily_rmse": np.sqrt(((merged["forecast"] - merged["units"])**2).mean()),
        "n_obs": len(merged),
    }


def evaluate_30d_total(model_preds, actuals_30d, label):
    """
    Evaluate 30-day total predictions.
    
    Sum daily forecasts (Mar 2-31) per ASIN and compare against units_30d actuals.
    This is the primary evaluation metric for inventory decisions.
    """
    our_totals = model_preds[model_preds["date"].between(FC_START, FC_END_30)] \
        .groupby("asin")["forecast"].sum().reset_index()
    our_totals.columns = ["asin", "pred_total"]
    merged = our_totals.merge(actuals_30d[["asin", "units_30d"]], on="asin", how="inner")
    if len(merged) == 0:
        return {"model": label, "total_mae": np.nan}
    
    err = (merged["pred_total"] - merged["units_30d"]).abs()
    bias = (merged["pred_total"] - merged["units_30d"]).mean()
    wape = err.sum() / max(merged["units_30d"].sum(), 1)
    
    return {
        "model": label,
        "total_mae": err.mean(),
        "total_rmse": np.sqrt(((merged["pred_total"] - merged["units_30d"])**2).mean()),
        "total_bias": bias,
        "wape": wape,
        "our_sum": merged["pred_total"].sum(),
        "actual_sum": merged["units_30d"].sum(),
    }


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    t_start = time.time()
    panel, actuals_30d, team_30d, team_75d, daily_actuals, asins = load_data()

    # Melt team forecasts from wide to long format
    team_30d_long = melt_team_forecasts(team_30d, "30d")
    team_75d_long = melt_team_forecasts(team_75d, "75d")

    all_results = {}
    all_eval_daily = []
    all_eval_total = []

    # ── Horizon = 30 days (Mar 2 - Mar 31) ──────────────────────────────────
    log.info("=" * 60)
    log.info("HORIZON = 30 days (Mar 2 - Mar 31)")
    log.info("=" * 60)

    stat_30 = run_stat_baselines(panel, asins, H30)
    all_results.update(stat_30)

    neural_30 = run_neural_models(panel, asins, H30)
    all_results.update(neural_30)

    boost_30 = run_boosting_models(panel, asins, H30)
    all_results.update(boost_30)

    ens_30 = build_ensemble(all_results, H30)
    all_results.update(ens_30)

    # ── Horizon = 75 days (Mar 2 - May 15) ──────────────────────────────────
    log.info("=" * 60)
    log.info("HORIZON = 75 days (Mar 2 - May 15)")
    log.info("=" * 60)

    stat_75 = run_stat_baselines(panel, asins, H75)
    all_results.update(stat_75)

    neural_75 = run_neural_models(panel, asins, H75)
    all_results.update(neural_75)

    boost_75 = run_boosting_models(panel, asins, H75)
    all_results.update(boost_75)

    ens_75 = build_ensemble(all_results, H75)
    all_results.update(ens_75)

    # ── Evaluation ──────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("EVALUATION")
    log.info("=" * 60)

    # Evaluate team forecast
    team_daily_eval = evaluate_daily(team_30d_long, daily_actuals, "Team_30d")
    team_total_eval = evaluate_30d_total(team_30d_long, actuals_30d, "Team_30d")
    all_eval_daily.append(team_daily_eval)
    all_eval_total.append(team_total_eval)

    # Evaluate our models
    for label, preds in all_results.items():
        d_eval = evaluate_daily(preds, daily_actuals, label)
        all_eval_daily.append(d_eval)
        if "h30" in label.lower() or "h30" in label:
            t_eval = evaluate_30d_total(preds, actuals_30d, label)
            all_eval_total.append(t_eval)

    # Build and save leaderboards
    daily_lb = pd.DataFrame(all_eval_daily).sort_values("daily_mae")
    total_lb = pd.DataFrame(all_eval_total).sort_values("total_mae")
    combined = daily_lb.merge(total_lb, on="model", how="outer", suffixes=("_daily", "_total"))
    combined = combined.sort_values("total_mae")

    daily_lb.to_csv(OUT_DIR / "daily_leaderboard.csv", index=False)
    total_lb.to_csv(OUT_DIR / "total_30d_leaderboard.csv", index=False)
    combined.to_csv(OUT_DIR / "forward_facing_leaderboard.csv", index=False)

    # ── Print results ──────────────────────────────────────────────────────
    log.info("\n" + "=" * 80)
    log.info("DAILY MAE LEADERBOARD (Mar 2-8 actuals)")
    log.info("=" * 80)
    print(daily_lb.to_string(index=False))

    log.info("\n" + "=" * 80)
    log.info("30-DAY TOTAL LEADERBOARD (sum of daily forecasts vs units_30d)")
    log.info("=" * 80)
    print(total_lb.to_string(index=False))

    elapsed = time.time() - t_start
    log.info(f"\nTotal runtime: {elapsed/60:.1f} minutes")
    log.info(f"Results saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
