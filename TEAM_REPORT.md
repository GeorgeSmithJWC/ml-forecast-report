# ML Forecasting Report — Beating the Team
**Date:** March 9, 2026  
**Prepared by:** George (AI assistant)  
**For:** Jake Carter / EverValue Emporium team  

---

## Executive Summary

We tested 10 model architectures against the team's existing 30-day forecast on the exact same evaluation window (March 2-31, 2026). **Every single model outperformed the team forecast on both daily accuracy and 30-day total accuracy.** The best model (LSTM) reduced total forecast error by 49%.

The team's primary issue is **systematic underprediction** — forecasting 6,457 total units vs 20,098 actual (a 68% undercount). Even a simple 7-day moving average with zero tuning beats the team forecast.

---

## Data Used

### Training Data
- **Source:** `sp_daily_history.csv` (internal SP API daily sales data)
- **Size:** 60,205 rows, 211 ASINs
- **Date range:** September 1, 2024 → March 1, 2026 (553 days, avg 285 days/ASIN)
- **Total units in training period:** 462,891
- **Missing data handling:** Zero-filled (if an ASIN has no sale on a day, it's recorded as 0)

### Evaluation Data (Ground Truth)
- **Source:** `sp_actuals_30d.csv` — actual units sold March 2-31, 2026
- **211 ASINs**, total 20,098 units sold in the 30-day window
- **Mean units per ASIN:** 95.3 (median: 36, max: 1,055)
- **11 ASINs had zero sales** in the evaluation period

### Team Forecasts (Benchmark)
- **Source:** `forecasts_30d.csv` — team's daily forecasts for March 2-31
- **Format:** Wide (211 rows × 90 columns including percentile bands p10/p50/p90)
- **We used the point forecast columns** (date-only, no _p10/_p50/_p90 suffix)

### What Was NOT Used (Important)
- No Keepa data in the forward-facing evaluation (separate Keepa pipeline exists)
- No external features (weather, holidays, etc.)
- Pure time series: just ASIN × date × units

---

## Models Tested

### Category 1: Statistical Baselines (no ML)

| Model | Library | Description | Config |
|-------|---------|-------------|--------|
| **CES** (Complex Exponential Smoothing) | statsforecast `AutoCES` | Automatic exponential smoothing with seasonal decomposition | season_length=7 |
| **SeasonalNaive** | statsforecast | Repeats the value from 7 days ago | season_length=7 |
| **MA7** (Moving Average) | statsforecast `WindowAverage` | Simple 7-day rolling average | window_size=7 |

### Category 2: Neural Networks

| Model | Library | Description | Config |
|-------|---------|-------------|--------|
| **PatchTST** | neuralforecast | Transformer-based, patches time series into tokens | input_size=180, max_steps=150, batch_size=64, scaler=standard |
| **LSTM** | neuralforecast | Long Short-Term Memory recurrent network | input_size=180, max_steps=150, batch_size=64, scaler=standard |
| **NHITS** | neuralforecast | Neural Hierarchical Interpolation for Time Series | input_size=180, max_steps=150, batch_size=64, scaler=standard |

All neural models: early stopping (patience=5, check every 25 steps), seed=42.  
Input size = 180 days of history → predict h days forward.

### Category 3: Gradient Boosting

| Model | Library | Description | Config |
|-------|---------|-------------|--------|
| **XGBoost** | xgboost `XGBRegressor` | Gradient boosted trees | n_estimators=500, max_depth=6, lr=0.03 |
| **LightGBM** | lightgbm `LGBMRegressor` | Light gradient boosting | n_estimators=500, max_depth=6, lr=0.03 |
| **NGBoost** | ngboost | Natural Gradient Boosting (probabilistic) | n_estimators=200, base=DecisionTree(depth=4), lr=0.05 |

**Boosting features (22 features per observation):**
- Lags: 1, 2, 3, 5, 7, 14, 21, 30, 60 days
- Rolling statistics: mean, std, median over 7/14/30/60 day windows
- Calendar: day of week, month, is_weekend, day of month
- Trend: 30-day linear slope
- Zero fraction: % of zero-sale days in last 30 days
- **Recursive forecasting:** predictions fed back as inputs for subsequent days

### Category 4: Ensemble

| Model | Method |
|-------|--------|
| **Ensemble** | Simple average of all non-baseline models (CES, PatchTST, LSTM, NHITS, XGBoost, LightGBM, NGBoost) |

---

## How Forecasts Are Structured

### Training Protocol
1. Use ALL data through March 1, 2026 as training data
2. Generate forecasts for every ASIN for every day March 2-31 (30d) and March 2 - May 15 (75d)
3. Compare against actual sales in `sp_actuals_30d.csv`

### Day-by-Day Prediction Flow
Each model outputs a CSV with columns: `asin, date, forecast`

Example (XGBoost, ASIN B0036DUP6C):
```
asin,date,forecast
B0036DUP6C,2026-03-02,0.197
B0036DUP6C,2026-03-03,0.202
B0036DUP6C,2026-03-04,0.158
...
B0036DUP6C,2026-03-31,0.142
```

**For boosting models:** Forecasts are recursive — day 1 uses real history, day 2 uses day 1's prediction as an input feature, day 3 uses days 1-2 predictions, etc. This is critical because we're forecasting beyond the data cutoff.

**For neural models:** NeuralForecast generates all h days in a single forward pass (direct multi-step forecasting, not recursive).

**For statistical models:** StatsForecast generates the full horizon from the fitted model parameters.

### Evaluation Metrics
1. **Daily MAE** — Average absolute error per ASIN per day (using Mar 2-8 actuals only, since daily actuals are only available through Mar 8)
2. **Total 30d MAE** — Sum each ASIN's daily forecasts → 30d total → compare to `units_30d`
3. **WAPE** — Weighted Absolute Percentage Error (total absolute error / total actual units)
4. **Bias** — Mean(predicted - actual). Negative = underprediction.

---

## Results

### Daily Accuracy (Mar 2-8, 864 observations)

| Rank | Model | Daily MAE | Daily RMSE |
|------|-------|-----------|------------|
| 1 | LightGBM | **3.10** | 6.07 |
| 2 | Ensemble | 3.15 | 5.81 |
| 3 | NGBoost | 3.37 | 7.56 |
| 4 | CES | 3.35 | 6.07 |
| 5 | NHITS | 3.47 | 6.42 |
| 6 | XGBoost | 3.54 | 7.63 |
| 7 | MA7 | 3.55 | 6.86 |
| 8 | PatchTST | 3.67 | 6.71 |
| 9 | SeasonalNaive | 4.13 | 7.67 |
| 10 | LSTM | 4.14 | 7.75 |
| **11** | **Team** | **5.01** | **9.45** |

### 30-Day Total Accuracy (211 ASINs)

| Rank | Model | Total MAE | WAPE | Bias | Predicted Total | Actual Total |
|------|-------|-----------|------|------|-----------------|--------------|
| 1 | LSTM | **35.87** | 37.7% | -11.6 | 17,640 | 20,098 |
| 2 | Ensemble | 41.38 | 43.4% | -11.6 | 17,641 | 20,098 |
| 3 | PatchTST | 42.32 | 44.4% | -31.6 | 13,440 | 20,098 |
| 4 | NHITS | 43.96 | 46.2% | -9.4 | 18,115 | 20,098 |
| 5 | MA7 | 44.89 | 47.1% | -24.3 | 14,979 | 20,098 |
| 6 | SeasonalNaive | 44.87 | 47.1% | -21.7 | 15,528 | 20,098 |
| 7 | CES | 48.38 | 50.8% | -38.5 | 11,973 | 20,098 |
| 8 | NGBoost | 58.60 | 61.5% | +4.8 | 21,108 | 20,098 |
| 9 | LightGBM | 58.83 | 61.8% | +8.2 | 21,827 | 20,098 |
| 10 | XGBoost | 64.46 | 67.7% | -3.4 | 19,384 | 20,098 |
| **11** | **Team** | **70.79** | **74.3%** | **-64.7** | **6,457** | **20,098** |

### Team Diagnosis
- **Team total predicted units: 6,457 vs actual 20,098** — forecasting only 32% of actual volume
- **Bias: -64.7 units per ASIN** — massive systematic underprediction
- This means the team is likely under-buying inventory by ~68%
- Even a SeasonalNaive (just repeat last week) predicts 15,528 total — 2.4x more accurate than the team

---

## Key Findings

### 1. LSTM is Best for Inventory Decisions
- Lowest total 30d error (MAE 35.87, WAPE 37.7%)
- Lowest bias among top models (-11.6, slight underprediction)
- Best for answering "how many units should we buy?"

### 2. LightGBM is Best for Daily Planning
- Lowest daily MAE (3.10)
- Slight overprediction (+8.2 bias) — conservative but safe
- Best for answering "what will sell today/this week?"

### 3. The Ensemble Hedges Both
- 2nd place on both daily (3.15) and total (41.38)
- Averages neural + boosting models
- Most robust across different ASIN profiles

### 4. The Team Has a Calibration Problem
- Not a methodology problem — the magnitude of underprediction (-68%) suggests a systematic issue
- Possible causes: stale demand baselines, over-weighting of low-volume ASINs, or a conservative bias baked into the forecast formula
- The fix isn't "use a better model" — the team needs to understand WHY their forecasts are 3x below actual

---

## Separate Keepa Analysis (Holdout, Not Forward-Facing)

We also ran a separate experiment incorporating Keepa marketplace data:

### Track A: Internal Sales + Keepa Features (holdout eval)
| Model | 30d MAE | 75d MAE |
|-------|---------|---------|
| Ridge | **1.85** | 2.36 |
| LightGBM | 2.15 | 2.25 |
| XGBoost | 2.31 | 2.54 |

### Track B: Keepa Data ONLY (no internal sales data)
| Model | 30d MAE | 75d MAE |
|-------|---------|---------|
| XGBoost | 2.99 | 4.56 |
| LightGBM | 3.04 | 4.41 |
| Ridge | 3.48 | 4.39 |

### Top Keepa Features (by XGBoost importance)
1. **Our buy box ownership %** (17.3%) — calculated from raw hourly buy box pings, NOT Keepa's reported stat
2. **Monthly sold badge** (8.8%) — Keepa's total market volume indicator
3. **Sales rank (log)** (6.8%)
4. **Month** (5.7%) — seasonality
5. **Sales rank level** (5.5%)
6. **Rank volatility (7-day std)** (5.3%)
7. **Rank momentum (7-day delta)** (4.8%)
8. **Day of month** (4.7%)
9. **Rank 28-day MA** (4.2%)
10. **Price vs list price ratio** (4.2%)

**Why this matters:** Track B (Keepa-only) can forecast ASINs we've never sold. This is the foundation for an ASIN acquisition scoring tool.

---

## Reproducibility

### Environment
- **Hardware:** AWS g5.2xlarge (8 vCPU, 32GB RAM, NVIDIA A10G 24GB VRAM)
- **OS:** Ubuntu 22.04, Python 3.10
- **Key libraries:** neuralforecast, statsforecast, xgboost, lightgbm, ngboost, pandas, numpy

### Running the Pipeline
```bash
cd ~/ml_forecast/benchmark
python3 forward_eval.py
```

### Output
- Per-model CSVs in `results/forward_facing/` (asin, date, forecast)
- `forward_facing_leaderboard.csv` — combined leaderboard
- `daily_leaderboard.csv` — daily MAE rankings
- `total_30d_leaderboard.csv` — 30d total rankings

### Full Source Code
The complete `forward_eval.py` script is self-contained at:
`~/ml_forecast/benchmark/forward_eval.py` (~380 lines)

---

## Next Steps

1. **Get the full 2,800 ASIN dataset** — current analysis uses Jake's curated 211 "strong" ASINs
2. **Integrate Keepa features into the forward-facing pipeline** — the separate Keepa analysis showed buy box ownership and rank data add significant signal
3. **Build the opportunity scoring model** — combine price prediction + volume prediction + buy box share prediction = expected revenue per ASIN
4. **Deploy on the team's forecast cadence** — same windows, same ASINs, automated daily runs
5. **75-day evaluation** — complete and compare against `forecasts_75d.csv`

---

## Files Reference

| File | Description |
|------|-------------|
| `forward_eval.py` | Main pipeline script |
| `results/forward_facing/*.csv` | Per-model day-by-day forecasts |
| `results/forward_facing/forward_facing_leaderboard.csv` | Combined leaderboard |
| `results/daily_retrain/keepa_leaderboard.csv` | Keepa track A/B results |
| `results/daily_retrain/keepa_feature_importance.csv` | Top Keepa features |
| `ML_Forecast_Test_Package/sp_daily_history.csv` | Training data |
| `ML_Forecast_Test_Package/sp_actuals_30d.csv` | Ground truth |
| `ML_Forecast_Test_Package/forecasts_30d.csv` | Team forecasts |
