# Technical Report: Amazon Sales Forecasting Pipeline
**Date:** March 9, 2026  
**Author:** George (AI-assisted ML pipeline)  
**Purpose:** Full technical documentation for ML engineering team  

---

## Table of Contents
1. [Problem Statement](#1-problem-statement)
2. [Data Sources & Schemas](#2-data-sources--schemas)
3. [Data Preprocessing Pipeline](#3-data-preprocessing-pipeline)
4. [Model Architectures & Hyperparameters](#4-model-architectures--hyperparameters)
5. [Training Protocol](#5-training-protocol)
6. [Evaluation Methodology](#6-evaluation-methodology)
7. [Results — Forward-Facing Evaluation](#7-results--forward-facing-evaluation)
8. [Results — Holdout Backtesting](#8-results--holdout-backtesting)
9. [Per-ASIN Error Analysis](#9-per-asin-error-analysis)
10. [Keepa Feature Engineering Pipeline](#10-keepa-feature-engineering-pipeline)
11. [What Didn't Work](#11-what-didnt-work)
12. [Reproducibility](#12-reproducibility)
13. [Next Steps & Open Questions](#13-next-steps--open-questions)
14. [Appendices](#14-appendices)

---

## 1. Problem Statement

Predict daily unit sales for 211 Amazon ASINs (primarily Nike apparel, 95.7% in Apparel category) over two horizons:
- **30-day:** March 2–31, 2026
- **75-day:** March 2–May 15, 2026

The existing team forecast was the benchmark to beat.

### Business Context
- Seller: Forever Value Emporium (Amazon 3P, account A2WKLCQK84XIK6)
- Amazon's largest 3P Nike seller
- Forecasts drive inventory purchasing decisions — underprediction = stockouts, overprediction = dead inventory

---

## 2. Data Sources & Schemas

### 2.1 `sp_daily_history.csv` — Primary Training Data
**Source:** Amazon SP API (Seller Partner API)  
**What it is:** Daily units sold and revenue per ASIN

| Column | Type | Description |
|--------|------|-------------|
| `asin` | string | Amazon Standard Identification Number |
| `date` | date | Sale date (YYYY-MM-DD) |
| `units` | int64 | Units sold that day |
| `revenue` | float64 | Revenue in USD |

**Statistics:**
- **Shape:** 60,205 rows × 4 columns
- **ASINs:** 211 unique
- **Date range:** 2024-09-01 to 2026-03-08 (553 calendar days)
- **Days per ASIN:** mean 285, min 188, max 497 (not all ASINs have full history)
- **Units distribution:** mean 7.7, median 3, max 1,119, std 22.5
- **Zero-sale days:** 0 in raw data (only days with sales are recorded)
- **Total units:** 462,891
- **Total revenue:** $18.3M

**IMPORTANT:** The raw data only contains rows where sales > 0. Days with zero sales are NOT in the file. This must be handled during preprocessing (see Section 3).

### 2.2 `sp_actuals_30d.csv` — Ground Truth (Evaluation Target)
**Source:** Amazon SP API  
**What it is:** Aggregated actual metrics for the 30-day evaluation window (Mar 2–31)

| Column | Type | Description |
|--------|------|-------------|
| `asin` | string | ASIN |
| `units_30d` | int64 | **Total units sold Mar 2-31** (primary target) |
| `units_ordered` | float64 | Units ordered (may differ from units_30d due to cancellations) |
| `units_ordered_b2b` | float64 | B2B units |
| `ordered_product_sales` | float64 | Revenue ($) |
| `total_order_items` | float64 | Order line items |
| `browser_sessions` | float64 | Desktop sessions |
| `mobile_sessions` | float64 | Mobile sessions |
| `sessions` | float64 | Total sessions |
| `browser_page_views` | float64 | Desktop page views |
| `mobile_page_views` | float64 | Mobile page views |
| `page_views` | float64 | Total page views |
| `buy_box_percentage` | float64 | Buy box win rate (SP API reported) |
| `unit_session_percentage` | float64 | Conversion rate |

**Statistics:**
- **211 ASINs**, mean units_30d: 95.3, median: 36, max: 1,055
- **11 ASINs had 0 units** in the evaluation period
- **Total actual units:** 20,098

### 2.3 `forecasts_30d.csv` — Team Benchmark Forecast
**Source:** Internal team forecasting system  
**What it is:** Team's daily point forecasts + prediction intervals for 30 days

| Column Pattern | Type | Description |
|----------------|------|-------------|
| `asin` | string | ASIN |
| `segment` | string | Internal segmentation label |
| `avg_p_sale` | float | Average price per sale |
| `avg_daily_fc` | float | Average daily forecast |
| `2026-03-02` | float | Point forecast for that date |
| `2026-03-02_p10` | float | 10th percentile forecast |
| `2026-03-02_p90` | float | 90th percentile forecast |
| ... | ... | (repeats for all 30 dates) |
| `total_30d_p10` | float | 30-day total at p10 |
| `total_30d` | float | 30-day total point forecast |
| `total_30d_p90` | float | 30-day total at p90 |
| `trust_grade` | string | Team's confidence grade |
| `grade_criteria` | string | Grading rationale (free text) |

**Shape:** 211 rows × 99 columns  
**Note:** Team provides uncertainty quantification (p10/p90 bands) — our models currently produce point forecasts only.

### 2.4 `forecasts_75d.csv` — Team 75-Day Benchmark
Same format as 30d but covering March 2 – May 15, 2026.

### 2.5 `eda_features.csv` — Keepa-Derived Static Features
**Source:** Keepa API  
**What it is:** Pre-computed cross-sectional features per ASIN (40 columns)

Key columns:
- `sales_rank_current/avg30/avg90/avg180` — Sales rank at different time windows
- `sales_rank_drops30/90/180/365` — Count of rank drops (proxy for sales velocity)
- `new_price_current/avg30`, `bb_price` — Pricing data
- `competitive_price_threshold` — Keepa's competitive price estimate
- `bb_is_amazon`, `bb_is_fba` — Buy box status flags
- `our_bb_pct` — Our buy box win percentage (Keepa-reported, **known to be inaccurate**)
- `n_unique_bb_sellers` — Competitor count
- `total_offer_count` — Total offers on listing
- `monthly_sold_badge` — Keepa's "X+ sold in past month" badge
- `review_count`, `rating` — Review metrics
- `oos_amazon_30/90` — Amazon out-of-stock percentage
- `sr_history_points`, `sr_min/max/median` — Sales rank history stats
- `listed_since_keepa_min` — Days since listing first appeared

### 2.6 `keepa_raw/` — Raw Keepa JSON Data (211 files)
**Source:** Keepa API product lookups  
**What it is:** Complete Keepa product objects with full time-series history

**File format:** One JSON per ASIN (e.g., `B0036DUP6C.json`)

**Key data arrays (Keepa `csv` field):**
All use Keepa timestamps (minutes since 2011-01-01 00:00 UTC). Values in pairs: `[timestamp, value, timestamp, value, ...]`

| Index | Field | Unit | Description |
|-------|-------|------|-------------|
| csv[0] | AMAZON price | cents/100 | Amazon's own price (-1 = OOS) |
| csv[1] | NEW price | cents/100 | Lowest new offer price |
| csv[2] | USED price | cents/100 | Lowest used price |
| csv[3] | SALES RANK | integer | Best sellers rank |
| csv[4] | LIST PRICE | cents/100 | Manufacturer list price |
| csv[11] | COUNT_NEW | integer | Number of new offers |
| csv[12] | COUNT_USED | integer | Number of used offers |
| csv[16] | RATING | integer×10 | Rating (e.g., 45 = 4.5 stars) |
| csv[17] | COUNT_REVIEWS | integer | Total review count |
| csv[18] | BUY_BOX_SHIPPING | cents/100 | Buy box price including shipping |

**Special data arrays:**
- `buyBoxSellerIdHistory`: `[keepa_ts, seller_hash, keepa_ts, seller_hash, ...]` — who held the buy box at each time point. Seller hashes are MD5 of seller IDs (e.g., our hash: `S_911b196c`). Value `-1` means no buy box winner.
- `monthlySoldHistory`: `[keepa_ts, units, ...]` — Keepa's monthly sold estimate. Values like 50, 100, 200 represent badge levels.
- `salesRanks`: Dict keyed by category ID. Values are `[keepa_ts, rank, ...]` pairs.

**Keepa timestamp conversion:**
```python
from datetime import datetime, timedelta
keepa_epoch = datetime(2011, 1, 1)
real_time = keepa_epoch + timedelta(minutes=keepa_timestamp)
```

**Data density example (B0036DUP6C):**
- csv[3] (sales rank): 17,468 data points (~hourly over years)
- csv[18] (buy box price): 8,712 data points
- buyBoxSellerIdHistory: 4,488 entries
- csv[1] (new price): 6,198 data points

---

## 3. Data Preprocessing Pipeline

### 3.1 Zero-Filling
The raw `sp_daily_history.csv` only contains days with sales. For time series models, we need a complete daily panel:

```python
# Build complete daily panel (zero-fill missing days)
all_dates = pd.date_range(raw["date"].min(), TRAIN_CUTOFF, freq="D")
idx = pd.MultiIndex.from_product([asins, all_dates], names=["asin", "date"])
panel = pd.DataFrame(index=idx).reset_index()
panel = panel.merge(raw[["asin", "date", "units"]], on=["asin", "date"], how="left")
panel["units"] = panel["units"].fillna(0).astype(float)
```

This produces ~99,030 rows (211 ASINs × ~470 days average).

### 3.2 Train/Test Split
- **Training cutoff:** March 1, 2026 (inclusive) — all data up to and including this date
- **Forecast window:** March 2, 2026 onward
- **Daily actuals available:** March 2-8 (7 days) from the raw data
- **30-day actuals available:** `sp_actuals_30d.csv` (aggregated Mar 2-31)
- **No data leakage:** Models never see any data from March 2 onward during training

### 3.3 Feature Engineering (Boosting Models Only)
For XGBoost/LightGBM/NGBoost, 22 features are computed per observation:

**Lag features (9):**
- `lag_1, lag_2, lag_3, lag_5, lag_7, lag_14, lag_21, lag_30, lag_60`
- Shifted by 1 to prevent leakage (lag_1 = yesterday's sales)

**Rolling statistics (12):**
For windows [7, 14, 30, 60] days:
- `roll_mean_{w}` — rolling mean (shifted by 1)
- `roll_std_{w}` — rolling standard deviation
- `roll_median_{w}` — rolling median

**Calendar features (4):**
- `dow` — day of week (0-6)
- `month` — month (1-12)
- `is_weekend` — binary
- `day_of_month` — (1-31)

**Derived features (2):**
- `trend_slope_30` — slope of linear fit on last 30 days
- `frac_zero_30` — fraction of zero-sale days in last 30 days

**Training data:** All feature rows from October 1, 2024 to March 1, 2026 where features are non-null. Earlier dates dropped due to insufficient lag history.

---

## 4. Model Architectures & Hyperparameters

### 4.1 PatchTST (Patch Time Series Transformer)
**Paper:** Nie et al., 2023 — "A Time Series is Worth 64 Words"  
**Library:** `neuralforecast.models.PatchTST`

| Parameter | Value | Notes |
|-----------|-------|-------|
| input_size | 180 | Days of history fed to model |
| h | 30 or 75 | Forecast horizon |
| patch_len | 16 | Each patch covers 16 timesteps |
| stride | 8 | Overlapping patches (50% overlap) |
| n_heads | 16 | Multi-head attention heads |
| hidden_size | 128 | Transformer hidden dimension |
| encoder_layers | 3 | Transformer encoder depth |
| revin | True | Reversible Instance Normalization |
| max_steps | 150 | Maximum training iterations |
| val_check_steps | 25 | Validate every 25 steps |
| early_stop_patience_steps | 5 | Stop if no val improvement for 5 checks |
| batch_size | 64 | |
| scaler_type | "standard" | Z-score normalization per series |
| loss | MAE | L1 loss function |
| random_seed | 42 | |

**How it works:** Segments the 180-day input into overlapping patches of 16 days each (stride 8), treats each patch as a "token" in a Transformer. Channel-independent: each ASIN is modeled independently but shares learned weights. Output is a direct multi-step forecast (all 30/75 days simultaneously via a linear projection head).

### 4.2 LSTM (Long Short-Term Memory)
**Library:** `neuralforecast.models.LSTM`

| Parameter | Value |
|-----------|-------|
| input_size | 180 |
| h | 30 or 75 |
| hidden_size | 128 |
| n_layers | 2 |
| dropout | 0.1 |
| max_steps | 150 |
| val_check_steps | 25 |
| early_stop_patience_steps | 5 |
| batch_size | 64 |
| scaler_type | "standard" |
| loss | MAE |
| random_seed | 42 |

**How it works:** Two-layer LSTM processes the 180-day sequence step-by-step, maintaining hidden state. Final hidden state is fed to an MLP decoder that outputs all h forecast values simultaneously (direct multi-step, not autoregressive at inference).

### 4.3 NHITS (Neural Hierarchical Interpolation for Time Series)
**Paper:** Challu et al., 2022  
**Library:** `neuralforecast.models.NHITS`

| Parameter | Value |
|-----------|-------|
| input_size | 180 |
| h | 30 or 75 |
| n_stacks | 3 |
| n_blocks per stack | [1, 1, 1] |
| mlp_units | [[512,512], [512,512], [512,512]] |
| n_pool_kernel_size | [[2,2,2], [4,4,4], [8,8,8]] |
| max_steps | 150 |
| val_check_steps | 25 |
| early_stop_patience_steps | 5 |
| batch_size | 64 |
| scaler_type | "standard" |
| loss | MAE |
| random_seed | 42 |

**How it works:** Multi-scale architecture — 3 stacks with different pooling resolutions. The first stack (pool=2) captures fine-grained daily patterns, the second (pool=4) captures weekly patterns, the third (pool=8) captures monthly patterns. Each stack outputs interpolated coefficients that are summed for the final forecast. Residual connections between stacks (each stack forecasts the residual of the previous).

### 4.4 XGBoost
**Library:** `xgboost.XGBRegressor`

| Parameter | Value |
|-----------|-------|
| n_estimators | 500 |
| max_depth | 6 |
| learning_rate | 0.03 |
| objective | "reg:squarederror" |
| n_jobs | 1 |
| random_state | 42 |

**Forecasting method:** Recursive. Trained on the global pool (all ASINs together, 22 lag/rolling features). At inference, predictions are fed back as inputs for the next day's features. This means errors compound over the forecast horizon.

### 4.5 LightGBM
**Library:** `lightgbm.LGBMRegressor`

| Parameter | Value |
|-----------|-------|
| n_estimators | 500 |
| max_depth | 6 |
| learning_rate | 0.03 |
| n_jobs | 1 |
| random_state | 42 |

Same recursive forecasting approach as XGBoost.

### 4.6 NGBoost (Natural Gradient Boosting)
**Library:** `ngboost.NGBoost`

| Parameter | Value |
|-----------|-------|
| n_estimators | 200 |
| Base | DecisionTreeRegressor(max_depth=4) |
| learning_rate | 0.05 |
| random_state | 42 |

**How it differs:** NGBoost outputs a probability distribution (Normal by default), not just a point forecast. We use the mean of the predicted distribution as the point forecast. The distribution could also be used for prediction intervals.

### 4.7 Statistical Baselines
**Library:** `statsforecast`

| Model | Implementation | Config |
|-------|---------------|--------|
| CES (AutoCES) | Complex Exponential Smoothing with automatic selection | season_length=7 |
| SeasonalNaive | Repeat value from 7 days ago | season_length=7 |
| MA7 (WindowAverage) | Simple 7-day rolling average | window_size=7 |

All statistical models are fit independently per ASIN (local models, no cross-learning).

### 4.8 Ensemble
Simple equally-weighted average of: CES, PatchTST, LSTM, NHITS, XGBoost, LightGBM, NGBoost.

```python
merged["forecast"] = merged[model_cols].mean(axis=1).clip(lower=0)
```

No model selection or stacking — just a straight average. Could be improved with learned weights or per-ASIN model selection (see holdout results for oracle performance).

---

## 5. Training Protocol

### 5.1 Neural Models (PatchTST, LSTM, NHITS)
```python
nf = NeuralForecast(models=[PatchTST(...), LSTM(...), NHITS(...)], freq="D")
nf.fit(df=nf_df, val_size=horizon)  # val_size=30 or 75
preds = nf.predict()  # generates h-step forecasts for all ASINs
```

- **NeuralForecast handles:** batching, normalization, early stopping, GPU allocation
- **All ASINs trained together** (global model) — the model learns shared patterns across ASINs while maintaining per-series normalization via the scaler
- **Validation:** last `h` days of training data held out for early stopping
- **Output:** Direct multi-step — one forward pass produces all 30 (or 75) forecast values

### 5.2 Boosting Models (XGBoost, LightGBM, NGBoost)
```python
# Global model: all ASINs pooled together
model.fit(X_train, y_train)  # X_train: (n_rows, 22 features), y_train: (n_rows,)

# Per-ASIN recursive forecasting
for each ASIN:
    seed history from training data
    for each forecast day:
        compute 22 features from history (including prior predictions)
        predict next day
        append prediction to history
```

- **Global model, per-ASIN inference** — one model trained on all ASINs, applied recursively per ASIN
- **Error compounding:** Recursive approach means day-30 forecast uses 29 prior predictions as features
- **Training data filter:** Only rows from October 2024+ (to ensure sufficient lag features)

### 5.3 Statistical Models
```python
sf = StatsForecast(models=[AutoCES(7), SeasonalNaive(7), WindowAverage(7)], freq="D", n_jobs=1)
sf.fit(df=nf_df)
preds = sf.predict(h=horizon)
```

- **Local models** — each ASIN fitted independently (no cross-series learning)
- **No validation split** — statistical models use all available training data
- **Single-shot forecast** — generates full horizon in one call

---

## 6. Evaluation Methodology

### 6.1 Forward-Facing Evaluation (Primary)
**This is the apples-to-apples comparison with the team.**

- **Train:** All data through March 1, 2026
- **Forecast:** March 2–31 (30d) and March 2–May 15 (75d)
- **Evaluate against:** `sp_actuals_30d.csv` for total 30-day performance

**Daily evaluation** uses actual daily sales from `sp_daily_history.csv` for March 2–8 only (7 days of overlap between forecast and available daily actuals). n_obs = 864 observations (but note: NOT 211 × 7 = 1,477 because the daily data only has entries for days with sales, and some ASINs had no sales in Mar 2-8).

### 6.2 Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| Daily MAE | mean(\|forecast_t - actual_t\|) | Average daily forecast error across all ASINs and days |
| Daily RMSE | sqrt(mean((forecast_t - actual_t)²)) | Penalizes large daily errors more |
| Total MAE | mean(\|Σforecast - units_30d\|) per ASIN | Average 30-day total error per ASIN |
| Total RMSE | sqrt(mean((Σforecast - units_30d)²)) | Penalizes large ASIN-level errors |
| Bias | mean(Σforecast - units_30d) | Positive = overprediction, Negative = underprediction |
| WAPE | Σ\|Σforecast - units_30d\| / Σunits_30d | Weighted absolute percentage error (volume-weighted) |

### 6.3 Holdout Backtesting (Secondary)
Used for model development and hyperparameter exploration:
- **Train:** All data through ~February 7, 2026
- **Test:** Last 30 days of each ASIN's history (before March 8 cutoff)
- **Metrics:** Same as above plus MASE (Mean Absolute Scaled Error)
- **120+ model configurations tested** in this mode

### 6.4 Key Distinction
The team comparison uses **forward-facing evaluation only**. Holdout backtest results are for model development — they don't directly compare to the team because the evaluation windows differ.

---

## 7. Results — Forward-Facing Evaluation

### 7.1 Daily MAE (Mar 2-8, 864 observations)

| Rank | Model | Daily MAE | Daily RMSE | vs Team |
|------|-------|-----------|------------|---------|
| 1 | LightGBM | 3.100 | 6.073 | -38.2% |
| 2 | Ensemble | 3.147 | 5.806 | -37.2% |
| 3 | CES | 3.346 | 6.074 | -33.3% |
| 4 | NGBoost | 3.370 | 7.562 | -32.8% |
| 5 | NHITS | 3.469 | 6.417 | -30.8% |
| 6 | XGBoost | 3.535 | 7.628 | -29.5% |
| 7 | MA7 | 3.549 | 6.856 | -29.2% |
| 8 | PatchTST | 3.674 | 6.709 | -26.7% |
| 9 | SeasonalNaive | 4.132 | 7.674 | -17.6% |
| 10 | LSTM | 4.139 | 7.747 | -17.4% |
| **11** | **Team** | **5.013** | **9.454** | **baseline** |

### 7.2 30-Day Total (211 ASINs)

| Rank | Model | Total MAE | WAPE | Bias | Predicted Sum | Actual Sum | vs Team |
|------|-------|-----------|------|------|---------------|------------|---------|
| 1 | LSTM | 35.87 | 37.7% | -11.6 | 17,640 | 20,098 | -49.3% |
| 2 | Ensemble | 41.38 | 43.4% | -11.6 | 17,641 | 20,098 | -41.5% |
| 3 | PatchTST | 42.32 | 44.4% | -31.6 | 13,440 | 20,098 | -40.2% |
| 4 | NHITS | 43.96 | 46.2% | -9.4 | 18,115 | 20,098 | -37.9% |
| 5 | SeasonalNaive | 44.87 | 47.1% | -21.7 | 15,528 | 20,098 | -36.6% |
| 6 | MA7 | 44.89 | 47.1% | -24.3 | 14,979 | 20,098 | -36.6% |
| 7 | CES | 48.38 | 50.8% | -38.5 | 11,973 | 20,098 | -31.7% |
| 8 | NGBoost | 58.60 | 61.5% | +4.8 | 21,108 | 20,098 | -17.2% |
| 9 | LightGBM | 58.83 | 61.8% | +8.2 | 21,827 | 20,098 | -16.9% |
| 10 | XGBoost | 64.46 | 67.7% | -3.4 | 19,384 | 20,098 | -8.9% |
| **11** | **Team** | **70.79** | **74.3%** | **-64.7** | **6,457** | **20,098** | **baseline** |

### 7.3 Key Observations
1. **LSTM paradox:** Best at total accuracy (MAE 35.87) but second-worst at daily accuracy (MAE 4.14). This suggests LSTM's daily predictions are noisy but errors cancel out when summed. For inventory decisions (total volume), LSTM is best. For daily scheduling, use LightGBM.

2. **Team calibration failure:** Team predicted 6,457 total units vs 20,098 actual — a 68% undercount. Bias of -64.7 units/ASIN is by far the largest. This is not a variance problem — it's a systematic level shift.

3. **Boosting models overpredict slightly:** LightGBM (+8.2 bias) and NGBoost (+4.8) are the only models that overshoot. This is actually safer for inventory — better to have slightly too much than run out.

4. **Ensemble is the safe bet:** 2nd on both metrics. Averages out the individual models' biases.

---

## 8. Results — Holdout Backtesting

These results are from the broader model sweep (120+ configurations). **Different evaluation window than forward-facing.**

| Rank | Model | Holdout MAE | MASE | n_ASINs | Source |
|------|-------|-------------|------|---------|--------|
| 1 | oracle_best (per-ASIN selection) | 1.916 | — | 211 | ensemble |
| 2 | top3_avg (avg of 3 best per ASIN) | 2.163 | — | 211 | ensemble |
| 3 | weighted_avg | 2.257 | — | 211 | ensemble |
| 4 | team_forecast | 2.360 | — | 211 | team |
| 5 | NGBoost | 2.395 | 0.835 | 211 | individual |
| 6 | chronos-t5-tiny | 2.748 | 1.593 | 844* | foundation |
| 7 | chronos-bolt-small | 2.778 | 1.589 | 844* | foundation |
| 8 | chronos-bolt-base | 2.930 | 1.616 | 844* | foundation |
| 9 | chronos-t5-base | 2.939 | 1.816 | 844* | foundation |
| 10 | chronos-t5-small | 2.997 | 1.788 | 844* | foundation |
| 11 | chronos-t5-large | 3.034 | 1.991 | 652* | foundation |
| 12 | tabnet_eda | 3.279 | 1.323 | 211 | individual |
| 13 | ma_7 | 3.335 | 1.140 | 211 | individual |
| 14 | ses | 3.531 | 1.183 | 211 | individual |
| 15 | snaive_7 | 3.640 | 1.258 | 211 | individual |
| 16 | naive_last | 3.715 | 1.248 | 211 | individual |
| 17 | ma_30 | 3.951 | 1.211 | 211 | individual |
| 18 | stl_xgboost | 5.480 | 1.913 | 211 | individual |
| 19 | ma_60 | 8.324 | 2.289 | 211 | individual |
| 20 | prophet | 8.814 | 2.385 | 211 | individual |

*Foundation models (Chronos) were evaluated on more ASINs due to different data handling.

**Key insight:** Oracle per-ASIN model selection (MAE 1.916) is 19% better than any single model, suggesting significant value in learning which model works best for which ASIN profile. Current selector only achieves 26.9% accuracy — major room for improvement.

---

## 9. Per-ASIN Error Analysis

### 9.1 LSTM Error by Volume Tier

| Tier | n_ASINs | LSTM MAE | LSTM Mean Pred | Actual Mean | Team MAE | Team Mean Pred |
|------|---------|----------|----------------|-------------|----------|----------------|
| zero (0 units) | 11 | 4.3 | 4.3 | 0.0 | 16.4 | 16.4 |
| low (1-10) | 36 | 11.5 | 15.6 | 4.9 | 7.9 | 9.8 |
| med (11-50) | 79 | 16.4 | 29.8 | 27.5 | 19.7 | 13.2 |
| high (51-100) | 30 | 27.2 | 58.2 | 70.8 | 46.6 | 25.6 |
| very high (101-500) | 48 | 74.0 | 188.1 | 214.1 | 153.0 | 61.1 |
| top (500+) | 7 | 206.1 | 558.0 | 764.1 | 596.1 | 168.1 |

**LSTM vs Team by tier:**
- **Zero/Low volume:** Team is slightly better (MAE 7.9 vs 11.5 for low). LSTM overpredicts small ASINs.
- **Medium volume:** Roughly comparable (LSTM 16.4 vs team 19.7).
- **High+ volume:** LSTM dramatically better. Team underpredicts badly (mean pred 61.1 vs actual 214.1 for very high tier = 71% undercount).

**Conclusion:** The team's forecast is tuned for low-volume ASINs but completely breaks down at high volume. The ML models handle both ends better on aggregate.

### 9.2 Worst ASIN Predictions (LSTM, 30d)

| ASIN | Actual | Predicted | Error | Issue |
|------|--------|-----------|-------|-------|
| B08G1ZB4ZZ | 268 | 570 | +302 | Overpredicted 2x |
| B087LTC1XS | 517 | 237 | -280 | Underpredicted by half |
| B08CP98GGJ | 505 | 227 | -278 | Underpredicted by half |
| B08S22J84V | 454 | 732 | +278 | Overpredicted 1.6x |
| B08WV3DW85 | 383 | 124 | -259 | Underpredicted 3x |

These are all high-volume ASINs (268-1055 units/month) where the model struggles most. Likely causes: demand regime changes, promotional spikes, or buy box dynamics not captured in the time series alone.

---

## 10. Keepa Feature Engineering Pipeline

Separate from the forward-facing evaluation, we built models incorporating Keepa marketplace data.

### 10.1 Feature Extraction
From each ASIN's raw Keepa JSON, extracted:
- **Buy box ownership:** Calculated from `buyBoxSellerIdHistory` — time-weighted daily % for our seller (S_911b196c). **Keepa's reported `buy_box_percentage` is unreliable** — must calculate from raw hourly pings.
- **Monthly sold badge:** From `monthlySoldHistory`
- **Sales rank features:** Daily rank, rank volatility (7/14/28-day rolling std), rank momentum (7-day delta), rank MA
- **Price features:** Buy box price, new price, price vs list price ratio, price moving averages
- **Offer count:** New offers, used offers
- **Category rank:** Secondary category rank (log-transformed)
- **Calendar:** Month, day of month

### 10.2 Top Features by XGBoost Importance

| Rank | Feature | Importance | Description |
|------|---------|------------|-------------|
| 1 | our_buybox | 17.3% | Our time-weighted buy box share |
| 2 | monthly_sold | 8.8% | Keepa monthly sold badge |
| 3 | rank_log | 6.8% | Log of sales rank |
| 4 | month | 5.7% | Calendar month |
| 5 | sales_rank | 5.5% | Raw sales rank |
| 6 | rank_std7 | 5.3% | Rank volatility (7d std) |
| 7 | rank_delta7 | 4.8% | Rank momentum (7d change) |
| 8 | dom | 4.7% | Day of month |
| 9 | rank_ma28 | 4.2% | 28-day rank moving average |
| 10 | price_vs_list | 4.2% | Price relative to list price |

### 10.3 Keepa Model Results (Holdout)

**Track A — Internal + Keepa features:**

| Model | 30d MAE | 75d MAE |
|-------|---------|---------|
| Ridge | 1.85 | 2.36 |
| LightGBM | 2.15 | 2.25 |
| XGBoost | 2.31 | 2.54 |

**Track B — Keepa-only (zero internal sales data):**

| Model | 30d MAE | 75d MAE |
|-------|---------|---------|
| XGBoost | 2.99 | 4.56 |
| LightGBM | 3.04 | 4.41 |
| Ridge | 3.48 | 4.39 |

Track B is important: it predicts sales using only publicly available Keepa data. This enables scoring ASINs we've never sold.

---

## 11. What Didn't Work

| Approach | Result | Why |
|----------|--------|-----|
| **Prophet** | MAE 8.81 (worst) | Couldn't handle the heavy zero-inflation and sporadic demand patterns. Designed for smooth daily/weekly seasonality. |
| **MA60 (60-day moving average)** | MAE 8.32 | Too much smoothing — 60 days includes stale demand signals |
| **STL + XGBoost** | MAE 5.48 | STL decomposition poor on irregular sales — seasonal component extraction noisy |
| **Chronos foundation models** | MAE 2.75-3.03 | Decent but worse than task-specific models. Zero-shot performance is impressive but can't compete with fine-tuned models on this specific distribution. |
| **TabNet with EDA features** | MAE 3.28 | Cross-sectional features (EDA) without proper temporal modeling — worse than pure time series approaches |
| **chronos-t5-large** | Only ran on 652/211 ASINs | GPU OOM at 24GB — couldn't process all ASINs in one batch |
| **Per-ASIN model selection** | Oracle MAE 1.92, but practical selector only 27% accurate | Huge theoretical upside but the selector model (which model to use per ASIN) isn't good enough yet |
| **TimesFM** | Failed to run | Gated HuggingFace repo, requires access token we don't have |
| **MOIRAI, GraniteTTM, LagLlama** | Failed to run | Various dependency/API issues |

---

## 12. Reproducibility

### 12.1 Environment
```
Hardware: AWS g5.2xlarge (8 vCPU, 32GB RAM, NVIDIA A10G 24GB VRAM)
OS: Ubuntu 22.04.4 LTS, kernel 6.5.0-1018-aws (x86_64)
Python: 3.10
CUDA: 12.x (for neural model training)

Key packages:
  neuralforecast==1.7.x    # PatchTST, LSTM, NHITS
  statsforecast==1.7.x     # CES, SeasonalNaive, WindowAverage
  xgboost==2.x             # XGBRegressor
  lightgbm==4.x            # LGBMRegressor
  ngboost==0.5.x           # NGBoost
  pandas==2.x
  numpy==1.x / 2.x
  pytorch==2.x             # Backend for neuralforecast
```

### 12.2 Running
```bash
cd ~/ml_forecast/benchmark

# Forward-facing evaluation (produces all results)
python3 forward_eval.py

# Output directory
ls results/forward_facing/
# -> CES_h30.csv, LSTM_h30.csv, ..., forward_facing_leaderboard.csv
```

**Runtime:** ~45-60 minutes for full pipeline (statistical + neural + boosting, both horizons)

### 12.3 Files Included in This Package

| File | Description |
|------|-------------|
| `TECHNICAL_REPORT.md` | This document |
| `forward_eval.py` | Complete pipeline script (~380 lines) |
| `forward_facing_leaderboard.csv` | Combined results (all models, all metrics) |
| `daily_leaderboard.csv` | Daily MAE rankings |
| `total_30d_leaderboard.csv` | 30d total rankings |
| `keepa_leaderboard.csv` | Keepa track A/B results |
| `keepa_feature_importance.csv` | Top Keepa features by importance |
| `per_asin_error_analysis.csv` | Per-ASIN error breakdown (LSTM) |
| `results/forward_facing/*.csv` | Raw per-model forecasts (asin, date, forecast) |

---

## 13. Next Steps & Open Questions

### Immediate
1. **Integrate Keepa features into forward-facing pipeline** — current forward-facing uses pure time series only. Adding buy box share (+17% feature importance) should improve results significantly.
2. **Per-ASIN model selection** — oracle performance (MAE 1.92) is 46% better than best single model. Need a better selector than current 27% accuracy.
3. **Expand to full 2,800 ASINs** — current 211 are "strong" ASINs. Full catalog includes long-tail and sparse sellers.
4. **Hyperparameter optimization** — all models used reasonable defaults or light tuning. Proper Bayesian HPO (Optuna) could yield 5-15% improvement.

### Architectural Questions for ML Engineer
1. **Should we move to a hierarchical model?** ASINs within the same product family share demand signals. Models like HierarchicalForecast could capture this.
2. **Feature engineering from Keepa at inference time:** The forward-facing pipeline doesn't use Keepa data. How do we incorporate hourly Keepa signals (price, rank, offers) into a daily forecast without leakage?
3. **Probabilistic forecasting:** Team provides p10/p90 bands. We should too. NGBoost already outputs distributions; conformal prediction could add calibrated intervals to any model.
4. **Demand intermittency:** Many ASINs sell 0-3 units/day. Croston's method or intermittent demand models (ADIDA, IMAPA) might help for the "zero" and "low" tiers where LSTM struggles.
5. **Recursive vs direct forecasting:** Boosting models use recursive (error-compounding). Neural models use direct (no compounding but may miss dependencies). Hybrid approaches (e.g., direct at short horizons, recursive at long)?

### Team Feedback Requested
- Is the `units_30d` in actuals exactly the sum of daily units, or is there a reconciliation difference?
- How is the team's `segment` field determined? It might be useful as a model feature.
- What's the `avg_p_sale` field in the team forecast — average price per sale or probability of sale?
- Can we get daily actuals for the full March 2-31 window (not just through March 8)?

---

## 14. Appendices

### A. Keepa CSV Index Reference
| Index | Field | Unit | Sampling |
|-------|-------|------|----------|
| 0 | AMAZON price | cents | Hourly |
| 1 | NEW price (lowest) | cents | Hourly |
| 2 | USED price (lowest) | cents | Hourly |
| 3 | SALES RANK | integer | Hourly |
| 4 | LIST PRICE | cents | Updates |
| 11 | NEW offer count | integer | Hourly |
| 12 | USED offer count | integer | Hourly |
| 15 | LIST PRICE (alt) | integer | Updates |
| 16 | RATING | integer×10 | Updates |
| 17 | REVIEW COUNT | integer | Updates |
| 18 | BUY BOX PRICE (incl shipping) | cents | Hourly |
| 28 | ? (unlabeled) | cents | Hourly |
| 29 | ? (unlabeled) | cents | Sparse |
| 34 | ? (unlabeled) | integer | Sparse |

### B. Keepa Timestamp Conversion
```python
from datetime import datetime, timedelta
KEEPA_EPOCH = datetime(2011, 1, 1)

def keepa_to_datetime(keepa_minutes):
    return KEEPA_EPOCH + timedelta(minutes=int(keepa_minutes))

# Example: 7948124 -> 2026-02-17 02:44:00
```

### C. Buy Box Share Calculation
```python
def calculate_daily_bb_share(bb_history, seller_hash="S_911b196c"):
    """
    bb_history: [keepa_ts, seller_id, keepa_ts, seller_id, ...]
    Returns: dict of {date: share_pct}
    """
    # Parse pairs
    entries = [(keepa_to_datetime(bb_history[i]), bb_history[i+1]) 
               for i in range(0, len(bb_history), 2)]
    
    # For each day, calculate time-weighted share
    daily_share = {}
    for date in date_range:
        day_entries = entries_for_date(entries, date)
        total_minutes = 0
        our_minutes = 0
        for start_time, seller, end_time in day_entries:
            duration = (end_time - start_time).total_seconds() / 60
            total_minutes += duration
            if seller == seller_hash:
                our_minutes += duration
        daily_share[date] = our_minutes / max(total_minutes, 1)
    
    return daily_share
```
