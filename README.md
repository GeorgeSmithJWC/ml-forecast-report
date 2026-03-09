# ML Sales Forecasting Pipeline — Complete Technical Package

## One-Command Reproduction

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the entire pipeline (trains all models, generates all forecasts, evaluates)
python3 forward_eval.py

# Output: results/ directory with all forecasts and leaderboards
```

**Expected runtime:** 45-60 minutes on a machine with 8+ cores and 32GB RAM.  
**GPU:** Optional. Neural models will use GPU if available (CUDA), falls back to CPU.  
**Original hardware:** AWS g5.2xlarge (8 vCPU, 32GB RAM, NVIDIA A10G 24GB).

---

## What This Is

A forecasting pipeline that predicts daily unit sales for 211 Amazon ASINs (primarily Nike apparel). We tested 10 model architectures against the existing team forecast. **Every model beat the team** on the same March 2-31, 2026 evaluation window.

**Headline results:**
- Best daily accuracy: LightGBM (MAE 3.10 vs team 5.01, **-38%**)
- Best total accuracy: LSTM (MAE 35.9 vs team 70.8, **-49%**)
- Team predicted 6,457 total units vs 20,098 actual — **32% of reality**

---

## Directory Structure

```
.
├── README.md                          # This file
├── TECHNICAL_REPORT.md                # Deep technical documentation (33KB)
├── requirements.txt                   # Exact Python dependencies
├── forward_eval.py                    # Complete pipeline script (run this)
├── data/
│   ├── sp_daily_history.csv           # Training data: daily units per ASIN
│   ├── sp_actuals_30d.csv             # Ground truth: Mar 2-31 actual units
│   ├── forecasts_30d.csv              # Team's 30-day forecast (benchmark)
│   ├── forecasts_75d.csv              # Team's 75-day forecast (benchmark)
│   └── eda_features.csv              # Keepa-derived static features (40 cols)
├── results/
│   ├── forward_facing_leaderboard.csv # Combined leaderboard
│   ├── daily_leaderboard.csv          # Daily MAE rankings
│   ├── total_30d_leaderboard.csv      # 30d total rankings
│   └── forecasts/                     # Per-model forecast CSVs
│       ├── LSTM_h30.csv               # (asin, date, forecast) per model
│       ├── LightGBM_h30.csv
│       └── ...
├── analysis/
│   ├── per_asin_error_analysis.csv    # Error breakdown per ASIN per model
│   ├── holdout_leaderboard.csv        # Broader model sweep results (120+ models)
│   ├── keepa_leaderboard.csv          # Keepa pipeline results
│   └── keepa_feature_importance.csv   # Top Keepa features by importance
└── TEAM_REPORT.md                     # Executive summary (shorter version)
```

---

## Data Files — Complete Schema Reference

### `data/sp_daily_history.csv` — Primary Training Data
**Source:** Amazon SP API  
**60,205 rows × 4 columns**

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `asin` | str | Amazon ASIN | B0036DUP6C |
| `date` | date | YYYY-MM-DD | 2024-09-04 |
| `units` | int | Units sold that day | 2 |
| `revenue` | float | Revenue (USD) | 101.77 |

**Key stats:**
- 211 unique ASINs
- Date range: 2024-09-01 to 2026-03-08 (553 calendar days)
- Days per ASIN: mean 285, min 188, max 497 (not all ASINs have full history)
- Units: mean 7.7, median 3, max 1,119, std 22.5
- **CRITICAL:** Only rows with sales > 0 are included. Zero-sale days are MISSING. The pipeline zero-fills during preprocessing (see forward_eval.py `load_data()`).
- Total units: 462,891 | Total revenue: $18.3M

### `data/sp_actuals_30d.csv` — Ground Truth for Evaluation
**Source:** Amazon SP API  
**211 rows × 14 columns**

| Column | Type | Description |
|--------|------|-------------|
| `asin` | str | ASIN |
| `units_30d` | int | **PRIMARY TARGET:** Total units sold Mar 2-31 |
| `units_ordered` | float | Units ordered (may differ due to cancellations) |
| `units_ordered_b2b` | float | B2B units |
| `ordered_product_sales` | float | Revenue ($) |
| `total_order_items` | float | Order line items |
| `browser_sessions` | float | Desktop sessions |
| `mobile_sessions` | float | Mobile sessions |
| `sessions` | float | Total sessions |
| `browser_page_views` | float | Desktop page views |
| `mobile_page_views` | float | Mobile page views |
| `page_views` | float | Total page views |
| `buy_box_percentage` | float | Buy box win rate (%) |
| `unit_session_percentage` | float | Conversion rate (%) |

**Key stats:**
- units_30d: mean 95.3, median 36, max 1,055
- 11 ASINs had 0 sales in the evaluation period
- Total actual units: 20,098
- Some columns have NaN for zero-sale ASINs

### `data/forecasts_30d.csv` — Team Benchmark (30 days)
**Source:** Internal team forecasting system  
**211 rows × 99 columns**

| Column Pattern | Type | Description |
|----------------|------|-------------|
| `asin` | str | ASIN |
| `segment` | str | Internal segmentation label (e.g., "A", "B", "C") |
| `avg_p_sale` | float | Average price per sale |
| `avg_daily_fc` | float | Team's average daily forecast |
| `2026-03-02` | float | Point forecast for March 2 |
| `2026-03-02_p10` | float | 10th percentile (lower bound) |
| `2026-03-02_p90` | float | 90th percentile (upper bound) |
| ... | ... | (repeats for all 30 dates: Mar 2-31) |
| `total_30d_p10` | float | 30-day total at p10 |
| `total_30d` | float | 30-day total point forecast |
| `total_30d_p90` | float | 30-day total at p90 |
| `trust_grade` | str | Confidence grade |
| `grade_criteria` | str | Free text grading rationale |

**Team total predicted: 6,457 units vs actual 20,098 (68% under-forecast)**

### `data/forecasts_75d.csv` — Team Benchmark (75 days)
Same schema as 30d, covering March 2 – May 15, 2026.

### `data/eda_features.csv` — Keepa-Derived Static Features
**Source:** Keepa API  
**211 rows × 40 columns**

| Column | Type | Description |
|--------|------|-------------|
| `asin` | str | ASIN |
| `title` | str | Product title |
| `brand` | str | Brand name |
| `parent_asin` | str | Parent/variation ASIN |
| `category_tree` | str | Full category path |
| `product_group` | str | Amazon product group |
| `size` | str | Size variant |
| `color` | str | Color variant |
| `model` | str | Model number |
| `variation_count` | int | Number of variations |
| `sales_rank_current` | float | Current BSR |
| `sales_rank_avg30` | float | 30-day average BSR |
| `sales_rank_avg90` | float | 90-day average BSR |
| `sales_rank_avg180` | float | 180-day average BSR |
| `sales_rank_drops30` | float | Rank drops in 30d (proxy for sales velocity) |
| `sales_rank_drops90` | float | Rank drops in 90d |
| `sales_rank_drops180` | float | Rank drops in 180d |
| `sales_rank_drops365` | float | Rank drops in 365d |
| `new_price_current` | float | Current lowest new price (cents) |
| `new_price_avg30` | float | 30-day avg new price |
| `bb_price` | float | Current buy box price |
| `competitive_price_threshold` | float | Keepa competitive price threshold |
| `bb_is_amazon` | bool | Amazon holds buy box? |
| `bb_is_fba` | bool | Buy box is FBA? |
| `our_bb_pct` | float | Our buy box win % (**Keepa's number — known inaccurate**) |
| `n_unique_bb_sellers` | int | Unique buy box winners |
| `total_offer_count` | int | Total active offers |
| `monthly_sold_badge` | float | "X+ sold in past month" badge |
| `monthly_sold_delta90` | float | Change in monthly sold over 90d |
| `monthly_sold_last3` | str | Last 3 monthly sold values |
| `review_count` | int | Total reviews |
| `rating` | float | Star rating (1-5) |
| `oos_amazon_30` | float | Amazon OOS % in 30d |
| `oos_amazon_90` | float | Amazon OOS % in 90d |
| `sr_history_points` | int | Sales rank data points |
| `sr_min` | float | Historical min rank |
| `sr_max` | float | Historical max rank |
| `sr_median` | float | Historical median rank |
| `listed_since_keepa_min` | int | Keepa timestamp of first listing |
| `data_pulled_at` | str | When data was pulled |

**Note:** `our_bb_pct` is Keepa's pre-computed buy box percentage which is known to be inaccurate. For accurate buy box share, calculate from raw `buyBoxSellerIdHistory` in raw Keepa JSONs (not included in this package — see TECHNICAL_REPORT.md for calculation methodology).

---

## Pipeline Details (`forward_eval.py`)

### Preprocessing
1. Load `sp_daily_history.csv`
2. Create complete daily panel: all 211 ASINs × all dates from first sale to March 1
3. Zero-fill: any missing (asin, date) pair gets `units=0`
4. Result: ~99,030 rows (dense panel)

### Train/Test Split
- **Train:** Everything through March 1, 2026 (inclusive)
- **Test:** March 2 onward
- **No leakage:** Models never see March 2+ data during training

### Models Trained (10 architectures)

**Statistical (statsforecast library):**
1. **CES (AutoCES)** — Complex Exponential Smoothing, season_length=7
2. **SeasonalNaive** — Repeat value from 7 days ago
3. **MA7 (WindowAverage)** — 7-day moving average

**Neural (neuralforecast library, PyTorch backend):**
4. **PatchTST** — Transformer with patched input (Nie et al., 2023)
5. **LSTM** — 2-layer Long Short-Term Memory
6. **NHITS** — Neural Hierarchical Interpolation (Challu et al., 2022)

**Boosting (gradient boosting with recursive forecasting):**
7. **XGBoost** — xgboost.XGBRegressor
8. **LightGBM** — lightgbm.LGBMRegressor
9. **NGBoost** — Natural Gradient Boosting (probabilistic)

**Meta:**
10. **Ensemble** — Equal-weight average of models 1, 4-9

### Boosting Feature Engineering (22 features)

```
Lag features (9):     lag_1, lag_2, lag_3, lag_5, lag_7, lag_14, lag_21, lag_30, lag_60
Rolling mean (4):     roll_mean_7, roll_mean_14, roll_mean_30, roll_mean_60
Rolling std (4):      roll_std_7, roll_std_14, roll_std_30, roll_std_60
Rolling median (4):   roll_median_7, roll_median_14, roll_median_30, roll_median_60
Calendar (4):         dow, month, is_weekend, day_of_month
Derived (2):          trend_slope_30, frac_zero_30
```

All lag/rolling features are shifted by 1 to prevent leakage (lag_1 = yesterday's actual or predicted value).

### Boosting Forecast Method: Recursive
```
For each forecast day t:
  1. Compute features from history (actual data + prior predictions)
  2. Predict units_t
  3. Append prediction to history
  4. Repeat for t+1
```
This means errors compound — day 30's forecast depends on 29 prior predictions used as features.

### Neural Forecast Method: Direct Multi-Step
```
Model takes 180-day input window → outputs all h values at once
No recursive feedback — each forecast day is independent
```

### Evaluation Metrics
| Metric | Formula | What it measures |
|--------|---------|-----------------|
| Daily MAE | mean(|forecast_t - actual_t|) per day per ASIN | Average daily error |
| Daily RMSE | sqrt(mean((forecast_t - actual_t)²)) | Daily error, penalizing outliers |
| Total MAE | mean(|Σforecast - units_30d|) per ASIN | 30-day total error per ASIN |
| Total RMSE | sqrt(mean((Σforecast - units_30d)²)) | Total error, penalizing outliers |
| Bias | mean(Σforecast - units_30d) per ASIN | Systematic over/under prediction |
| WAPE | Σ|Σforecast - units_30d| / Σunits_30d | Volume-weighted % error |

**Daily evaluation** uses March 2-8 daily actuals (only 7 days available in sp_daily_history.csv after the training cutoff). n_obs = 864 (not 211×7=1,477 because only days with sales are recorded).

**Total evaluation** uses `sp_actuals_30d.csv` which has the complete Mar 2-31 ground truth.

---

## Full Hyperparameter Reference

### PatchTST
```python
PatchTST(
    h=30,              # forecast horizon (or 75)
    input_size=180,    # lookback window (days)
    patch_len=16,      # timesteps per patch
    stride=8,          # patch overlap (50%)
    n_heads=16,        # attention heads
    hidden_size=128,   # transformer hidden dim
    encoder_layers=3,  # transformer depth
    revin=True,        # reversible instance normalization
    max_steps=150,     # max training iterations
    val_check_steps=25,
    early_stop_patience_steps=5,
    batch_size=64,
    scaler_type="standard",  # z-score per series
    loss=MAE(),        # L1 loss
    random_seed=42,
)
```

### LSTM
```python
LSTM(
    h=30,
    input_size=180,
    hidden_size=128,
    n_layers=2,
    dropout=0.1,
    max_steps=150,
    val_check_steps=25,
    early_stop_patience_steps=5,
    batch_size=64,
    scaler_type="standard",
    loss=MAE(),
    random_seed=42,
)
```

### NHITS
```python
NHITS(
    h=30,
    input_size=180,
    n_stacks=3,
    n_blocks=[1, 1, 1],
    mlp_units=[[512, 512], [512, 512], [512, 512]],
    n_pool_kernel_size=[[2, 2, 2], [4, 4, 4], [8, 8, 8]],
    max_steps=150,
    val_check_steps=25,
    early_stop_patience_steps=5,
    batch_size=64,
    scaler_type="standard",
    loss=MAE(),
    random_seed=42,
)
```

### XGBoost
```python
xgb.XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.03,
    objective="reg:squarederror",
    n_jobs=1,
    random_state=42,
)
```

### LightGBM
```python
lgb.LGBMRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.03,
    n_jobs=1,
    random_state=42,
)
```

### NGBoost
```python
NGBoost(
    n_estimators=200,
    Base=DecisionTreeRegressor(max_depth=4),
    learning_rate=0.05,
    random_state=42,
)
```

---

## Environment Variable Requirements

```bash
export CUDA_VISIBLE_DEVICES=""          # Force CPU for LightGBM (cupy/dask conflict)
export NIXTLA_ID_AS_COL="1"             # NeuralForecast column naming
```

Both are set automatically in `forward_eval.py`.

If you have a GPU and want neural models to use it, modify the script to only set `CUDA_VISIBLE_DEVICES=""` for boosting models and unset it for neural models.

---

## Known Issues & Gotchas

1. **LightGBM + CUDA conflict:** If LightGBM is built with CUDA support and you have GPU libraries installed, it can conflict with cupy/dask. Setting `CUDA_VISIBLE_DEVICES=""` at the process level avoids this but forces all models to CPU.

2. **Zero-filling creates misleading stats:** After zero-filling, many ASINs show long runs of zeros. This is real (they genuinely didn't sell) but can confuse models. The neural models handle this via their scaler; boosting models handle it via the `frac_zero_30` feature.

3. **Daily actuals only go to March 8:** We can only evaluate daily predictions for 7 days. The 30-day total is the more reliable evaluation.

4. **Team forecasts are in wide format:** Each date is a column. The pipeline melts this to long format for evaluation.

5. **LSTM paradox:** Best at 30d totals (MAE 35.9), worst at daily (MAE 4.14). Its daily predictions are noisy but errors cancel when summed. Choose your model based on your use case (inventory planning = LSTM, daily scheduling = LightGBM).

6. **Recursive error compounding:** Boosting models compound errors over the horizon. Day 30's prediction quality depends on days 1-29's predictions. This is why LightGBM is great at daily but worse at 30d totals than LSTM.

---

## What's NOT in This Package

1. **Raw Keepa JSON files** (211 files, ~500MB) — too large. The `eda_features.csv` contains pre-extracted features. For buy box analysis or custom feature engineering, you need the raw Keepa data (contact Jake).

2. **75-day ground truth** — we don't have it yet (evaluation period hasn't ended). The 75d results are forecasts only.

3. **Hyperparameter optimization results** — we used reasonable defaults/light tuning. No Optuna/Bayesian HPO was run. Estimated 5-15% improvement potential.

4. **The broader 120+ model holdout sweep** — see `analysis/holdout_leaderboard.csv` for results. Those models were evaluated on a different time window (not forward-facing).

5. **The Keepa-only model (Track B)** — uses raw Keepa data for prediction without any internal sales data. Separate pipeline. Results in `analysis/keepa_leaderboard.csv`.

---

## For Further Questions

See `TECHNICAL_REPORT.md` for:
- Complete Keepa data format reference (csv indices, timestamp conversion, buy box calculation)
- Per-ASIN error analysis with volume tier breakdown
- What didn't work and why (Prophet, MA60, STL, etc.)
- Next steps and open architectural questions
