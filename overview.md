## Overview

This guide shows how to run everything on your local machine: prepare data, train the models, generate predictions, and evaluate results. No server-specific steps are required.


## 1) Prerequisites

- Python 3.9+ and a working virtual environment
- Packages: `torch`, `pandas`, `numpy`, `scikit-learn`, `joblib`

Install (example):

```bash
python -m pip install --upgrade pip
python -m pip install torch pandas numpy scikit-learn joblib
```


## 2) Project layout (key files)

- `"seq2seq implementation/prepare_csv_features.py"`: feature engineering for hourly data
- `"seq2seq implementation/train_hourly_with_embeddings.py"`: trainer (works for both hourly and monthly features)
- `"seq2seq implementation/infer_hourly_with_embeddings.py"`: hourly inference (48h horizon)
- `infer_monthly_next.py`: monthly inference (iterative, configurable months)
- `evaluate_monthly.py`: monthly evaluation
- `normalize_monthly_predictions.py`: normalize prediction values (min-max or z-score)


## 3) Hourly pipeline (48-hour horizon)

### 3.1 Prepare hourly features
Input CSV must include: `measured_at;group_id;consumption;price`.

```bash
python "seq2seq implementation/prepare_csv_features.py" input_hourly.csv formatted_features.csv \
  --groups-file "challenge data/groups.md" \
  --timestamp-col measured_at --consumption-col consumption --price-col price \
  --lags 24 48 168 --scaler standard --drop-na
```

### 3.2 Train (example defaults)

```bash
python "seq2seq implementation/train_hourly_with_embeddings.py" \
  --csv formatted_features.csv \
  --epochs 30 \
  --batch-size 1024 \
  --window-size 168 --horizon 48 \
  --hidden-size 384 --layers 3 --dropout 0.2 \
  --teacher-forcing 0.3 --group-emb-dim 16 \
  --num-workers 8 --prefetch-factor 8 \
  --rolling-folds 0 \
  --save-dir artifacts/models_hourly_local
```

The best checkpoint is saved as: `artifacts/models_hourly_local/seq2seq_hourly_embeddings.pt`.

### 3.3 Inference (48h from latest) and evaluation

```bash
python "seq2seq implementation/infer_hourly_with_embeddings.py" \
  --model artifacts/models_hourly_local/seq2seq_hourly_embeddings.pt \
  --csv formatted_features.csv \
  --out predictions_hourly.csv \
  --window-size 168 --horizon 48

python "seq2seq implementation/eval_predictions.py" \
  --pred predictions_hourly.csv \
  --truth formatted_features.csv \
  --per-group --unscaled-only
```

For backtest of the last 48 hours (aligned to truth), add `--offset-hours 48` to the inference command.


## 4) Monthly pipeline (12-month horizon)

You can train directly on monthly-aggregated features. Two options:

- Option A (requires ≥24 months): `--window-size 12 --horizon 12`
- Option B (works with ~13 months): `--window-size 9 --horizon 3` and iterate forecasts 4 times to reach 12 months

### 4.1 Build monthly features (from your hourly, already prepped or raw)

Use your prepared script (or this minimal recipe):

```bash
python - <<'PY'
import pandas as pd, numpy as np
src = "formatted_hourly_or_raw.csv"   # set to your hourly source (must have measured_at, group_id, consumption, price)
out = "formatted_features_monthly.csv"
df = pd.read_csv(src, sep=';')
df['measured_at'] = pd.to_datetime(df['measured_at'])
has_price = 'price' in df.columns
g = (df.assign(month=df['measured_at'].dt.to_period('M'))
       .groupby(['group_id','month'], as_index=False)
       .agg(consumption=('consumption','sum'), **({'price':('price','mean')} if has_price else {})))
g['measured_at'] = g['month'].dt.to_timestamp(how='start')
g = g.drop(columns=['month']).sort_values(['group_id','measured_at'])
g['month_num'] = g['measured_at'].dt.month.astype(float)
g['month_sin'] = np.sin(2*np.pi*g['month_num']/12.0)
g['month_cos'] = np.cos(2*np.pi*g['month_num']/12.0)
g = g.drop(columns=['month_num'])
for k in range(1,13):
    g[f'consumption_lag_{k}m'] = g.groupby('group_id')['consumption'].shift(k)
gg = g.copy()
cm = gg.groupby('group_id')['consumption'].transform('mean')
cs = gg.groupby('group_id')['consumption'].transform('std').replace(0, np.nan)
gg['scaled_consumption'] = (gg['consumption'] - cm) / cs
if 'price' in gg.columns:
    pm = gg.groupby('group_id')['price'].transform('mean')
    ps = gg.groupby('group_id')['price'].transform('std').replace(0, np.nan)
    gg['scaled_price'] = (gg['price'] - pm) / ps
else:
    gg['price'] = 0.0; gg['scaled_price'] = 0.0
for c in ['hour_sin','hour_cos','day_of_week_sin','day_of_week_cos','is_holiday','is_weekend']:
    gg[c] = 0.0
cols=['measured_at','group_id','consumption','price','scaled_consumption','scaled_price','month_sin','month_cos']+[f'consumption_lag_{k}m' for k in range(1,13)]
gg[cols].dropna().to_csv(out, sep=';', index=False)
print(f"[done] wrote {out}, rows={len(gg.dropna())}")
PY
```

### 4.2 Train monthly

Option B example (works with ~13 months of data):
```bash
python "seq2seq implementation/train_hourly_with_embeddings.py" \
  --csv formatted_features_monthly.csv \
  --epochs 30 \
  --batch-size 2048 \
  --window-size 9 --horizon 3 \
  --hidden-size 512 --layers 3 --dropout 0.2 \
  --teacher-forcing 0.3 --group-emb-dim 16 \
  --num-workers 8 --prefetch-factor 8 \
  --rolling-folds 0 \
  --save-dir artifacts/models_monthly_w9_h3
```

### 4.3 Inference (monthly)

- Predict the next 12 months (from the latest month):
```bash
python infer_monthly_next.py \
  --ckpt artifacts/models_monthly_w9_h3/seq2seq_hourly_embeddings.pt \
  --csv formatted_features_monthly.csv \
  --months 12 \
  --out predictions_monthly_next12.csv
```

- Evaluation-aligned forecast (example: forecast the next 12 months starting after 2023‑09‑01):
```bash
python infer_monthly_next.py \
  --ckpt artifacts/models_monthly_w9_h3/seq2seq_hourly_embeddings.pt \
  --csv formatted_features_monthly.csv \
  --months 12 \
  --eval-end 2023-09-01 \
  --out predictions_monthly_eval_12m.csv
```

### 4.4 Evaluate monthly

```bash
python evaluate_monthly.py \
  --pred predictions_monthly_eval_12m.csv \
  --truth formatted_features_monthly.csv \
  --out eval_monthly_per_group_12m.csv
```


## 5) Normalizing monthly predictions (optional)

Normalize per-group to 0–1 (good for dashboards):

```bash
python normalize_monthly_predictions.py \
  --in predictions_monthly_next12.csv \
  --out predictions_monthly_next12_norm.csv \
  --method minmax --column predicted_consumption --per-group
```


## 6) Troubleshooting (quick)

- “No overlapping rows” in evaluation: align prediction months to the truth months (use `--eval-end` so forecast window fits inside the truth range).
- NaN/Inf in loss (monthly): ensure monthly lags have no missing values (drop the first 12 months per group) and per-group scaling doesn’t divide by zero.
- Shape mismatch when loading a checkpoint: make sure inference uses the same input feature columns as training (include the 6 extra zero columns for monthly: `hour_sin`, `hour_cos`, `day_of_week_sin`, `day_of_week_cos`, `is_holiday`, `is_weekend`). 


