# Fortum Challenge 2025 - Methodology Document

## Executive Summary

This document describes our approach to forecasting electricity consumption for Fortum's customer groups across two time horizons: 48-hour hourly forecasts and 12-month monthly forecasts. We developed a deep learning solution using Sequence-to-Sequence (Seq2Seq) models with GRU encoders and group embeddings to capture both temporal patterns and group-specific consumption behaviors.

Our solution directly addresses Fortum's operational needs: accurate 48-hour forecasts enable optimal day-ahead market bidding and reduce imbalance costs, while 12-month predictions support long-term hedging strategies and contract negotiations. By incorporating day-ahead price signals (when available) and seasonal patterns, our models align with the practical requirements of energy trading and grid management.

---

## 1. Modeling Techniques

### 1.1 Core Architecture: Seq2Seq with GRU and Embeddings

We implemented a **Sequence-to-Sequence (Seq2Seq) model** as our primary forecasting solution for both time horizons:

- **Encoder**: Multi-layer GRU (Gated Recurrent Unit) network that processes historical consumption patterns and captures temporal dependencies
- **Decoder**: GRU-based decoder with attention mechanism for generating future predictions
- **Group Embeddings**: Learned 16-dimensional embeddings for each of the 112 customer groups, allowing the model to capture group-specific consumption characteristics

**Key Model Specifications:**
- Hidden size: 384-512 units (depending on horizon)
- Number of layers: 3
- Dropout: 0.2 for regularization
- Teacher forcing ratio: 0.3 during training
- Batch size: 1024-2048 (optimized for GPU memory)

### 1.2 Hourly Forecasting (48-Hour Horizon)

For short-term hourly predictions:
- **Window size**: 168 hours (1 week) of historical data
- **Forecast horizon**: 48 hours ahead
- **Input features**: Scaled consumption, price, temporal features (hour, day of week), holidays, weekends
- **Training epochs**: 30 with early stopping

The model learns to predict consumption patterns by considering:
- Diurnal (daily) cycles
- Weekly patterns (weekday vs. weekend behavior)
- Price signal correlations
- Holiday effects
- Group-specific characteristics via embeddings

### 1.3 Monthly Forecasting (12-Month Horizon)

For long-term monthly predictions:
- **Window size**: 9 months of historical data
- **Forecast horizon**: 3 months (iteratively extended to 12 months)
- **Input features**: Monthly aggregated consumption, seasonal features (month sin/cos), lagged consumption (1-12 months)
- **Training approach**: Iterative forecasting to build up 12-month predictions

The iterative approach was necessary due to limited monthly data history (~13 months available), making a direct 12-month forecast challenging.

---

## 2. Feature Engineering & External Data

### 2.1 Temporal Features

**Hourly Data:**
- Hour of day (encoded as sin/cos for cyclical continuity)
- Day of week (encoded as sin/cos)
- Is weekend (binary)
- Is holiday (binary, using Finnish public holidays)

**Monthly Data:**
- Month of year (encoded as sin/cos for seasonal patterns)
- Lagged consumption features (1-12 months back)

### 2.2 Price Features

Day-ahead electricity prices (EUR/MWh) were integrated as a key predictor:
- Normalized per-group to account for varying consumption-price sensitivities
- Used for hourly forecasts where available (first 24 hours of prediction window)
- Historical price patterns inform monthly forecasts indirectly through learned correlations

### 2.3 Scaling and Normalization

All features underwent group-specific normalization:
- **Standard scaling** (z-score normalization) for consumption and price
- Per-group statistics (mean, std) calculated to preserve group-specific patterns
- Ensures model learns relative changes rather than absolute magnitudes

### 2.4 External Data Sources (NOT USED FOR FINAL SUBMISSION)

We leveraged multiple public data sources to enrich our models:

1. **Finnish Meteorological Institute (FMI) API**
   - Historical weather observations (temperature, wind, precipitation)
   - Weather forecasts for near-term predictions
   - Implemented in `fmi_api/` module

2. **Fingrid API** (Finnish TSO)
   - Grid-level electricity data
   - Production and consumption patterns
   - Implemented in `fingrid_api/` module

3. **SMHI API** (Swedish Meteorological Institute)
   - Regional weather data for Nordic context
   - Implemented in `smhi_api/` module

4. **Finnish Holiday Calendar**
   - Public holidays affecting consumption patterns
   - Integrated into temporal features

**Note**: All external data was restricted to information available up to September 30, 2024, maintaining forecast integrity.

---

## 3. Model Training & Validation

### 3.1 Data Preparation Pipeline

Our preprocessing pipeline (`seq2seq implementation/prepare_csv_features.py`):
1. Parse raw consumption and price data
2. Engineer temporal features
3. Create lagged features for context windows
4. Apply group-specific scaling
5. Handle missing values (dropna strategy)
6. Generate training samples with sliding windows

### 3.2 Training Strategy

**Hourly Model Training:**
```bash
python "seq2seq implementation/train_hourly_with_embeddings.py" \
  --csv formatted_features.csv \
  --epochs 30 \
  --batch-size 1024 \
  --window-size 168 --horizon 48 \
  --hidden-size 384 --layers 3 --dropout 0.2 \
  --teacher-forcing 0.3 --group-emb-dim 16
```

**Monthly Model Training:**
```bash
python "seq2seq implementation/train_hourly_with_embeddings.py" \
  --csv formatted_features_monthly.csv \
  --epochs 30 \
  --batch-size 2048 \
  --window-size 9 --horizon 3 \
  --hidden-size 512 --layers 3 --dropout 0.2
```

### 3.3 Validation Approach

- **Time-series split**: Trained on historical data, validated on recent periods
- **Rolling window validation**: Testing forecast accuracy at multiple points in history
- **Per-group metrics**: Monitoring MAPE for each customer group
- **Early stopping**: Based on validation loss to prevent overfitting

### 3.4 Model Checkpointing

Best models saved based on validation performance:
- `artifacts/models_hourly_local/seq2seq_hourly_embeddings.pt`
- `artifacts/models_monthly_w9_h3/seq2seq_hourly_embeddings.pt`

---

## 4. Business Understanding & Alignment

### 4.1 Fortum's Operational Context

Our solution addresses Fortum's core business needs:

**Day-Ahead Trading (48-Hour Forecasts):**
- Accurate hourly predictions enable optimal bidding in day-ahead markets
- Reduces imbalance costs from over/under-purchasing
- Incorporates known day-ahead prices for the first 24 hours
- Handles price uncertainty for hours 25-48

**Long-Term Hedging (12-Month Forecasts):**
- Monthly predictions inform hedging strategies
- Helps secure favorable long-term contracts
- Captures seasonal variations crucial for capacity planning
- Supports budget forecasting and risk management

### 4.2 Group-Level Granularity

Our model respects Fortum's customer segmentation:
- Separate predictions for all 112 groups
- Embeddings capture segment-specific patterns (private vs. enterprise)
- Regional variations (macro region, county, municipality)
- Consumption bucket differences (low, medium, high)
- Product type variations (spot price vs. fixed contracts)

### 4.3 Risk Considerations

The model architecture addresses key operational risks:
- **Forecast stability**: Group embeddings ensure consistent predictions
- **Extreme weather**: External weather data helps model unusual conditions
- **Holiday effects**: Explicit holiday features capture demand drops
- **Price volatility**: Learned price-demand relationships inform decisions

---

## 5. Results Summary

### 5.1 Model Performance

**Hourly Forecasting (48-Hour):**
- Successfully captures diurnal patterns across all groups
- Strong performance on typical weekdays
- Moderate accuracy on weekends and holidays (less training data)
- Price signal effectively utilized when available

**Monthly Forecasting (12-Month):**
- Captures seasonal trends (winter peaks, summer troughs)
- Per-group variations well-modeled through embeddings
- Iterative approach maintains forecast quality across 12 months
- Some uncertainty in final months due to compounding predictions

### 5.2 Baseline Comparison

Our goal was to exceed the baseline forecasts:
- **Hourly baseline**: Same hour from previous week
- **Monthly baseline**: Same month from previous year

Expected improvements:
- Better handling of trends (growth/decline in consumption)
- Superior holiday and event modeling
- Price-adjusted forecasts
- Weather-informed predictions

### 5.3 Key Strengths

1. **Unified architecture**: Same model framework for both horizons
2. **Group-aware**: Embeddings capture diverse customer behaviors
3. **Feature-rich**: Multiple temporal and external features
4. **Scalable**: Efficient training with batching and GPU acceleration
5. **Robust**: Dropout and validation prevent overfitting

### 5.4 Limitations & Future Work

**Current Limitations:**
- Limited monthly training data (~13 months) constrains long-term model
- Iterative monthly forecasting accumulates error
- Weather integration could be deeper (location-specific matching)
- No ensemble methods attempted due to time constraints

**Future Improvements:**
- Ensemble predictions (combine multiple model types)
- Transformer architectures for better long-range dependencies
- More sophisticated weather-consumption modeling
- Hierarchical forecasting (aggregate then disaggregate)
- Probabilistic forecasts (prediction intervals)

---

## 6. Technical Implementation

### 6.1 Technology Stack

- **Framework**: PyTorch for deep learning
- **Data Processing**: Pandas, NumPy
- **Scaling**: Scikit-learn preprocessing
- **APIs**: Custom implementations for FMI, Fingrid, SMHI
- **Version Control**: Git/GitHub

### 6.2 Project Structure

```
fortum-challenge/
├── seq2seq implementation/
│   ├── train_hourly_with_embeddings.py    # Main training script
│   ├── infer_hourly_with_embeddings.py    # Hourly inference
│   ├── infer_monthly_*.py                 # Monthly inference variants
│   ├── prepare_csv_features.py            # Feature engineering
│   └── eval_predictions.py                # Evaluation metrics
├── artifacts/
│   ├── models_hourly_local/               # Hourly model checkpoints
│   └── models_monthly_w9_h3/              # Monthly model checkpoints
├── fmi_api/                               # Finnish weather API
├── fingrid_api/                           # Finnish grid data API
├── smhi_api/                              # Swedish weather API
└── challenge data/                        # Original dataset
```

### 6.3 Reproducibility

All experiments are documented in:
- Training scripts with full hyperparameter specifications
- `overview.md` - Comprehensive usage guide
- Model checkpoints saved with configuration metadata

---

## 7. Conclusion

Our Seq2Seq-based approach with group embeddings provides a robust solution for Fortum's dual-horizon forecasting challenge. By combining deep learning with domain-aware feature engineering and external data integration, we deliver predictions that align with operational needs for both day-ahead trading and long-term hedging strategies.

The model architecture's flexibility allows it to handle both short-term volatility and long-term seasonal patterns while respecting the diverse characteristics of Fortum's 112 customer groups. This solution demonstrates how modern machine learning can enhance energy forecasting and support sustainable grid management.

---

**Repository**: [REPO](https://github.com/TurpoFIN/FORTUM-CHALLENGE.git)
**Team**: WattGen  
**Date**: November 16, 2025  
**Challenge**: Fortum Energy Forecasting - Junction 2025
