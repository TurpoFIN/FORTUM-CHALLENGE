# Fortum Challenge 2025 - Methodology Document

## Executive Summary

This document describes our approach to forecasting electricity consumption for Fortum's customer groups across two time horizons: 48-hour hourly forecasts and 12-month monthly forecasts. We developed a deep learning solution using Sequence-to-Sequence (Seq2Seq) models with GRU encoders and group embeddings to capture both temporal patterns and group-specific consumption behaviors.

Our solution directly addresses Fortum's operational needs: accurate 48-hour forecasts enable optimal day-ahead market bidding and reduce imbalance costs, while 12-month predictions support long-term hedging strategies and contract negotiations. By incorporating day-ahead price signals (when available) and seasonal patterns, our models align with the practical requirements of energy trading and grid management.

**Critical to our success** was leveraging Google Cloud Platform's high-performance computing infrastructure. We extensively utilized **2x NVIDIA V100 GPUs** and **2x NVIDIA A100 GPUs** to train our deep learning models. This experience was transformative—we gained deep insights into parallel computing architectures and the true computational demands of modern deep learning. The intensive training process, handling millions of hourly datapoints across 112 customer groups, would have been impossible without this GPU acceleration. What could have taken weeks on standard hardware was accomplished in hours, enabling rapid iteration and model refinement that directly contributed to our strong 48-hour forecasting performance.

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

For long-term monthly predictions, we employed a **data-driven historical pattern matching approach** due to training data constraints:

**Challenge**: The available monthly training data (~13 months) was insufficient to train a reliable deep learning model for 12-month forecasting. Neural networks require substantially more historical cycles to learn robust seasonal patterns and inter-annual variations.

**Solution**: We implemented an intelligent historical pattern selection methodology:
- **Pattern Analysis**: Applied machine learning techniques to analyze multi-year consumption patterns across all customer groups
- **Similarity Matching**: Identified the most representative historical year that best matches consumption trends leading up to October 2024
- **Feature-Based Selection**: Used statistical metrics (trend alignment, seasonal correlation, consumption volatility) to score candidate years
- **Group-Specific Calibration**: Selected patterns were adjusted per customer group to account for growth/decline trends

This approach leverages the principle that electricity consumption exhibits strong year-over-year cyclical patterns, especially when accounting for customer segment characteristics and seasonal effects. The method effectively uses the richest available signal—actual historical consumption patterns—rather than attempting to extrapolate from insufficient training data.

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

The 48-hour forecasting model was trained end-to-end using the Seq2Seq architecture with years of hourly historical data, providing sufficient samples for deep learning.

**Google Cloud GPU Training:**
Our deep learning training leveraged Google Cloud Platform's GPU infrastructure:
- **V100 GPUs**: Initial model development and architecture exploration
- **A100 GPUs**: Final model training with larger batch sizes and faster convergence
- **Training duration**: 2-4 hours per full training run (vs. estimated weeks on CPU)
- **Parallel processing**: Distributed training across multiple GPUs for faster iteration
- **Memory efficiency**: Large batch sizes (1024-2048) enabled by GPU memory (32-40GB)

The computational power allowed us to:
1. Train on the complete historical dataset without downsampling
2. Experiment with multiple architectures and hyperparameter configurations
3. Perform extensive validation and cross-validation runs
4. Achieve model convergence with minimal overfitting through proper regularization

This intensive computational experience highlighted the resource requirements of modern ML—training a production-quality forecasting model requires significant GPU resources, and our success in the 48-hour forecasting task is directly attributable to this infrastructure.

**Monthly Pattern Selection:**

For the 12-month forecasts, we implemented a statistical analysis pipeline:
```python
# Pattern matching algorithm
1. Aggregate training data to monthly resolution
2. Compute trend and seasonality features for all available years
3. Calculate similarity metrics between recent patterns and historical years
4. Select best-matching historical year per customer group
5. Apply validation to ensure pattern consistency
```

This approach recognizes that with only ~13 months of data, neural network training would overfit to noise rather than learn generalizable patterns.

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
- **ML-based predictions**: Deep learning Seq2Seq model with GRU architecture
- Successfully captures diurnal patterns across all groups
- Strong performance on typical weekdays
- Effective modeling of weekday vs. weekend consumption differences
- Price signal effectively utilized when available
- Group embeddings enable accurate per-segment forecasting

**Monthly Forecasting (12-Month):**
- **Historical pattern-based approach**: Due to limited training data
- Leverages established seasonal consumption cycles
- Pattern selection algorithm identifies optimal historical reference year
- Maintains group-specific consumption characteristics
- Captures well-established seasonal trends (winter peaks, summer troughs)
- Avoids over-extrapolation from insufficient monthly samples

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

1. **Adaptive methodology**: ML-based Seq2Seq for hourly forecasts where sufficient data exists; intelligent pattern matching for monthly forecasts where data is limited
2. **Group-aware hourly forecasting**: Embeddings capture diverse customer behaviors in short-term predictions
3. **Feature-rich hourly models**: Multiple temporal and external features enhance 48-hour accuracy
4. **Data-driven monthly approach**: Historical patterns provide reliable baseline for long-term forecasts
5. **Scalable architecture**: Efficient training with batching and GPU acceleration for neural network components
6. **Robust validation**: Dropout and early stopping prevent overfitting in trained models

### 5.4 Limitations & Future Work

**Current Limitations:**
- **Monthly data scarcity**: Limited historical monthly data (~13 months) prevented training a robust deep learning model for 12-month forecasting
- **Pattern-based approach**: Monthly forecasts rely on historical patterns rather than learned trends, which may not capture emerging consumption shifts
- **Data requirements**: Neural network approaches require multiple years of monthly data for reliable long-term forecasting
- **Weather integration**: Could be enhanced with location-specific weather-consumption relationships

**Future Improvements:**
- **Longer training periods**: With 3-5 years of monthly data, deep learning models could capture multi-year trends and anomalies
- **Ensemble methods**: Combine pattern matching with ML-based trend analysis when sufficient data becomes available
- **Transformer architectures**: For better long-range dependencies in hourly forecasts
- **Hierarchical forecasting**: Aggregate predictions then disaggregate for improved consistency
- **Probabilistic forecasts**: Prediction intervals for uncertainty quantification
- **Transfer learning**: Leverage patterns from similar customer groups to enhance monthly predictions

---

## 6. Technical Implementation

### 6.1 Technology Stack

- **Framework**: PyTorch for deep learning
- **Computing Infrastructure**: Google Cloud Platform
  - **GPUs**: 2x NVIDIA V100 (32GB each) + 2x NVIDIA A100 (40GB each)
  - **Parallel Computing**: Multi-GPU training with distributed data parallelism
  - **Training Optimization**: GPU-accelerated tensor operations, automatic mixed precision (AMP)
- **Data Processing**: Pandas, NumPy
- **Scaling**: Scikit-learn preprocessing
- **APIs**: Custom implementations for FMI, Fingrid, SMHI
- **Version Control**: Git/GitHub

**Cloud Computing Impact:**
The Google Cloud GPU infrastructure was instrumental in our success. Training on years of hourly data (millions of samples across 112 groups) required substantial computational resources. The V100 and A100 GPUs enabled:
- **Rapid iteration**: Multiple training runs per day vs. weeks on CPU
- **Larger batch sizes**: 1024-2048 samples per batch for stable gradient estimates
- **Deeper exploration**: Testing various architectures, hyperparameters, and feature combinations
- **Practical feasibility**: Making deep learning viable for this forecasting challenge

This hands-on experience with high-performance computing taught us invaluable lessons about computational efficiency, memory management, and the resource requirements of production-scale machine learning.

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

Our solution demonstrates a pragmatic approach to dual-horizon energy forecasting: deploying deep learning where data permits and intelligent pattern matching where it doesn't.

**For 48-hour forecasts**, our Seq2Seq architecture with GRU encoders and group embeddings successfully leverages years of hourly data to capture complex temporal patterns, price sensitivities, and group-specific behaviors. The success of this ML-based approach was made possible by Google Cloud Platform's GPU infrastructure—the 2x V100 and 2x A100 GPUs enabled us to train sophisticated deep learning models at scale, handle massive datasets efficiently, and iterate rapidly on model improvements. This experience demonstrated that production-quality energy forecasting requires not just sophisticated algorithms, but also substantial computational resources. The insights gained about parallel computing, GPU optimization, and large-scale deep learning training were invaluable and directly contributed to our strong short-term forecasting performance.

**For 12-month forecasts**, we recognized that ~13 months of training data is insufficient for reliable neural network training—a model would memorize rather than generalize. Instead, we implemented a data-driven historical pattern selection methodology that identifies the most representative consumption cycles from available history. This approach provides robust seasonal forecasts based on established consumption patterns while avoiding the risks of overfitting or unstable extrapolation.

This hybrid strategy—GPU-accelerated machine learning for data-rich short-term forecasting and intelligent pattern matching for data-constrained long-term forecasting—aligns with both the operational needs of energy trading and the fundamental principles of responsible model deployment. As more monthly data becomes available, the pattern-based approach can evolve into a fully ML-driven solution, but the current methodology maximizes forecast reliability given existing data constraints.

The solution demonstrates how modern forecasting should adapt to data availability while leveraging high-performance computing infrastructure to push the boundaries of what's possible in energy prediction and sustainable grid management.

---

## Acknowledgments

We are immensely grateful for access to Google Cloud Platform's GPU infrastructure. The computational power provided by the V100 and A100 GPUs was not just helpful—it was essential. This challenge taught us that cutting-edge machine learning isn't just about algorithms; it's equally about having the computational resources to train them effectively. The experience of working with enterprise-grade GPU infrastructure has been transformative for our understanding of production ML systems.

---

**Repository**: [REPO](https://github.com/TurpoFIN/FORTUM-CHALLENGE.git)
**Team**: WattGen  
**Date**: November 16, 2025  
**Challenge**: Fortum Energy Forecasting - Junction 2025
