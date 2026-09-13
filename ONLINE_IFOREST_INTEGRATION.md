# Online-iForest Integration & Streaming Evaluation - Implementation Summary

**Date**: September 13, 2026
**Status**: ✅ Complete

## Overview

Successfully integrated Online Isolation Forest (ICML 2024) as a 5th baseline model and implemented real-time streaming parquet collection for efficient multi-model evaluation.

---

## What Was Implemented

### 1. **Streaming Parquet Collector** ✅
**File**: `scripts/stream_collector.py`

A separate Kafka consumer that runs in parallel with the detection service and writes scores to parquet in real-time.

**Key Features**:
- Separate consumer group (no interference with detection service)
- Micro-batch buffering (configurable, default 1000 messages)
- Incremental parquet writes using pyarrow
- Graceful shutdown with buffer flush
- Progress monitoring (rate, file size)
- Memory efficient (only buffers small batches)

**Usage**:
```bash
# Terminal 1: Detection service
python main.py --config config/baselines.yaml

# Terminal 2: Stream collector (parallel)
python scripts/stream_collector.py \
    --output-file ./data/run_001/scores.parquet \
    --buffer-size 1000 \
    --flush-interval 5

# Terminal 3: Feed simulator
python simulator.py

# After stopping: evaluate immediately (no post-collection needed!)
python scripts/evaluate_model.py \
    --scores-file ./data/run_001/scores.parquet \
    --output-dir ./results/run_001
```

**Benefits**:
- ✅ No post-processing step needed
- ✅ Data ready for evaluation immediately
- ✅ Memory efficient
- ✅ No interference with detection service
- ✅ Fault tolerant (periodic flushes)

---

### 2. **Online-iForest Baseline Detector** ✅
**File**: `src/baselines/online_iforest_detector.py`

Integrated Online Isolation Forest from the ICML 2024 paper as the 5th baseline model.

**Model Details**:
- **Paper**: "Online Isolation Forest" (Leveni et al., ICML 2024)
- **Type**: Genuinely online streaming detector with sliding window
- **Key Feature**: Learning + forgetting procedures (adds new, removes old)
- **Direct competitor to RRCF**: Both are online tree-based methods

**Implementation**:
- Implements `BaseDetector` interface (uniform with other baselines)
- Uses Welford statistics for z-score calibration
- Handles cold start with `min_fill_threshold`
- One instance per `(exchange, instrument_class)`
- Integrates with existing worker/partitioning infrastructure

**Configuration** (`config/baselines.yaml`):
```yaml
detector:
  onlineiforest:
    window_size: 1024        # Sliding window size
    num_trees: 32            # Number of trees in forest
    max_leaf_samples: 32     # Max samples per leaf before split
    type: "adaptive"         # "adaptive" or "fixed"
    min_fill_threshold: 50   # Warmup samples before scoring
```

**Added to models list**:
```yaml
models:
  - rrcf
  - zscore
  - isoforest
  - halfspace
  - onlineiforest  # NEW!
```

---

### 3. **Multi-Model Evaluation Script** ✅
**File**: `scripts/evaluate_model.py` (completely rewritten)

Enhanced evaluation script that compares all models side-by-side from streaming parquet file.

**Key Features**:
- Reads from single parquet file (from stream_collector)
- Pivots by "model" column for comparison
- Generates multi-model comparison plots
- Per-model metrics when ground truth available
- Statistical summaries

**Generated Outputs**:
1. **Multi-model ROC curves** - All models on same plot with AUC scores
2. **Score distributions** - Raw scores and z-scores per model (side-by-side)
3. **Alert level comparison** - Stacked bar chart showing alert distributions
4. **Summary table** - Statistical summary for all models
5. **Per-model metrics** - Precision/Recall/F1 for each model (if ground truth)

**Usage**:
```bash
# Basic comparison (no ground truth)
python scripts/evaluate_model.py \
    --scores-file ./data/run_001/scores.parquet \
    --output-dir ./results/run_001

# With ground truth for metrics
python scripts/evaluate_model.py \
    --scores-file ./data/run_001/scores.parquet \
    --ground-truth ./ground_truth/anomalies.json \
    --output-dir ./results/run_001
```

---

### 4. **Dependencies & Configuration** ✅

**Added to `requirements.txt`**:
```
pyarrow==18.0.0
```

**Updated files**:
- `src/baselines/__init__.py` - Export `OnlineIForestDetector`
- `config/baselines.yaml` - Added online-iforest config and model entry

---

## Architecture Changes

### Before:
```
Detection Service → Kafka "anomaly-scores"
                           ↓
                    (after simulation)
              Manual: collect_data.py
                           ↓
              input_vectors.parquet
              output_scores.parquet
                           ↓
              Merge & evaluate
```

### After:
```
Detection Service → Kafka "anomaly-scores"
                           ↓
                    Stream Collector ← (parallel, no interference)
                           ↓
                    scores.parquet (grows in real-time)
                           ↓
                    Evaluate immediately (no post-processing!)
```

---

## Model Comparison Matrix

| Model | Type | Training | Sliding Window | Paper |
|-------|------|----------|----------------|-------|
| **RRCF** | Online | None | Yes | Guha et al. (ICML 2016) |
| **Z-Score** | Batch | 2 weeks frozen | No | Rule-based (status quo) |
| **IsoForest** | Batch | 2 weeks frozen | No | Liu et al. (ICDM 2008) |
| **Half-Space Trees** | Online | None | Yes | Tan et al. (IJCAI 2011) |
| **Online-iForest** | Online | None | Yes | **Leveni et al. (ICML 2024)** ✨ |

### Key Comparisons for Thesis:

1. **Online vs Batch** (concept drift):
   - RRCF, HST, Online-iForest (adapt) vs Z-Score, IsoForest (frozen)

2. **Online vs Online** (algorithm effectiveness):
   - **RRCF** vs **Online-iForest** vs HST
   - All three are streaming, question is: which best for financial feeds?

3. **Modern vs Classic**:
   - Online-iForest (2024) vs RRCF (2016) vs HST (2011)

---

## Testing the Implementation

### 1. Quick Syntax Check:
```bash
python -m py_compile scripts/stream_collector.py
python -m py_compile scripts/evaluate_model.py
python -c "from src.baselines import OnlineIForestDetector; print('✓ Import successful')"
```

### 2. Test Stream Collector (without running full service):
```bash
# Assuming Kafka is running and anomaly-scores topic exists
python scripts/stream_collector.py \
    --output-file ./test_scores.parquet \
    --buffer-size 100 \
    --flush-interval 2
```

### 3. Test Online-iForest Detector:
```python
from src.baselines import OnlineIForestDetector
from src.kafka.consumer import NormalizedVectorDto
from datetime import datetime

config = {
    "window_size": 1024,
    "num_trees": 32,
    "max_leaf_samples": 32,
    "type": "adaptive",
    "min_fill_threshold": 50
}

detector = OnlineIForestDetector(config)
print(f"✓ Detector initialized: {detector.get_model_name()}")

# Test with dummy data
dummy_vector = NormalizedVectorDto(
    exchange="binance",
    instrument="BTC-USDT",
    instrument_class="crypto_spot",
    timestamp=datetime.now(),
    z_intertick_fast=0.5,
    z_price_step_fast=-0.3,
    z_intertick_slow=0.2,
    z_price_step_slow=0.1,
    cusum_intertick=0.0,
    cusum_price_step=0.0
)

# First few will return None (cold start)
result = detector.ingest_data(dummy_vector)
print(f"Cold start result: {result}")

# After min_fill_threshold samples, will return scores
for _ in range(60):
    result = detector.ingest_data(dummy_vector)

print(f"✓ Warm result: {result}")
```

### 4. Full Integration Test:
```bash
# 1. Start Kafka
docker-compose up -d kafka

# 2. Start stream collector
python scripts/stream_collector.py \
    --output-file ./data/test_run/scores.parquet &

# 3. Start detection service with all 5 models
python main.py --config config/baselines.yaml &

# 4. Run feed simulator
python simulator.py --duration 60

# 5. Stop services (Ctrl+C)

# 6. Evaluate immediately
python scripts/evaluate_model.py \
    --scores-file ./data/test_run/scores.parquet \
    --output-dir ./results/test_run

# 7. Check results
ls -lh ./results/test_run/
# Should see: roc_curves_comparison.png, score_distributions.png, etc.
```

---

## Files Changed/Created

### New Files:
- ✅ `scripts/stream_collector.py` - Real-time parquet collector
- ✅ `src/baselines/online_iforest_detector.py` - Online-iForest wrapper
- ✅ `ONLINE_IFOREST_INTEGRATION.md` - This document

### Modified Files:
- ✅ `scripts/evaluate_model.py` - Complete rewrite for multi-model comparison
- ✅ `src/baselines/__init__.py` - Export OnlineIForestDetector
- ✅ `config/baselines.yaml` - Added online-iforest config
- ✅ `requirements.txt` - Added pyarrow

### Unchanged (but relevant):
- ✅ `src/baselines/Online-Isolation-Forest/` - Cloned repo (already exists)
- ✅ `src/baselines/base_detector.py` - Interface (no changes needed)
- ✅ `src/detection/generic_worker.py` - Works with any BaseDetector

---

## Next Steps for Thesis

1. **Test with real data**:
   - Run simulation with injected anomalies
   - Compare all 5 models
   - Generate ROC curves and metrics

2. **Document training assumptions**:
   - Online-iForest: no training, learns from stream
   - Update DETECT-5 section in skill file

3. **Parameter tuning**:
   - Experiment with window_size (512, 1024, 2048)
   - Experiment with num_trees (16, 32, 64)
   - Compare adaptive vs fixed mode

4. **Evaluation metrics**:
   - Detection latency per model
   - Memory consumption per model
   - Throughput (messages/sec) per model

5. **Thesis comparison**:
   - RRCF vs Online-iForest (online vs online)
   - Both vs frozen models (concept drift evidence)
   - Performance vs accuracy tradeoffs

---

## Key Advantages of This Approach

### For Development:
- ✅ Fast iteration: no waiting for post-collection
- ✅ Memory efficient: stream to disk, not RAM
- ✅ Parallelism: collect while detecting
- ✅ Fault tolerant: periodic flushes

### For Evaluation:
- ✅ Direct model comparison: all in one file
- ✅ Beautiful plots: side-by-side visualizations
- ✅ Flexible: works with/without ground truth
- ✅ Reproducible: parquet files are portable

### For Thesis:
- ✅ 5 models spanning 18 years (2006-2024)
- ✅ Clear online vs batch comparison
- ✅ State-of-the-art baseline (ICML 2024)
- ✅ Production-ready architecture

---

## References

1. **Online Isolation Forest**: Leveni, F., Cassales, G. W., Pfahringer, B., Bifet, A., & Boracchi, G. (2024). Online Isolation Forest. In Proceedings of the 41st International Conference on Machine Learning (ICML), 235, 27288-27298.
   - Paper: https://proceedings.mlr.press/v235/leveni24a.html
   - Code: https://github.com/ineveLoppiliF/Online-Isolation-Forest

2. **RRCF**: Guha, S., Mishra, N., Roy, G., & Schrijvers, O. (2016). Robust random cut forest based anomaly detection on streams. In International conference on machine learning (pp. 2712-2721). PMLR.

3. **Half-Space Trees**: Tan, S. C., Ting, K. M., & Liu, T. F. (2011). Fast anomaly detection for streaming data. In IJCAI (Vol. 22, No. 1, p. 1511).

---

## Summary

✅ **Stream collector**: Real-time parquet writing (parallel, efficient)
✅ **Online-iForest**: 5th baseline model integrated (ICML 2024)
✅ **Multi-model evaluation**: Side-by-side comparison plots
✅ **Dependencies**: pyarrow added, configs updated
✅ **Architecture**: Production-ready, thesis-ready

**All tasks complete!** 🎉

Ready for simulation runs and thesis evaluation.
