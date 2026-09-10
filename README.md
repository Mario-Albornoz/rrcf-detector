# Quick Start - Baseline Models

## Installation

```bash
# Install River library for Half-Space Trees
pip install -r requirements.txt
```

**Note**: River requires numpy<2.0.0 and will downgrade from 2.5.2 to 1.26.4 automatically.

## Running Tests

```bash
# Run all baseline tests (21 tests)
pytest tests/test_detect5_baselines.py -v

# Run specific model tests
pytest tests/test_detect5_baselines.py::TestZScoreDetector -v
pytest tests/test_detect5_baselines.py::TestIsolationForestDetector -v
pytest tests/test_detect5_baselines.py::TestHalfSpaceTreesDetector -v
```

## Usage Examples

### Example 1: Use Z-Score Detector

```python
from src.baselines import ZScoreDetector
from src.kafka.consumer import NormalizedVectorDto
from datetime import datetime

# Create detector
detector = ZScoreDetector(config={"training_samples": 20000})

# Ingest vectors
for vector in stream:
    result = detector.ingest_data(vector)
    
    if result:  # None during training phase
        print(f"Raw score: {result['raw_score']}")
        print(f"Z-score: {result['z_score']}")
        print(f"Alert: {detector.determine_alert_level(result['z_score'])}")
```

### Example 2: Use Isolation Forest Detector

```python
from src.baselines import IsolationForestDetector

detector = IsolationForestDetector(config={
    "training_samples": 20000,
    "n_estimators": 100,
    "contamination": 0.1
})

# Same interface as Z-Score
result = detector.ingest_data(vector)
```

### Example 3: Use Half-Space Trees (Online)

```python
from src.baselines import HalfSpaceTreesDetector

detector = HalfSpaceTreesDetector(config={
    "window_size": 1000,
    "n_trees": 25,
    "height": 8,
    "min_fill_threshold": 50
})

# No training phase - learns from stream immediately
result = detector.ingest_data(vector)
```

### Example 4: Use RRCF via Adapter

```python
from src.baselines import RRCFDetectorAdapter

detector = RRCFDetectorAdapter(config={
    "window_size": 1000,
    "num_trees": 40,
    "tree_size": 256,
    "min_fill_threshold": 50
})

# Same interface as all baselines
result = detector.ingest_data(vector)
```

### Example 5: Run with Generic Worker

```python
from src.baselines import ZScoreDetector
from src.detection.generic_worker import GenericWorker
import multiprocessing as mp

# Create any detector
detector = ZScoreDetector(config={"training_samples": 20000})

# Spawn worker
input_queue = mp.Queue()
kafka_config = {
    "bootstrap_servers": "localhost:9092",
    "output_topic": "anomaly-scores"
}

process = GenericWorker.start_worker(
    worker_id=0,
    detector=detector,
    input_queue=input_queue,
    kafka_config=kafka_config
)
```

## Configuration

All models use the same config structure:

```yaml
# config/baselines.yaml
detector:
  # For RRCF & HST
  window_size: 1000
  min_fill_threshold: 50
  
  # For Z-Score & IsoForest
  training_samples: 20000
  
  # For IsoForest only
  n_estimators: 100
  contamination: 0.1
  
  # For HST only
  n_trees: 25
  height: 8
```

## Output Format

All models produce identical output:

```python
{
    "raw_score": 12.34,       # Model-specific raw score
    "z_score": 2.5,           # Self-calibrated z-score
    "stats": {
        "mean": 5.0,          # Rolling mean of scores
        "std": 3.2,           # Rolling std of scores
        "count": 1500         # Number of scores seen
    }
}
```

When published to Kafka, additional fields are added:

```json
{
  "exchange": "binance",
  "instrument": "BTC-USDT",
  "instrument_class": "crypto_spot",
  "timestamp": "2024-01-15T10:30:00",
  "timestamp_ms": 1705318200000,
  "model": "rrcf|zscore|isoforest|halfspace",
  "raw_score": 12.34,
  "z_score": 2.5,
  "alert_level": "normal|medium|high",
  "stats_mean": 5.0,
  "stats_std": 3.2,
  "stats_count": 1500,
  "worker_id": 0
}
```

## Alert Levels

All models use the same z-score based alert thresholds:

- `|z| < 2.0` → **normal** (within 2 standard deviations)
- `2.0 ≤ |z| < 3.0` → **medium** (between 2-3 sigma)
- `|z| ≥ 3.0` → **high** (beyond 3 sigma, 99.7% outlier)

## Next Steps

1. **For Evaluation**: See `DETECT-5-TRAINING-ASSUMPTIONS.md` for training data requirements
2. **For Deployment**: See `DETECT-5-COMPLETE.md` for multi-model architecture
3. **For Testing**: Run `pytest tests/test_detect5_baselines.py -v`

## Troubleshooting

### River import error
```bash
pip install river==0.21.2
```

### Numpy version conflict
River requires numpy<2.0.0. This is handled automatically by pip.

### Tests fail
Make sure you're in the venv:
```bash
source venv/bin/activate
pytest tests/test_detect5_baselines.py -v
```
