# Complete Multi-Model Evaluation Workflow

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  Kafka Topic: "normalized-vectors"                              │
│  (Published by Go aggregator - your feed simulator)             │
└─────────────────────────────┬───────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Multi-Model Runner (scripts/run_multi_model.py)                │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ Consumer: reads "normalized-vectors"                      │  │
│  │ Fans out each vector to all 5 models                      │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ RRCF Worker  │  │ Z-Score     │  │ IsoForest    │         │
│  │ Queue→Detect │  │ Worker      │  │ Worker       │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
│  ┌──────────────┐  ┌──────────────┐                            │
│  │ HST Worker   │  │ Online-iForest│                            │
│  │              │  │ Worker        │                            │
│  └──────────────┘  └──────────────┘                            │
│                                                                   │
│  Each worker:                                                     │
│  1. Gets vector from queue                                       │
│  2. Calls detector.ingest_data(vector)                          │
│  3. Gets score: {raw_score, z_score, stats}                     │
│  4. Publishes to Kafka with "model" tag                         │
└───────────────────────────┬───────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│  Kafka Topic: "anomaly-scores"                                  │
│  Example messages:                                               │
│  {"model": "rrcf", "z_score": 2.3, ...}                         │
│  {"model": "zscore", "z_score": 1.5, ...}                       │
│  {"model": "isoforest", "z_score": 3.1, ...}                    │
│  {"model": "halfspace", "z_score": 1.8, ...}                    │
│  {"model": "onlineiforest", "z_score": 2.9, ...}                │
└───────────────────────────┬───────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│  Stream Collector (scripts/stream_collector.py)                 │
│  - Separate consumer (no interference)                           │
│  - Buffers 1000 messages                                        │
│  - Writes incrementally to parquet                              │
└───────────────────────────┬───────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│  scores.parquet                                                  │
│  Columns: exchange, instrument, timestamp, model, raw_score,    │
│           z_score, alert_level, stats_mean, stats_std, ...      │
└───────────────────────────┬───────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│  Evaluation (scripts/evaluate_model.py)                         │
│  - Reads parquet file                                           │
│  - Pivots by "model" column                                     │
│  - Generates comparison plots:                                   │
│    * Multi-model ROC curves                                     │
│    * Score distributions per model                              │
│    * Alert level comparisons                                     │
│    * Summary statistics table                                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Complete Step-by-Step Workflow

### Prerequisites

1. **Kafka is running**:
   ```bash
   # Check if Kafka is up
   kafka-topics.sh --list --bootstrap-server localhost:9092
   ```

2. **Topics exist**:
   ```bash
   # Create topics if needed
   kafka-topics.sh --create --topic normalized-vectors \
       --bootstrap-server localhost:9092 \
       --partitions 4 --replication-factor 1
   
   kafka-topics.sh --create --topic anomaly-scores \
       --bootstrap-server localhost:9092 \
       --partitions 4 --replication-factor 1
   ```

3. **Dependencies installed**:
   ```bash
   pip install -r requirements.txt
   ```

---

### Simulation Run

Open **3 terminals**:

#### Terminal 1: Stream Collector
```bash
cd /Users/marioandresalbornoz/Desktop/Projects/inDevelopment/Thesis/rrcf-detector

python scripts/stream_collector.py \
    --output-file ./data/run_001/scores.parquet \
    --buffer-size 1000 \
    --flush-interval 5 \
    --bootstrap-servers localhost:9092 \
    --topic anomaly-scores
```

**What this does**:
- Listens to `anomaly-scores` topic
- Buffers messages in memory (1000 at a time)
- Writes to parquet every 5 seconds or when buffer full
- Shows progress: "Flushed 1000 messages | Total: 5,000 | Rate: 250 msg/s | File: 2.5 MB"

#### Terminal 2: Multi-Model Detection Service
```bash
cd /Users/marioandresalbornoz/Desktop/Projects/inDevelopment/Thesis/rrcf-detector

python scripts/run_multi_model.py --config config/baselines.yaml
```

**What this does**:
- Consumes from `normalized-vectors` topic
- Spawns 5 worker processes (one per model)
- Fans out each vector to all 5 models
- Each model scores independently
- All publish to `anomaly-scores` with model tag
- Shows progress: "Processed 5,000 messages | Rate: 300 msg/s"

#### Terminal 3: Feed Simulator
```bash
cd /Users/marioandresalbornoz/Desktop/Projects/inDevelopment/Thesis/rrcf-detector

# Run your simulator (produces to "normalized-vectors" topic)
python simulator.py --duration 600  # 10 minutes
```

**What this does**:
- Publishes normalized feature vectors to `normalized-vectors` topic
- Your Go aggregator or Python simulator

---

### During the Run

**Watch the logs**:
- Terminal 1: See scores being written to parquet in real-time
- Terminal 2: See vectors being processed by all models
- Terminal 3: See simulator producing data

**Check Kafka (optional)**:
```bash
# See messages flowing through
kafka-console-consumer.sh \
    --bootstrap-server localhost:9092 \
    --topic anomaly-scores \
    --max-messages 5
```

You should see 5 messages per input vector (one from each model).

---

### After Simulation (Stop Services)

**Press Ctrl+C in each terminal**:
1. Terminal 3: Stop simulator
2. Terminal 2: Stop multi-model service (waits for workers to finish)
3. Terminal 1: Stop stream collector (flushes remaining buffer)

**Verify data was collected**:
```bash
ls -lh ./data/run_001/scores.parquet
# Expected: File exists, size ~100-500 MB depending on run length

# Quick inspection
python -c "
import pandas as pd
df = pd.read_parquet('./data/run_001/scores.parquet')
print(f'Total records: {len(df):,}')
print(f'Models: {sorted(df[\"model\"].unique())}')
print(f'Records per model:')
print(df[\"model\"].value_counts().sort_index())
"
```

**Expected output**:
```
Total records: 15,000
Models: ['halfspace', 'isoforest', 'onlineiforest', 'rrcf', 'zscore']
Records per model:
halfspace       3000
isoforest       3000
onlineiforest   3000
rrcf            3000
zscore          3000
```

---

### Evaluation

**Run evaluation script**:
```bash
python scripts/evaluate_model.py \
    --scores-file ./data/run_001/scores.parquet \
    --output-dir ./results/run_001

# With ground truth (if you have labeled anomalies):
python scripts/evaluate_model.py \
    --scores-file ./data/run_001/scores.parquet \
    --ground-truth ./ground_truth/anomalies.json \
    --output-dir ./results/run_001
```

**What this generates**:
```bash
cd ./results/run_001
ls -lh

# Files created:
# - summary_table.png              # Statistics for all models
# - raw_score_distributions.png    # Raw score histograms per model
# - z_score_distributions.png      # Z-score histograms per model
# - alert_levels_comparison.png    # Alert level bars per model
# - roc_curves_comparison.png      # ROC curves (if ground truth)
# - metrics_per_model.json         # Precision/Recall/F1 (if ground truth)
```

**View results**:
```bash
# macOS
open ./results/run_001/*.png

# Linux
xdg-open ./results/run_001/*.png

# Or just open in file browser
```

---

## How Each Component Knows What to Do

### 1. Multi-Model Runner (`run_multi_model.py`)

**Knows models from**:
```python
# Hard-coded list in _init_models()
model_configs = [
    ("rrcf", RRCFDetectorAdapter, {...}),
    ("zscore", ZScoreDetector, {...}),
    ("isoforest", IsolationForestDetector, {...}),
    ("halfspace", HalfSpaceTreesDetector, {...}),
    ("onlineiforest", OnlineIForestDetector, {...}),
]
```

**Reads config from**: `config/baselines.yaml`
- Kafka servers
- Topic names
- Consumer group ID
- Model parameters

**For each model**:
1. Creates detector instance: `detector = ZScoreDetector(config)`
2. Creates input queue: `queue = mp.Queue(maxsize=1000)`
3. Spawns worker: `GenericWorker.start_worker(detector, queue, kafka_config)`

**Consumer loop**:
```python
while running:
    vector = consumer.poll()
    # Fan out to ALL models
    for model_name, model_info in models.items():
        model_info["queue"].put(vector)
```

### 2. Generic Worker (`src/detection/generic_worker.py`)

**Each worker**:
1. Gets detector instance in constructor
2. Runs in separate process
3. Pulls from its dedicated queue
4. Calls `detector.ingest_data(vector)`
5. Gets back: `{"raw_score": X, "z_score": Y, "stats": {...}}`
6. Builds output message with `"model": detector.get_model_name()`
7. Publishes to Kafka `anomaly-scores` topic

### 3. Stream Collector (`stream_collector.py`)

**Listens to**: `anomaly-scores` topic (from `--topic` arg)
**Consumer group**: `stream-collector-<timestamp>` (unique, no conflicts)
**Output**: `scores.parquet` (from `--output-file` arg)

**Buffer/Flush**:
- Collects messages in RAM (default 1000)
- Flushes to parquet when buffer full OR every 5 seconds
- Uses pyarrow for incremental writes

### 4. Evaluation (`evaluate_model.py`)

**Reads**: `scores.parquet` (from `--scores-file` arg)
**Pivots by**: `"model"` column in parquet
**Generates**: One plot per comparison type

```python
# Example code inside evaluate_model.py
models = df["model"].unique()  # ["rrcf", "zscore", ...]

for model_name in models:
    model_data = df[df["model"] == model_name]
    # Plot ROC curve for this model
    fpr, tpr, _ = roc_curve(y_true, model_data["z_score"])
    plt.plot(fpr, tpr, label=model_name)
```

---

## Configuration Summary

### `config/baselines.yaml` - Used by `run_multi_model.py`

```yaml
# Kafka connection
kafka:
  bootstrap_servers: "localhost:9092"
  consumer_group_id: "rrcf-detector-baselines"
  input_topic: "normalized-vectors"      # ← Reads from here
  output_topic: "anomaly-scores"         # ← Writes to here

# Model parameters
detector:
  window_size: 1000
  min_fill_threshold: 50
  # ... other params
```

### Stream Collector - Command line args
```bash
--output-file ./data/run_001/scores.parquet   # Where to write
--bootstrap-servers localhost:9092             # Kafka connection
--topic anomaly-scores                         # Topic to consume
```

### Evaluation - Command line args
```bash
--scores-file ./data/run_001/scores.parquet   # Where to read
--output-dir ./results/run_001                 # Where to save plots
```

---

## Troubleshooting

### "No messages consumed"
```bash
# Check if simulator is running and producing
kafka-console-consumer.sh --bootstrap-server localhost:9092 \
    --topic normalized-vectors --max-messages 1

# Check if multi-model service is running
ps aux | grep run_multi_model
```

### "Empty parquet file"
```bash
# Check if multi-model service is actually producing scores
kafka-console-consumer.sh --bootstrap-server localhost:9092 \
    --topic anomaly-scores --max-messages 5

# If you see messages in Kafka but not in parquet:
# - Check stream collector logs for errors
# - Make sure stream collector is running
```

### "Models missing in evaluation"
```bash
# Check which models are in the parquet
python -c "
import pandas as pd
df = pd.read_parquet('./data/run_001/scores.parquet')
print(df['model'].value_counts())
"

# If a model is missing:
# - Check multi-model service logs for that worker
# - Check if that worker crashed (look for PID in logs)
```

---

## Summary

**The complete chain**:
1. Simulator → `normalized-vectors` Kafka topic
2. `run_multi_model.py` consumes → fans out to 5 workers → all publish to `anomaly-scores`
3. `stream_collector.py` consumes `anomaly-scores` → writes `scores.parquet`
4. `evaluate_model.py` reads `scores.parquet` → generates comparison plots

**Key insight**: Each vector gets scored by all 5 models, and each score is tagged with the model name. The parquet file contains all scores from all models, and evaluation pivots by the "model" column.

**You're now ready to run experiments!** 🎉
