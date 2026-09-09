# Running the RRCF Anomaly Detection Service

Complete guide for running, testing, and evaluating the RRCF anomaly detector.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Running the Service](#running-the-service)
5. [Data Collection](#data-collection)
6. [Evaluation](#evaluation)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Services

- **Kafka**: Running on `localhost:9092` (or configured address)
- **Python**: 3.12+ (tested on 3.12.4)

### Required Topics

The service expects these Kafka topics to exist:

- `normalized-features` (input) - Feature vectors from Go aggregator
- `anomaly-scores` (output) - Anomaly detection results

Create topics:
```bash
kafka-topics.sh --create --topic normalized-features \\
    --bootstrap-server localhost:9092 \\
    --partitions 4 --replication-factor 1

kafka-topics.sh --create --topic anomaly-scores \\
    --bootstrap-server localhost:9092 \\
    --partitions 4 --replication-factor 1
```

---

## Installation

### 1. Clone and Setup

```bash
cd rrcf-detector
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies include:**
- `confluent-kafka` - Kafka client
- `rrcf` - RRCF anomaly detection
- `pandas`, `scikit-learn`, `matplotlib` - Evaluation
- `pyyaml` - Configuration
- `orjson`, `ciso8601` - Performance optimizations

---

## Configuration

### Default Configuration

The service uses `config/default.yaml` by default:

```yaml
service:
  num_workers: 4
  queue_max_size: 10000

detector:
  window_size: 1000
  min_fill_threshold: 50

kafka:
  bootstrap_servers: "localhost:9092"
  input_topic: "normalized-features"
  output_topic: "anomaly-scores"
  consumer_group_id: "rrcf-detector-consumer"
```

### Custom Configuration

Create a custom config for experiments:

```bash
cp config/default.yaml config/experiment_001.yaml
# Edit experiment_001.yaml with your settings
```

Key parameters to tune:
- `window_size`: RRCF sliding window (larger = more memory, better long-term patterns)
- `min_fill_threshold`: Samples before scoring (cold start duration)
- `num_workers`: Worker processes (scale with CPU cores)

---

## Running the Service

### Start the Service

**With default config:**
```bash
python main.py
```

**With custom config:**
```bash
python main.py --config config/experiment_001.yaml
```

### Expected Output

```
============================================================
RRCF Anomaly Detection Service
============================================================
[Service] Subscribed to topic: normalized-features
[Partitioner] Starting 4 workers...
[Partitioner] Worker 0 started (PID: 12345)
[Partitioner] Worker 1 started (PID: 12346)
[Partitioner] Worker 2 started (PID: 12347)
[Partitioner] Worker 3 started (PID: 12348)
[Partitioner] All 4 workers started
[Service] Started 4 workers
[Service] Starting message consumption loop...
[Service] Press Ctrl+C to stop
============================================================
[Service] Processed 1000 messages
[Service] Processed 2000 messages
...
```

### Verify It's Working

**1. Check worker health:**
The service performs health checks every 60 seconds and will automatically restart dead workers.

**2. Monitor output topic:**
```bash
kafka-console-consumer.sh \\
    --bootstrap-server localhost:9092 \\
    --topic anomaly-scores \\
    --from-beginning
```

**3. Check for errors:**
Look for error messages in the console output.

### Stop the Service

Press `Ctrl+C` for graceful shutdown:
```
^C
[Service] Received signal 2
[Service] Shutting down...
[Service] Stopping workers...
[Partitioner] Shutting down 4 workers...
[Service] Closing consumer...
[Service] Shutdown complete
```

---

## Data Collection

After running a simulation, collect data for analysis:

### Collect from Kafka Topics

```bash
python scripts/collect_data.py \\
    --input-topic normalized-features \\
    --output-topic anomaly-scores \\
    --output-dir ./data/run_001 \\
    --bootstrap-servers localhost:9092
```

**Output:**
- `data/run_001/input_vectors.parquet` - All input feature vectors
- `data/run_001/output_scores.parquet` - All anomaly scores

**Options:**
- `--skip-input` - Only collect output scores
- `--skip-output` - Only collect input vectors

### Verify Collection

```bash
ls -lh data/run_001/
# Should show:
# input_vectors.parquet
# output_scores.parquet
```

---

## Evaluation

### Basic Evaluation (No Ground Truth)

Generate score distributions and alert level plots:

```bash
python scripts/evaluate_model.py \\
    --data-dir ./data/run_001 \\
    --output-dir ./results/run_001
```

**Generated files:**
- `results/run_001/score_distribution.png` - Raw scores and z-scores
- `results/run_001/alert_levels.png` - Distribution of alert levels

### Full Evaluation (With Ground Truth)

If you have injected anomalies with timestamps:

**1. Create ground truth file (`ground_truth/anomalies.json`):**
```json
[
  {
    "exchange": "binance",
    "instrument": "BTC-USDT",
    "timestamp": "2024-01-01T12:00:00",
    "type": "silence"
  },
  {
    "exchange": "coinbase",
    "instrument": "ETH-USD",
    "timestamp": "2024-01-01T13:30:00",
    "type": "lag"
  }
]
```

**2. Run evaluation:**
```bash
python scripts/evaluate_model.py \\
    --data-dir ./data/run_001 \\
    --ground-truth ./ground_truth/anomalies.json \\
    --output-dir ./results/run_001
```

**Additional generated files:**
- `results/run_001/metrics.json` - Precision, recall, F1 for multiple thresholds
- `results/run_001/roc_curve.png` - ROC curve with AUC

### Interpretation

**Score Distribution:**
- Raw scores: Raw CoDisp values from RRCF
- Z-scores: Self-calibrated (learned from data per exchange/class)

**Alert Levels:**
- `normal`: |z| < 2.0 (within 95% confidence)
- `medium`: 2.0 ≤ |z| < 3.0 (95-99.7% confidence)
- `high`: |z| ≥ 3.0 (>99.7% outlier)

**Metrics:**
```json
{
  "z_score_2": {
    "precision": 0.85,
    "recall": 0.92,
    "f1": 0.88
  },
  "z_score_3": {
    "precision": 0.95,
    "recall": 0.75,
    "f1": 0.84
  }
}
```

---

## Troubleshooting

### Kafka Connection Issues

**Error:** `Failed to resolve 'localhost:9092'`

**Solution:**
1. Verify Kafka is running: `netstat -an | grep 9092`
2. Check config: `bootstrap_servers` in `config/default.yaml`
3. Test connection:
   ```bash
   kafka-topics.sh --list --bootstrap-server localhost:9092
   ```

### No Messages Consumed

**Symptoms:** Service starts but no messages processed

**Check:**
1. **Topic exists:**
   ```bash
   kafka-topics.sh --list --bootstrap-server localhost:9092
   ```

2. **Messages in topic:**
   ```bash
   kafka-console-consumer.sh --bootstrap-server localhost:9092 \\
       --topic normalized-features --from-beginning --max-messages 1
   ```

3. **Consumer group offset:**
   ```bash
   kafka-consumer-groups.sh --bootstrap-server localhost:9092 \\
       --group rrcf-detector-consumer --describe
   ```

### Worker Crashes

**Symptoms:** Health check reports dead workers

**Check:**
1. **Memory usage:** Workers may OOM with large window sizes
2. **Logs:** Check console output for errors
3. **Queue overflow:** Reduce `queue_max_size` or add more workers

### Evaluation Errors

**Error:** `File not found: input_vectors.parquet`

**Solution:** Run `scripts/collect_data.py` first to collect data

**Error:** `KeyError: 'timestamp'`

**Solution:** Check your Kafka message format matches expected schema

---

## Performance Tuning

### For High Throughput (100K+ msg/sec)

```yaml
service:
  num_workers: 8  # Scale with CPU cores

detector:
  window_size: 500  # Smaller window = less CPU per message

kafka:
  max_poll_records: 1000  # Larger batches
  compression_type: "lz4"  # Fast compression
```

### For Better Detection

```yaml
detector:
  window_size: 2000  # Larger window captures longer patterns
  min_fill_threshold: 100  # More stable statistics
```

---

## Next Steps

1. **Baseline Models (DETECT-5):** Implement z-score, Isolation Forest, LSTM
2. **Multi-Model Comparison:** Compare RRCF vs baselines
3. **Production Deployment:** Add monitoring, logging, persistence

---

## Questions?

Check the project README or skill file:
- `.cursor/skills/rrcf-detector-tasks/SKILL.md`
