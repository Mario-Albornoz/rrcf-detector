# RRCF Anomaly Detection Service

Real-time anomaly detection service using Robust Random Cut Forest (RRCF) algorithm for streaming market data.

## Overview

This service consumes normalized feature vectors from Kafka, processes them through RRCF-based anomaly detection, and publishes anomaly scores back to Kafka. It uses multiprocessing for parallel processing across multiple workers.

## Prerequisites

- **Python**: 3.12+ (tested on 3.12.4)
- **Kafka**: Running on `localhost:9092` (or configured address)

### Required Kafka Topics

Create these topics before running the service:

```bash
# Input topic - receives normalized feature vectors
kafka-topics.sh --create --topic normalized-vectors \
    --bootstrap-server localhost:9092 \
    --partitions 4 --replication-factor 1

# Output topic - publishes anomaly scores
kafka-topics.sh --create --topic anomaly-scores \
    --bootstrap-server localhost:9092 \
    --partitions 4 --replication-factor 1
```

## Installation

```bash
# Clone and navigate to project
cd rrcf-detector

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

The service uses `config/default.yaml` by default:

```yaml
service:
  num_workers: 4           # Number of worker processes
  queue_max_size: 10000    # Max routing queue size

detector:
  window_size: 1000        # RRCF sliding window size
  min_fill_threshold: 50   # Minimum samples before scoring

kafka:
  bootstrap_servers: "localhost:9092"
  input_topic: "normalized-vectors"
  output_topic: "anomaly-scores"
  consumer_group_id: "rrcf-detector-consumer"
```

### Custom Configuration

Create a custom config for experiments:

```bash
cp config/default.yaml config/experiment_001.yaml
# Edit experiment_001.yaml with your settings
python main.py --config config/experiment_001.yaml
```

## Running the Service

### Start with Default Config

```bash
python main.py
```

### Start with Custom Config

```bash
python main.py --config config/experiment_001.yaml
```

### Expected Output

```
============================================================
RRCF Anomaly Detection Service
============================================================
[Service] Subscribed to topic: normalized-vectors
[Service] Started 4 workers
[Service] Starting message consumption loop...
[Service] Press Ctrl+C to stop
============================================================
[Service] Processed 1000 messages
[Service] Processed 2000 messages
...
```

### Verify It's Running

Monitor the output topic:

```bash
kafka-console-consumer.sh \
    --bootstrap-server localhost:9092 \
    --topic anomaly-scores \
    --from-beginning
```

### Stop the Service

Press `Ctrl+C` for graceful shutdown:

```
^C
[Service] Received signal 2
[Service] Shutting down...
[Service] Stopping workers...
[Service] Closing consumer...
[Service] Shutdown complete
```

## Running Tests

```bash
# Activate virtual environment
source venv/bin/activate

# Run all tests
pytest tests/ -v

# Run specific test modules
pytest tests/test_detect5_baselines.py -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=html
```

### Baseline Model Tests

The project includes baseline models for comparison:

```bash
# Run baseline tests (Z-Score, Isolation Forest, Half-Space Trees, RRCF)
pytest tests/test_detect5_baselines.py -v

# Run specific model tests
pytest tests/test_detect5_baselines.py::TestZScoreDetector -v
pytest tests/test_detect5_baselines.py::TestIsolationForestDetector -v
pytest tests/test_detect5_baselines.py::TestHalfSpaceTreesDetector -v
pytest tests/test_detect5_baselines.py::TestRRCFDetector -v
```

## Output Format

Anomaly scores published to Kafka have the following format:

```json
{
  "exchange": "binance",
  "instrument": "BTC-USDT",
  "instrument_class": "crypto_spot",
  "timestamp": "2024-01-15T10:30:00",
  "timestamp_ms": 1705318200000,
  "model": "rrcf",
  "raw_score": 12.34,
  "z_score": 2.5,
  "alert_level": "normal",
  "stats_mean": 5.0,
  "stats_std": 3.2,
  "stats_count": 1500,
  "worker_id": 0
}
```

### Alert Levels

- `normal`: |z| < 2.0 (within 2 standard deviations)
- `medium`: 2.0 ≤ |z| < 3.0 (between 2-3 sigma)
- `high`: |z| ≥ 3.0 (beyond 3 sigma, 99.7% outlier)

## Performance Tuning

### High Throughput Configuration

For processing 100K+ messages/second:

```yaml
service:
  num_workers: 8  # Scale with CPU cores

detector:
  window_size: 500  # Smaller window = less CPU per message

kafka:
  compression_type: "lz4"  # Fast compression
```

### Better Detection Configuration

For improved anomaly detection quality:

```yaml
detector:
  window_size: 2000  # Larger window captures longer patterns
  min_fill_threshold: 100  # More stable statistics
```

## Troubleshooting

### Kafka Connection Issues

**Error:** `Failed to resolve 'localhost:9092'`

**Solution:**
1. Verify Kafka is running: `netstat -an | grep 9092`
2. Check `bootstrap_servers` in config
3. Test connection: `kafka-topics.sh --list --bootstrap-server localhost:9092`

### No Messages Consumed

**Check:**
1. Topic exists: `kafka-topics.sh --list --bootstrap-server localhost:9092`
2. Messages in topic: `kafka-console-consumer.sh --bootstrap-server localhost:9092 --topic normalized-vectors --from-beginning --max-messages 1`
3. Consumer group offset: `kafka-consumer-groups.sh --bootstrap-server localhost:9092 --group rrcf-detector-consumer --describe`

### Worker Crashes

**Symptoms:** Health check reports dead workers

**Solutions:**
1. Check memory usage - workers may OOM with large window sizes
2. Check console output for errors
3. Reduce `queue_max_size` or increase `num_workers`

### Import Errors

**Error:** `ModuleNotFoundError: No module named 'river'`

**Solution:**
```bash
source venv/bin/activate
pip install -r requirements.txt
```

## Dependencies

- `confluent-kafka` - Kafka client library
- `rrcf` - Robust Random Cut Forest implementation
- `pandas`, `scikit-learn`, `matplotlib` - Data analysis and evaluation
- `river` - Online machine learning (for Half-Space Trees baseline)
- `pyyaml` - Configuration file handling
- `orjson`, `ciso8601` - Performance optimizations

## Architecture

- **Main Service** (`main.py`): Kafka consumer and message router
- **Partitioner** (`src/multiprocessing/partitioner.py`): Routes messages to workers
- **Workers** (`src/multiprocessing/worker.py`): RRCF detection in parallel processes
- **Detector** (`src/detection/detector.py`): Core RRCF anomaly detection logic
- **Baselines** (`src/baselines/`): Alternative detection models for comparison

## License

MIT
