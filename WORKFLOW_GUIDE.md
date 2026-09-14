# Complete Multi-Model Evaluation Workflow

## Quick Start (Using Makefile)

```bash
# Terminal 1
make run-collector

# Terminal 2  
make run-multi

# Terminal 3 - Start your Go aggregator
cd /path/to/go-aggregator
go run main.go
```

That's it! All configuration is in `config/baselines.yaml`.

---

## Configuration-Driven Approach

**All settings are in `config/baselines.yaml`** - no command-line arguments needed!

### Stream Collector Config
```yaml
stream_collector:
  output_file: "./data/scores.parquet"
  buffer_size: 1000
  flush_interval: 5
  bootstrap_servers: "localhost:9092"
  topic: "anomaly-scores"
```

### Multi-Model Config
```yaml
kafka:
  bootstrap_servers: "localhost:9092"
  input_topic: "normalized-vectors"
  output_topic: "anomaly-scores"

models:
  - rrcf
  - zscore
  - isoforest
  - halfspace
  - onlineiforest
```

---

## Makefile Commands

```bash
# Setup
make install          # Install dependencies
make test            # Run integration tests
make check-kafka     # Verify Kafka is running
make create-topics   # Create required topics

# Run services (separate terminals)
make run-collector   # Terminal 1: Stream collector
make run-multi       # Terminal 2: Multi-model detection

# Or run both in background
make run-all         # Start both services
make stop            # Stop both services

# Cleanup
make clean           # Remove generated data
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│  Go Aggregator (YOUR OTHER SERVICE)                         │
│  - Consumes raw exchange feeds                              │
│  - Computes z-scores, features                              │
│  - Publishes to "normalized-vectors"                        │
└────────────┬────────────────────────────────────────────────┘
             ↓
    Kafka Topic: "normalized-vectors"
             ↓
┌─────────────────────────────────────────────────────────────┐
│  Multi-Model Runner (make run-multi)                        │
│  - Reads config from baselines.yaml                         │
│  - Spawns 5 model workers                                   │
│  - All models score same vectors                            │
│  - All publish to "anomaly-scores" with model tag           │
└────────────┬────────────────────────────────────────────────┘
             ↓
    Kafka Topic: "anomaly-scores"
             ↓
┌─────────────────────────────────────────────────────────────┐
│  Stream Collector (make run-collector)                      │
│  - Reads config from baselines.yaml                         │
│  - Buffers 1000 messages                                    │
│  - Writes incrementally to parquet                          │
└────────────┬────────────────────────────────────────────────┘
             ↓
         scores.parquet
             ↓
┌─────────────────────────────────────────────────────────────┐
│  Evaluation (python scripts/evaluate_model.py)              │
│  - Pivots by "model" column                                 │
│  - Generates comparison plots                               │
└─────────────────────────────────────────────────────────────┘
```

---

## Complete Workflow

### 1. Prerequisites

```bash
# Check Kafka is running
make check-kafka

# Create topics
make create-topics

# Install dependencies (if not done)
make install

# Run integration test
make test
```

### 2. Start Services (3 terminals)

**Terminal 1 - Stream Collector:**
```bash
cd /Users/marioandresalbornoz/Desktop/Projects/inDevelopment/Thesis/rrcf-detector
make run-collector

# Or with custom config:
# make run-collector CONFIG=config/production.yaml
```

**Terminal 2 - Multi-Model Detection:**
```bash
cd /Users/marioandresalbornoz/Desktop/Projects/inDevelopment/Thesis/rrcf-detector
make run-multi

# Or with custom config:
# make run-multi CONFIG=config/production.yaml
```

**Terminal 3 - Go Aggregator:**
```bash
cd /path/to/your/go-aggregator
go run main.go
# (or however you start your Go aggregator)
```

### 3. Monitor Progress

**Stream collector output:**
```
✓ Subscribed to topic: anomaly-scores
  Consumer group: stream-collector-1726275600
  Bootstrap servers: localhost:9092

✓ Initialized parquet writer: ./data/scores.parquet
  Schema: 15 columns

  Flushed 1000 messages | Total: 1,000 | Rate: 250 msg/s | File: 0.52 MB
  Flushed 1000 messages | Total: 2,000 | Rate: 260 msg/s | File: 1.04 MB
```

**Multi-model service output:**
```
Initializing models:
  ✓ rrcf: RRCFDetectorAdapter
  ✓ zscore: ZScoreDetector
  ✓ isoforest: IsolationForestDetector
  ✓ halfspace: HalfSpaceTreesDetector
  ✓ onlineiforest: OnlineIForestDetector

Starting workers:
  ✓ rrcf worker (PID: 12345)
  ✓ zscore worker (PID: 12346)
  ...

Processed 1,000 messages | Rate: 300 msg/s
Processed 2,000 messages | Rate: 310 msg/s
```

### 4. Stop Services

Press `Ctrl+C` in each terminal:
1. Terminal 3: Stop Go aggregator
2. Terminal 2: Stop multi-model service
3. Terminal 1: Stop stream collector (auto-flushes buffer)

### 5. Evaluate Results

```bash
# Default evaluation
python scripts/evaluate_model.py \
    --scores-file ./data/scores.parquet \
    --output-dir ./results/run_001

# With ground truth (if available)
python scripts/evaluate_model.py \
    --scores-file ./data/scores.parquet \
    --ground-truth ./ground_truth/anomalies.json \
    --output-dir ./results/run_001

# View results
ls -lh ./results/run_001/
open ./results/run_001/*.png  # macOS
```

---

## Configuration Examples

### Different Output Directory
```yaml
# config/baselines.yaml
stream_collector:
  output_file: "./data/experiment_001/scores.parquet"
  # ... rest stays the same
```

### Production Settings
```yaml
# config/production.yaml
stream_collector:
  output_file: "/mnt/data/production/scores.parquet"
  buffer_size: 5000  # Larger buffer
  flush_interval: 10  # Less frequent writes
  bootstrap_servers: "kafka-prod-1:9092,kafka-prod-2:9092"
  topic: "anomaly-scores"

kafka:
  bootstrap_servers: "kafka-prod-1:9092,kafka-prod-2:9092"
  input_topic: "normalized-vectors"
  output_topic: "anomaly-scores"
```

Then run with:
```bash
make run-collector CONFIG=config/production.yaml
make run-multi CONFIG=config/production.yaml
```

---

## Troubleshooting

### "Config file missing 'stream_collector' section"
```bash
# Check your config has the section:
grep -A 5 "stream_collector:" config/baselines.yaml
```

### "Kafka is not running"
```bash
# Check Kafka status
make check-kafka

# Start Kafka (depends on your setup)
docker-compose up -d kafka  # or
brew services start kafka   # or
systemctl start kafka
```

### "No messages consumed"
```bash
# Check if Go aggregator is producing
kafka-console-consumer.sh --bootstrap-server localhost:9092 \
    --topic normalized-vectors --max-messages 1

# Check multi-model service logs
tail -f logs/multi_model.log  # if using make run-all
```

### "Empty parquet file"
```bash
# Check if multi-model service is producing scores
kafka-console-consumer.sh --bootstrap-server localhost:9092 \
    --topic anomaly-scores --max-messages 5

# Check stream collector is running
ps aux | grep stream_collector
```

---

## Benefits of Config-Driven Approach

✅ **Consistency**: Same config for all runs
✅ **Version Control**: Track config changes in git
✅ **Easy Makefile Integration**: Simple `make run-*` commands
✅ **Environment-Specific**: Different configs for dev/test/prod
✅ **No Command-Line Errors**: Typos eliminated
✅ **Documentation**: Config is self-documenting

---

## Summary

**Before (Command-line heavy)**:
```bash
python scripts/stream_collector.py \
    --output-file ./data/run_001/scores.parquet \
    --buffer-size 1000 \
    --flush-interval 5 \
    --bootstrap-servers localhost:9092 \
    --topic anomaly-scores
```

**After (Config-driven)**:
```bash
make run-collector  # All settings in baselines.yaml
```

**Much cleaner!** 🎉
