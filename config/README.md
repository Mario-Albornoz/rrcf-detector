# Configuration Files

## default.yaml

Default configuration for the RRCF anomaly detection service.

### Configuration Sections

#### service
- `num_workers`: Number of worker processes (default: 4)
- `queue_max_size`: Max size of routing queues (default: 10000)
- `log_level`: Logging level (default: INFO)

#### detector
- `window_size`: RRCF sliding window size (default: 1000)
- `min_fill_threshold`: Minimum samples before scoring (default: 50)

#### kafka
Connection and topic settings:
- `bootstrap_servers`: Kafka broker addresses
- `input_topic`: Topic to consume normalized vectors from
- `output_topic`: Topic to publish anomaly scores to

Consumer settings:
- `consumer_group_id`: Consumer group identifier
- `auto_offset_reset`: Where to start consuming (earliest/latest)
- `enable_auto_commit`: Auto-commit offsets
- `max_poll_records`: Batch size for consumption

Producer settings:
- `linger_ms`: Batching delay for producer
- `batch_size`: Max batch size in bytes
- `compression_type`: Compression algorithm (lz4/gzip/snappy)
- `acks`: Acknowledgment level (0/1/all)

## Creating Custom Configs

Copy `default.yaml` and modify for different experiments:

```bash
cp config/default.yaml config/experiment_001.yaml
# Edit experiment_001.yaml
python main.py --config config/experiment_001.yaml
```

## Environment Variable Overrides

Override any setting using environment variables:

```bash
export KAFKA_BOOTSTRAP_SERVERS="kafka:9092"
export NUM_WORKERS=8
python main.py
```
