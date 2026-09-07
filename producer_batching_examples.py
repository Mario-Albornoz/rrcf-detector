"""
Producer batching comparison and configuration guide.
"""

from datetime import datetime

from src.kafka.producer import AlertProducer, AnomalyAlertDto

# ============================================================
# OPTIMAL PRODUCER CONFIGURATION FOR BATCHING
# ============================================================

# High-throughput configuration (optimized for batching)
producer_config_optimized = {
    "bootstrap.servers": "localhost:9092",
    "client.id": "rrcf-detector-producer",
    # BATCHING SETTINGS (KEY FOR PERFORMANCE)
    "linger.ms": 10,  # Wait up to 10ms to batch more messages
    "batch.size": 65536,  # 64KB batch size (default is 16KB)
    "batch.num.messages": 1000,  # Up to 1000 messages per batch
    # COMPRESSION (reduces network I/O)
    "compression.type": "lz4",  # or 'snappy', 'gzip', 'zstd'
    # PERFORMANCE TUNING
    "queue.buffering.max.messages": 100000,  # Internal queue size
    "queue.buffering.max.kbytes": 1048576,  # 1GB queue
    "message.send.max.retries": 3,
    "retry.backoff.ms": 100,
}

# Low-latency configuration (immediate send, no batching)
producer_config_lowlatency = {
    "bootstrap.servers": "localhost:9092",
    "client.id": "rrcf-detector-producer",
    "linger.ms": 0,  # Send immediately
    "batch.size": 1024,  # Small batches
    "acks": 1,  # Don't wait for all replicas
}


# ============================================================
# EXAMPLE 1: Single Alert Publishing (one-by-one)
# ============================================================


def example_single_publish():
    """
    Publishes one alert at a time.
    Internal batching still happens based on linger.ms and batch.size.
    """
    producer = AlertProducer(config=producer_config_optimized)

    # Publish 100 alerts one by one
    for i in range(100):
        alert = AnomalyAlertDto(
            exchange="binance",
            instrument=f"BTC-USDT-{i}",
            instrument_class="crypto_spot",
            timestamp=datetime.now(),
            alert_type="anomaly_detected",
        )
        producer.publish(topic="anomaly-alerts", alert=alert)

    # Flush to ensure all messages are sent
    producer.flush()
    producer.close()


# ============================================================
# EXAMPLE 2: Batch Publishing (explicit batches)
# ============================================================


def example_batch_publish():
    """
    Collects alerts and publishes in explicit batches.
    More efficient when you have multiple alerts ready at once.
    """
    producer = AlertProducer(config=producer_config_optimized)

    # Collect alerts into a batch
    alerts = []
    for i in range(100):
        alert = AnomalyAlertDto(
            exchange="binance",
            instrument=f"BTC-USDT-{i}",
            instrument_class="crypto_spot",
            timestamp=datetime.now(),
            alert_type="anomaly_detected",
        )
        alerts.append(alert)

    # Publish entire batch at once
    queued = producer.publish_batch(topic="anomaly-alerts", alerts=alerts)
    print(f"Queued {queued}/{len(alerts)} alerts")

    producer.flush()
    producer.close()


# ============================================================
# EXAMPLE 3: Streaming with Periodic Batches
# ============================================================


def example_streaming_with_batches():
    """
    Real-world pattern: collect alerts as they come, publish in batches.
    Balances latency vs throughput.
    """
    producer = AlertProducer(config=producer_config_optimized)

    alert_buffer = []
    BATCH_SIZE = 50

    # Simulate processing 1000 vectors
    for i in range(1000):
        # Process vector (imagine RRCF scoring here)
        if i % 10 == 0:  # 10% anomaly rate
            alert = AnomalyAlertDto(
                exchange="binance",
                instrument=f"BTC-USDT-{i}",
                instrument_class="crypto_spot",
                timestamp=datetime.now(),
                alert_type="anomaly_detected",
            )
            alert_buffer.append(alert)

        # Publish batch when buffer is full
        if len(alert_buffer) >= BATCH_SIZE:
            queued = producer.publish_batch(topic="anomaly-alerts", alerts=alert_buffer)
            print(f"Published batch: {queued} alerts")
            alert_buffer.clear()

    # Publish remaining alerts
    if alert_buffer:
        producer.publish_batch(topic="anomaly-alerts", alerts=alert_buffer)

    producer.flush()
    producer.close()


# ============================================================
# PERFORMANCE COMPARISON
# ============================================================

"""
Throughput comparison (approximate):

1. Single publish with linger.ms=0 (no batching):
   - ~5,000-10,000 msg/s
   - Low latency (~1-2ms)
   
2. Single publish with linger.ms=10 (internal batching):
   - ~20,000-40,000 msg/s
   - Medium latency (~10-15ms)
   
3. Explicit batch publish (publish_batch) with linger.ms=10:
   - ~50,000-100,000 msg/s
   - Medium latency (~10-20ms)
   - Best for high-throughput scenarios

Recommendation for RRCF detector:
- Use publish_batch() when you have multiple alerts ready
- Set linger.ms=5-10 for good throughput/latency balance
- Enable compression (lz4 or snappy) for network efficiency
"""


if __name__ == "__main__":
    print("Example 1: Single publish")
    example_single_publish()

    print("\nExample 2: Batch publish")
    example_batch_publish()

    print("\nExample 3: Streaming with batches")
    example_streaming_with_batches()
