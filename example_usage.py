"""
Example usage of Kafka consumer and producer with integrated metrics.
"""

from datetime import datetime

from src.kafka.consumer import NormalizedVectorConsumer, NormalizedVectorDto
from src.kafka.metrics import KafkaMetrics
from src.kafka.producer import AlertProducer, AnomalyAlertDto


def example_message_handler(vector: NormalizedVectorDto):
    """
    Example handler that processes normalized vectors.
    Replace with actual RRCF scoring logic.
    """
    # TODO: Implement RRCF forest scoring here
    # For now, just print high z-scores
    if abs(vector.z_intertick_fast) > 3.0:
        print(
            f"High z-score detected: {vector.exchange}/{vector.instrument} = {vector.z_intertick_fast}"
        )


def main():
    # Consumer configuration
    consumer_config = {
        "bootstrap.servers": "localhost:9092",
        "group.id": "rrcf-detector-consumer",
        "auto.offset.reset": "earliest",
        "enable.auto.commit": True,
    }

    # Producer configuration
    producer_config = {
        "bootstrap.servers": "localhost:9092",
        "client.id": "rrcf-detector-producer",
    }

    # Initialize consumer with bulk processing (batch_size=100)
    consumer = NormalizedVectorConsumer(
        config=consumer_config,
        message_handler=example_message_handler,
        metrics_report_interval=10,  # Report metrics every 10 seconds
        batch_size=100,  # Process 100 messages per batch for high throughput
    )

    # Initialize producer
    producer = AlertProducer(config=producer_config, metrics_report_interval=10)

    try:
        # Start consuming from topics
        # The consumer will automatically report metrics from both consumer and producer
        consumer.start(topics=["normalized-features"])

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        # Final metrics report
        KafkaMetrics().report_all()


if __name__ == "__main__":
    main()
