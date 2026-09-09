import time

from src.config import DetectorConfig, KafkaConfig, PartitionerConfig
from src.kafka.consumer import NormalizedVectorConsumer
from src.multiprocessing import Partitioner


def main():
    detector_config = DetectorConfig(
        window_size=1000,
        min_fill_threshold=50,
    )

    kafka_config = KafkaConfig(
        bootstrap_servers="localhost:9092",
        input_topic="normalized-features",
        output_topic="anomaly-scores",
        consumer_group_id="rrcf-detector",
    )

    partitioner_config = PartitionerConfig(
        num_workers=4,
        queue_max_size=10000,
        detector_config=detector_config,
        kafka_config=kafka_config,
    )

    partitioner = Partitioner(partitioner_config)
    partitioner.start()

    consumer = NormalizedVectorConsumer(
        bootstrap_servers=kafka_config.bootstrap_servers,
        group_id=kafka_config.consumer_group_id,
        topics=[kafka_config.input_topic],
    )

    print("[Main] Starting message consumption loop...")
    print("[Main] Press Ctrl+C to stop")

    try:
        while True:
            message = consumer.poll(timeout=1.0)

            if message is not None:
                partitioner.route_vector(message)

            # Health check every 60 seconds
            if int(time.time()) % 60 == 0:
                health = partitioner.health_check()
                dead_workers = [wid for wid, alive in health.items() if not alive]

                if dead_workers:
                    print(f"[Main] WARNING: Dead workers detected: {dead_workers}")
                    for worker_id in dead_workers:
                        print(f"[Main] Restarting worker {worker_id}...")
                        partitioner.restart_worker(worker_id)

    except KeyboardInterrupt:
        print("\n[Main] Shutdown signal received")

    finally:
        print("[Main] Shutting down partitioner...")
        partitioner.shutdown()

        print("[Main] Closing consumer...")
        consumer.close()

        print("[Main] Shutdown complete")


if __name__ == "__main__":
    main()
