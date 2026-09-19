#!/usr/bin/env python3
"""
Run multi-model comparison service.

Runs all 5 models (RRCF + 4 baselines) in parallel, each with dedicated workers.
All models publish to same Kafka output topic with model name tagged.

Usage:
    python scripts/run_multi_model.py --config config/baselines.yaml

Architecture:
    One Kafka consumer feeds all models in parallel:
    - Each model has its own worker process
    - Each worker has its own input queue
    - All workers publish to same output topic with "model" tag
    - Stream collector consumes output for evaluation
"""

import argparse
import multiprocessing as mp
import os
import signal
import sys
import time
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml
from confluent_kafka import Consumer, KafkaError, KafkaException

from src.baselines import (
    HalfSpaceTreesDetector,
    IsolationForestDetector,
    OnlineIForestDetector,
    RRCFDetectorAdapter,
    ZScoreDetector,
)
from src.config import ServiceConfig
from src.detection.generic_worker import GenericWorker
from src.kafka import deserialize_vector


class MultiModelRunner:
    """
    Run all 5 models in parallel.

    Each model gets:
    - One dedicated worker process
    - One input queue for message routing
    - Access to shared Kafka producer (output topic)
    """

    def __init__(self, config: ServiceConfig, parquet_file: str = None):
        self.config = config
        self.parquet_file = parquet_file or "./data/scores.parquet"
        self.models = (
            {}
        )  # {model_name: {"detector": class, "queue": Queue, "process": Process}}
        self.consumer = None
        self.running = False

    def start(self):
        print("=" * 60)
        print("Multi-Model Anomaly Detection Service")
        print("=" * 60)
        print()

        self._init_models()
        self._start_workers()
        self._start_consumer()

        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

        self.running = True
        self._run_loop()

    def _init_models(self):
        """Initialize model configurations."""
        detector_config = self.config.partitioner_config.detector_config

        # Define all 5 models with their detector classes
        model_configs = [
            (
                "rrcf",
                RRCFDetectorAdapter,
                {
                    "window_size": detector_config.window_size,
                    "min_fill_threshold": detector_config.min_fill_threshold,
                },
            ),
            ("zscore", ZScoreDetector, {"training_samples": 20000}),
            (
                "isoforest",
                IsolationForestDetector,
                {"training_samples": 20000, "n_estimators": 100, "contamination": 0.1},
            ),
            (
                "halfspace",
                HalfSpaceTreesDetector,
                {
                    "window_size": detector_config.window_size,
                    "n_trees": 25,
                    "height": 8,
                    "min_fill_threshold": detector_config.min_fill_threshold,
                },
            ),
            (
                "onlineiforest",
                OnlineIForestDetector,
                {
                    "window_size": 1024,
                    "num_trees": 32,
                    "max_leaf_samples": 32,
                    "type": "adaptive",
                    "min_fill_threshold": detector_config.min_fill_threshold,
                },
            ),
        ]

        print("Initializing models:")
        for model_name, detector_class, config in model_configs:
            self.models[model_name] = {
                "detector_class": detector_class,
                "config": config,
                "queue": None,
                "process": None,
            }
            print(f"  ✓ {model_name}: {detector_class.__name__}")
        print()

    def _start_workers(self):
        """Start one worker process per model."""
        kafka_config = self.config.partitioner_config.kafka_config

        print("Starting workers:")
        for model_name, model_info in self.models.items():
            # Create input queue for this model
            queue = mp.Queue(maxsize=1000)
            model_info["queue"] = queue

            # Initialize detector instance
            detector = model_info["detector_class"](model_info["config"])

            # Build Kafka producer config for worker
            worker_kafka_config = {
                "bootstrap.servers": kafka_config.bootstrap_servers,
                "output_topic": kafka_config.output_topic,
                "client.id": f"multi-model-{model_name}",
                "linger.ms": kafka_config.linger_ms,
                "batch.size": kafka_config.batch_size,
                "compression.type": kafka_config.compression_type,
                "acks": kafka_config.acks,
                "retries": kafka_config.retries,
            }

            # Start worker process
            # Each model writes to its own parquet file to avoid concurrent write corruption
            model_parquet_file = None
            if self.parquet_file:
                base_dir = Path(self.parquet_file).parent
                model_parquet_file = str(base_dir / f"scores_{model_name}.parquet")
            
            process = GenericWorker.start_worker(
                worker_id=0,  # Single worker per model
                detector=detector,
                input_queue=queue,
                kafka_config=worker_kafka_config,
                parquet_file=model_parquet_file,
            )

            model_info["process"] = process
            print(f"  ✓ {model_name} worker (PID: {process.pid})")

        print(f"\n✓ Started {len(self.models)} model workers")
        print()

    def _start_consumer(self):
        """Start Kafka consumer for input vectors."""
        kafka_config = self.config.partitioner_config.kafka_config

        consumer_config = {
            "bootstrap.servers": kafka_config.bootstrap_servers,
            "group.id": kafka_config.consumer_group_id + "-multi",
            "auto.offset.reset": kafka_config.auto_offset_reset,
            "enable.auto.commit": True,
            "auto.commit.interval.ms": 5000,
        }

        self.consumer = Consumer(consumer_config)
        self.consumer.subscribe([kafka_config.input_topic])
        print(f"✓ Subscribed to input topic: {kafka_config.input_topic}")
        print()

    def _run_loop(self):
        """Main consumption loop - fan out vectors to all models."""
        print("=" * 60)
        print("Starting message consumption")
        print("Press Ctrl+C to stop")
        print("=" * 60)
        print()

        message_count = 0
        last_report = time.time()
        
        # Track dropped messages per model
        dropped_counts = {model_name: 0 for model_name in self.models.keys()}
        total_dropped = 0

        try:
            while self.running:
                msg = self.consumer.poll(timeout=1.0)

                if msg is None:
                    continue

                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    else:
                        raise KafkaException(msg.error())

                # Deserialize vector
                vector = deserialize_vector(msg)
                if vector:
                    # Fan out to ALL models (each gets a copy)
                    for model_name, model_info in self.models.items():
                        try:
                            model_info["queue"].put(vector, block=False)
                        except Exception:
                            # Track dropped messages without spamming logs
                            dropped_counts[model_name] += 1
                            total_dropped += 1

                    message_count += 1

                    # Progress report every 10000 messages
                    if message_count % 10000 == 0:
                        elapsed = time.time() - last_report
                        rate = 10000 / elapsed if elapsed > 0 else 0
                        
                        # Show aggregate drop stats
                        drop_summary = " | ".join([
                            f"{name}: {count:,} dropped" 
                            for name, count in dropped_counts.items() 
                            if count > 0
                        ])
                        
                        if drop_summary:
                            print(f"Processed {message_count:,} messages | Rate: {rate:.0f} msg/s | {drop_summary}")
                        else:
                            print(f"Processed {message_count:,} messages | Rate: {rate:.0f} msg/s | No drops")
                        
                        last_report = time.time()

        except KeyboardInterrupt:
            print("\n⚠ Shutdown signal received")
            print("\n" + "=" * 60)
            print("Message Routing Statistics")
            print("=" * 60)
            print(f"Total messages processed: {message_count:,}")
            print(f"Total messages dropped: {total_dropped:,}")
            if total_dropped > 0:
                drop_rate = (total_dropped / (message_count * len(self.models))) * 100
                print(f"Drop rate: {drop_rate:.2f}%")
                print("\nDrops by model:")
                for model_name, count in sorted(dropped_counts.items(), key=lambda x: x[1], reverse=True):
                    if count > 0:
                        model_drop_rate = (count / message_count) * 100 if message_count > 0 else 0
                        print(f"  {model_name:15s}: {count:,} ({model_drop_rate:.2f}% of messages)")
            else:
                print("No messages dropped!")
            print("=" * 60)
            print()
        except Exception as e:
            print(f"\n✗ Error in consumption loop: {e}")
            import traceback

            traceback.print_exc()
        finally:
            self._shutdown()

    def _signal_handler(self, signum, _frame):
        print(f"\n⚠ Received signal {signum}")
        self.running = False

    def _shutdown(self):
        """Gracefully shutdown all workers and consumer."""
        print()
        print("=" * 60)
        print("Shutting down multi-model service")
        print("=" * 60)

        # Send shutdown signal to all workers (None message)
        print("Stopping workers:")
        for model_name, model_info in self.models.items():
            try:
                if model_info["queue"]:
                    model_info["queue"].put(None, block=False)
                print(f"  ✓ Sent shutdown to {model_name}")
            except Exception as e:
                print(f"  ✗ Failed to signal {model_name}: {e}")

        # Wait for workers to finish
        print("\nWaiting for workers to finish:")
        for model_name, model_info in self.models.items():
            process = model_info["process"]
            if process and process.is_alive():
                print(f"  Waiting for {model_name}...")
                process.join(timeout=5.0)

                if process.is_alive():
                    print(f"  ⚠ {model_name} did not stop, terminating...")
                    process.terminate()
                    process.join(timeout=2.0)

                if process.is_alive():
                    print(f"  ⚠ {model_name} still alive, killing...")
                    process.kill()
                    process.join()

                print(f"  ✓ {model_name} stopped")

        # Close consumer
        if self.consumer:
            print("\nClosing consumer...")
            self.consumer.close()
            print("  ✓ Consumer closed")

        print()
        print("=" * 60)
        print("Shutdown complete")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Model Anomaly Detection Service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all 5 models with default config
  python scripts/run_multi_model.py
  
  # Run with custom config
  python scripts/run_multi_model.py --config config/baselines.yaml

Models:
  1. RRCF - Robust Random Cut Forest (online, 2016)
  2. Z-Score - Statistical threshold (batch/frozen)
  3. IsoForest - Isolation Forest (batch/frozen, 2008)
  4. Half-Space Trees - Online streaming (2011)
  5. Online-iForest - Online Isolation Forest (2024)

Output:
  All models write to Kafka "anomaly-scores" topic with "model" tag.
  Use stream_collector.py to capture scores in real-time.
""",
    )
    parser.add_argument(
        "--config",
        default="config/baselines.yaml",
        help="Path to YAML config file (default: config/baselines.yaml)",
    )
    parser.add_argument(
        "--output",
        default="./data/scores.parquet",
        help="Path to output parquet file (default: ./data/scores.parquet)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"✗ Error: Config not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    print("Loading configuration...")
    with open(args.config, "r") as f:
        config_dict = yaml.safe_load(f)

    service_config = ServiceConfig.from_dict(config_dict)
    print(f"✓ Loaded config from {args.config}")
    print(f"✓ Output file: {args.output}")
    print()

    runner = MultiModelRunner(service_config, parquet_file=args.output)
    runner.start()


if __name__ == "__main__":
    main()
