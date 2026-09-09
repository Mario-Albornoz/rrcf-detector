"""
RRCF Anomaly Detection Service - Main Entry Point

Consumes normalized feature vectors from Kafka, routes to worker processes
for RRCF scoring, and publishes anomaly scores to output Kafka topic.

Usage:
    python main.py [--config CONFIG_FILE]

    --config: Path to YAML config file (default: config/default.yaml)
"""

import argparse
import os
import signal
import sys
import time

import yaml
from confluent_kafka import Consumer, KafkaError, KafkaException

from src.config import ServiceConfig
from src.kafka import deserialize_vector
from src.multiprocessing.partitioner import Partitioner, PartitionerConfig


class ServiceRunner:
    def __init__(self, partitioner_config: PartitionerConfig):
        self.partitioner_config = partitioner_config
        self.partitioner = None
        self.consumer = None
        self.running = False

    def start(self):
        print("=" * 60)
        print("RRCF Anomaly Detection Service")
        print("=" * 60)

        kafka_config = self.partitioner_config.kafka_config

        consumer_config = {
            "bootstrap.servers": kafka_config.bootstrap_servers,
            "group.id": kafka_config.consumer_group_id,
            "auto.offset.reset": kafka_config.auto_offset_reset,
            "enable.auto.commit": kafka_config.enable_auto_commit,
            "auto.commit.interval.ms": kafka_config.auto_commit_interval_ms,
            "session.timeout.ms": kafka_config.session_timeout_ms,
            "max.poll.records": kafka_config.max_poll_records,
        }

        self.consumer = Consumer(consumer_config)
        self.consumer.subscribe([kafka_config.input_topic])
        print(f"[Service] Subscribed to topic: {kafka_config.input_topic}")

        self.partitioner = Partitioner(self.partitioner_config)
        self.partitioner.start()
        print(f"[Service] Started {self.partitioner_config.num_workers} workers")

        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

        self.running = True
        self._run_loop()

    def _run_loop(self):
        print("[Service] Starting message consumption loop...")
        print("[Service] Press Ctrl+C to stop")
        print("=" * 60)

        last_health_check = time.time()
        message_count = 0

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

                vector = deserialize_vector(msg)
                if vector:
                    self.partitioner.route_vector(vector)
                    message_count += 1

                    if message_count % 1000 == 0:
                        print(f"[Service] Processed {message_count} messages")

                now = time.time()
                if now - last_health_check >= 60:
                    self._health_check()
                    last_health_check = now

        except KeyboardInterrupt:
            print("\n[Service] Shutdown signal received")
        except Exception as e:
            print(f"[Service] Error: {e}", file=sys.stderr)
            raise
        finally:
            self._shutdown()

    def _health_check(self):
        health = self.partitioner.health_check()
        dead_workers = [wid for wid, alive in health.items() if not alive]

        if dead_workers:
            print(f"[Service] WARNING: Dead workers detected: {dead_workers}")
            for worker_id in dead_workers:
                print(f"[Service] Restarting worker {worker_id}...")
                self.partitioner.restart_worker(worker_id)
        else:
            print(f"[Service] Health check: All {len(health)} workers alive")

    def _signal_handler(self, signum, _frame):
        print(f"\n[Service] Received signal {signum}")
        self.running = False

    def _shutdown(self):
        print("[Service] Shutting down...")

        if self.partitioner:
            print("[Service] Stopping workers...")
            self.partitioner.shutdown()

        if self.consumer:
            print("[Service] Closing consumer...")
            self.consumer.close()

        print("[Service] Shutdown complete")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="RRCF Anomaly Detection Service")
    parser.add_argument(
        "--config",
        default="config/default.yaml",
        help="Path to YAML config file (default: config/default.yaml)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    with open(args.config, "r") as f:
        config_dict = yaml.safe_load(f)

    service_config = ServiceConfig.from_dict(config_dict)

    runner = ServiceRunner(service_config.partitioner_config)
    runner.start()


if __name__ == "__main__":
    main()
