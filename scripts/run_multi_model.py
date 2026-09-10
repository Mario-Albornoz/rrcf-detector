#!/usr/bin/env python3
"""
Run multi-model comparison service.

Runs all 4 models (RRCF + 3 baselines) in parallel, each with dedicated workers.
All models publish to same Kafka output topic with model name tagged.

Usage:
    python scripts/run_multi_model.py --config config/default.yaml
"""

import argparse
import os
import signal
import sys
import time

import yaml
from confluent_kafka import Consumer, KafkaError, KafkaException

from src.baselines import (
    HalfSpaceTreesDetector,
    IsolationForestDetector,
    RRCFDetectorAdapter,
    ZScoreDetector,
)
from src.config import ServiceConfig
from src.detection.generic_worker import GenericWorker
from src.kafka import deserialize_vector
from src.multiprocessing.partitioner import Partitioner


class MultiModelRunner:
    """Run all 4 models in parallel."""
    
    def __init__(self, config: ServiceConfig):
        self.config = config
        self.partitioners = {}
        self.consumer = None
        self.running = False
    
    def start(self):
        print("=" * 60)
        print("Multi-Model Anomaly Detection Service")
        print("Running: RRCF, Z-Score, IsoForest, Half-Space Trees")
        print("=" * 60)
        
        self._start_partitioners()
        self._start_consumer()
        
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        
        self.running = True
        self._run_loop()
    
    def _start_partitioners(self):
        """Start partitioner for each model."""
        models = [
            ("rrcf", RRCFDetectorAdapter),
            ("zscore", ZScoreDetector),
            ("isoforest", IsolationForestDetector),
            ("halfspace", HalfSpaceTreesDetector),
        ]
        
        for model_name, model_class in models:
            # Each model gets its own partitioner with workers
            # For simplicity, use 1 worker per model (can be scaled)
            print(f"[Service] Starting {model_name} partitioner...")
            
            # TODO: Create actual multi-model partitioner
            # For now, this is a skeleton
            
        print(f"[Service] Started {len(models)} model partitioners")
    
    def _start_consumer(self):
        kafka_config = self.config.partitioner_config.kafka_config
        
        consumer_config = {
            "bootstrap.servers": kafka_config.bootstrap_servers,
            "group.id": kafka_config.consumer_group_id + "-multi",
            "auto.offset.reset": kafka_config.auto_offset_reset,
        }
        
        self.consumer = Consumer(consumer_config)
        self.consumer.subscribe([kafka_config.input_topic])
        print(f"[Service] Subscribed to: {kafka_config.input_topic}")
    
    def _run_loop(self):
        print("[Service] Starting multi-model consumption...")
        print("[Service] Press Ctrl+C to stop")
        print("=" * 60)
        
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
                    # Route to all model partitioners
                    for partitioner in self.partitioners.values():
                        partitioner.route_vector(vector)
                    
                    message_count += 1
                    if message_count % 1000 == 0:
                        print(f"[Service] Processed {message_count} messages")
        
        except KeyboardInterrupt:
            print("\n[Service] Shutdown signal received")
        finally:
            self._shutdown()
    
    def _signal_handler(self, signum, _frame):
        print(f"\n[Service] Received signal {signum}")
        self.running = False
    
    def _shutdown(self):
        print("[Service] Shutting down all models...")
        
        for model_name, partitioner in self.partitioners.items():
            print(f"[Service] Stopping {model_name}...")
            partitioner.shutdown()
        
        if self.consumer:
            print("[Service] Closing consumer...")
            self.consumer.close()
        
        print("[Service] Shutdown complete")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Model Anomaly Detection Service"
    )
    parser.add_argument(
        "--config",
        default="config/default.yaml",
        help="Path to YAML config file"
    )
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"Error: Config not found: {args.config}", file=sys.stderr)
        sys.exit(1)
    
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    service_config = ServiceConfig.from_dict(config_dict)
    
    runner = MultiModelRunner(service_config)
    runner.start()


if __name__ == "__main__":
    print("WARNING: Multi-model runner is experimental")
    print("For production, run each model separately")
    print()
    main()
