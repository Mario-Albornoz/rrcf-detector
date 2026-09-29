#!/usr/bin/env python3

import argparse
import json
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from confluent_kafka import Consumer, KafkaError, KafkaException


class StreamCollector:
    def __init__(
        self,
        bootstrap_servers: str,
        topic: str,
        output_file: str,
        buffer_size: int = 1000,
        flush_interval: int = 5,
    ):
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.output_file = output_file
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        
        self.buffer: List[Dict] = []
        self.writer: Optional[pq.ParquetWriter] = None
        self.schema: Optional[pa.Schema] = None
        
        self.consumer: Optional[Consumer] = None
        self.running = False

        self.total_messages = 0
        self.total_bytes = 0
        self.last_flush_time = time.time()
        self.start_time = time.time()

        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    
    def _init_consumer(self):
        consumer_config = {
            "bootstrap.servers": self.bootstrap_servers,
            "group.id": f"stream-collector-{int(time.time())}",
            "auto.offset.reset": "earliest",
            "enable.auto.commit": True,
            "auto.commit.interval.ms": 5000,
        }
        
        self.consumer = Consumer(consumer_config)
        self.consumer.subscribe([self.topic])
        print(f"✓ Subscribed to topic: {self.topic}")
        print(f"  Consumer group: {consumer_config['group.id']}")
        print(f"  Bootstrap servers: {self.bootstrap_servers}")
    
    def _parse_message(self, msg) -> Optional[Dict]:
        try:
            data = json.loads(msg.value().decode("utf-8"))

            data["_kafka_offset"] = msg.offset()
            data["_kafka_partition"] = msg.partition()
            data["_kafka_timestamp"] = msg.timestamp()[1] if msg.timestamp()[0] else None
            
            return data
            
        except json.JSONDecodeError as e:
            print(f"⚠ Failed to decode message: {e}")
            return None
    
    def _create_schema(self) -> pa.Schema:
        return pa.schema([
            ("exchange", pa.string()),
            ("instrument", pa.string()),
            ("instrument_class", pa.string()),

            ("timestamp", pa.string()),
            ("timestamp_ms", pa.int64()),

            ("model", pa.string()),

            ("raw_score", pa.float64()),
            ("z_score", pa.float64()),
            ("alert_level", pa.string()),

            ("stats_mean", pa.float64()),
            ("stats_std", pa.float64()),
            ("stats_count", pa.int64()),

            ("worker_id", pa.int64()),

            ("_kafka_offset", pa.int64()),
            ("_kafka_partition", pa.int32()),
            ("_kafka_timestamp", pa.int64()),
        ])
    
    def _flush_buffer(self):
        if not self.buffer:
            return
        
        try:
            df = pd.DataFrame(self.buffer)
            table = pa.Table.from_pandas(df)

            if self.writer is None:
                self.schema = self._create_schema()
                self.writer = pq.ParquetWriter(
                    self.output_file,
                    self.schema,
                    compression="snappy",
                    use_dictionary=True,
                    write_statistics=True,
                )
                print(f"✓ Initialized parquet writer: {self.output_file}")
                print(f"  Schema: {len(self.schema)} columns")

            if table.schema != self.schema:
                table = table.cast(self.schema)

            self.writer.write_table(table)
            
            batch_size = len(self.buffer)
            self.total_messages += batch_size
            self.buffer.clear()
            self.last_flush_time = time.time()

            elapsed = time.time() - self.start_time
            rate = self.total_messages / elapsed if elapsed > 0 else 0
            file_size = os.path.getsize(self.output_file) if os.path.exists(self.output_file) else 0
            file_size_mb = file_size / (1024 * 1024)
            
            print(f"  Flushed {batch_size} messages | "
                  f"Total: {self.total_messages:,} | "
                  f"Rate: {rate:.1f} msg/s | "
                  f"File: {file_size_mb:.2f} MB")
            
        except Exception as e:
            print(f"✗ Error flushing buffer: {e}")
            import traceback
            traceback.print_exc()
    
    def _should_flush(self) -> bool:
        if len(self.buffer) >= self.buffer_size:
            return True

        if time.time() - self.last_flush_time >= self.flush_interval:
            return True
        
        return False
    
    def start(self):
        print("=" * 60)
        print("Stream Collector - Real-time Parquet Writer")
        print("=" * 60)
        print(f"Output file: {self.output_file}")
        print(f"Buffer size: {self.buffer_size} messages")
        print(f"Flush interval: {self.flush_interval} seconds")
        print()
        
        self._init_consumer()
        
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        
        self.running = True
        print("✓ Stream collector started")
        print("  Press Ctrl+C to stop")
        print("=" * 60)
        print()
        
        self._run_loop()
    
    def _run_loop(self):
        try:
            while self.running:
                msg = self.consumer.poll(timeout=1.0)
                
                if msg is None:
                    if self._should_flush():
                        self._flush_buffer()
                    continue
                
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    else:
                        raise KafkaException(msg.error())

                data = self._parse_message(msg)
                if data:
                    self.buffer.append(data)

                    if self._should_flush():
                        self._flush_buffer()
        
        except KeyboardInterrupt:
            print("\n⚠ Interrupted by user")
        
        except Exception as e:
            print(f"\n✗ Error in collection loop: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self._shutdown()
    
    def _signal_handler(self, signum, _frame):
        print(f"\n⚠ Received signal {signum}")
        self.running = False
    
    def _shutdown(self):
        print()
        print("=" * 60)
        print("Shutting down stream collector...")
        print("=" * 60)

        if self.buffer:
            print(f"  Flushing remaining {len(self.buffer)} messages...")
            self._flush_buffer()

        if self.writer:
            print("  Closing parquet writer...")
            self.writer.close()
            self.writer = None

        if self.consumer:
            print("  Closing Kafka consumer...")
            self.consumer.close()

        elapsed = time.time() - self.start_time
        rate = self.total_messages / elapsed if elapsed > 0 else 0
        file_size = os.path.getsize(self.output_file) if os.path.exists(self.output_file) else 0
        file_size_mb = file_size / (1024 * 1024)
        
        print()
        print("=" * 60)
        print("Collection Complete")
        print("=" * 60)
        print(f"  Total messages: {self.total_messages:,}")
        print(f"  Elapsed time: {elapsed:.1f} seconds")
        print(f"  Average rate: {rate:.1f} msg/s")
        print(f"  Output file: {self.output_file}")
        print(f"  File size: {file_size_mb:.2f} MB")
        print("=" * 60)


def load_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if 'stream_collector' not in config:
        raise ValueError(
            f"Config file missing 'stream_collector' section: {config_path}\n"
            "Add stream_collector configuration to your YAML file."
        )
    
    return config['stream_collector']


def main():
    parser = argparse.ArgumentParser(
        description="Streaming parquet collector for anomaly scores",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default config
  python scripts/stream_collector.py
  
  # Use custom config
  python scripts/stream_collector.py --config config/production.yaml
  
  # Override output file from config
  python scripts/stream_collector.py --output-file ./data/test/scores.parquet

Configuration:
  Settings are read from YAML config file under 'stream_collector' key.
  Default config: config/baselines.yaml
  
  Example config:
    stream_collector:
      output_file: "./data/scores.parquet"
      buffer_size: 1000
      flush_interval: 5
      bootstrap_servers: "localhost:9092"
      topic: "anomaly-scores"
  
  CLI arguments override config values.
"""
    )
    
    parser.add_argument(
        "--config",
        default="config/baselines.yaml",
        help="Path to YAML config file (default: config/baselines.yaml)"
    )
    parser.add_argument(
        "--output-file",
        help="Override output file from config"
    )
    parser.add_argument(
        "--bootstrap-servers",
        help="Override Kafka bootstrap servers from config"
    )
    parser.add_argument(
        "--topic",
        help="Override Kafka topic from config"
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        help="Override buffer size from config"
    )
    parser.add_argument(
        "--flush-interval",
        type=int,
        help="Override flush interval from config"
    )
    
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        print(f"✓ Loaded config from {args.config}")
    except Exception as e:
        print(f"✗ Error loading config: {e}", file=sys.stderr)
        sys.exit(1)

    output_file = args.output_file or config.get('output_file')
    bootstrap_servers = args.bootstrap_servers or config.get('bootstrap_servers', 'localhost:9092')
    topic = args.topic or config.get('topic', 'anomaly-scores')
    buffer_size = args.buffer_size or config.get('buffer_size', 1000)
    flush_interval = args.flush_interval or config.get('flush_interval', 5)

    if not output_file:
        print("✗ Error: output_file not specified in config or CLI", file=sys.stderr)
        sys.exit(1)
    
    if buffer_size < 1:
        print("✗ Error: buffer_size must be >= 1", file=sys.stderr)
        sys.exit(1)
    
    if flush_interval < 1:
        print("✗ Error: flush_interval must be >= 1", file=sys.stderr)
        sys.exit(1)
    
    print()

    collector = StreamCollector(
        bootstrap_servers=bootstrap_servers,
        topic=topic,
        output_file=output_file,
        buffer_size=buffer_size,
        flush_interval=flush_interval,
    )
    
    collector.start()


if __name__ == "__main__":
    main()
