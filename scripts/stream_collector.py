#!/usr/bin/env python3
"""
Streaming parquet collector for anomaly scores.

Consumes from Kafka anomaly-scores topic in real-time and writes incrementally
to parquet file. Runs in parallel with detection service without interference.

Key Features:
- Separate consumer group (no interference with detection service)
- Micro-batch buffering (memory efficient)
- Incremental parquet writes with pyarrow
- Graceful shutdown with buffer flush
- Progress monitoring

Usage:
    # Terminal 1: Start detection service
    python main.py --config config/baselines.yaml
    
    # Terminal 2: Start stream collector (parallel)
    python scripts/stream_collector.py \\
        --output-file ./data/run_001/scores.parquet \\
        --buffer-size 1000 \\
        --flush-interval 5
    
    # Terminal 3: Run feed simulator
    python simulator.py
    
    # After stopping: evaluate immediately
    python scripts/evaluate_model.py --data-file ./data/run_001/scores.parquet

Architecture:
    Kafka Topic "anomaly-scores"
            ↓
    Stream Collector (this script)
            ↓ (buffered writes)
    scores.parquet (grows in real-time)
            ↓
    evaluate_model.py (immediate analysis)
"""

import argparse
import json
import os
import signal
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from confluent_kafka import Consumer, KafkaError, KafkaException


class StreamCollector:
    """
    Streaming collector that consumes Kafka messages and writes to parquet incrementally.
    """
    
    def __init__(
        self,
        bootstrap_servers: str,
        topic: str,
        output_file: str,
        buffer_size: int = 1000,
        flush_interval: int = 5,
    ):
        """
        Initialize stream collector.
        
        Args:
            bootstrap_servers: Kafka bootstrap servers
            topic: Topic to consume from
            output_file: Path to output parquet file
            buffer_size: Number of messages to buffer before flush
            flush_interval: Time in seconds between forced flushes
        """
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
        
        # Metrics
        self.total_messages = 0
        self.total_bytes = 0
        self.last_flush_time = time.time()
        self.start_time = time.time()
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    
    def _init_consumer(self):
        """Initialize Kafka consumer with dedicated group ID."""
        consumer_config = {
            "bootstrap.servers": self.bootstrap_servers,
            "group.id": f"stream-collector-{int(time.time())}",  # Unique group ID
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
        """Parse Kafka message into dictionary."""
        try:
            data = json.loads(msg.value().decode("utf-8"))
            
            # Add Kafka metadata
            data["_kafka_offset"] = msg.offset()
            data["_kafka_partition"] = msg.partition()
            data["_kafka_timestamp"] = msg.timestamp()[1] if msg.timestamp()[0] else None
            
            return data
            
        except json.JSONDecodeError as e:
            print(f"⚠ Failed to decode message: {e}")
            return None
    
    def _infer_schema(self, df: pd.DataFrame) -> pa.Schema:
        """Infer pyarrow schema from first batch."""
        # Convert pandas dtypes to pyarrow types
        table = pa.Table.from_pandas(df)
        return table.schema
    
    def _flush_buffer(self):
        """Flush buffered messages to parquet file."""
        if not self.buffer:
            return
        
        try:
            df = pd.DataFrame(self.buffer)
            table = pa.Table.from_pandas(df)
            
            # Initialize writer on first flush
            if self.writer is None:
                self.schema = self._infer_schema(df)
                self.writer = pq.ParquetWriter(
                    self.output_file,
                    self.schema,
                    compression="snappy",
                    use_dictionary=True,
                    write_statistics=True,
                )
                print(f"✓ Initialized parquet writer: {self.output_file}")
                print(f"  Schema: {len(self.schema)} columns")
            
            # Ensure schema consistency
            if table.schema != self.schema:
                # Cast to expected schema if needed
                table = table.cast(self.schema)
            
            # Write batch
            self.writer.write_table(table)
            
            batch_size = len(self.buffer)
            self.total_messages += batch_size
            self.buffer.clear()
            self.last_flush_time = time.time()
            
            # Progress logging
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
        """Check if buffer should be flushed."""
        # Flush on buffer size
        if len(self.buffer) >= self.buffer_size:
            return True
        
        # Flush on time interval
        if time.time() - self.last_flush_time >= self.flush_interval:
            return True
        
        return False
    
    def start(self):
        """Start streaming collection."""
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
        """Main consumption loop."""
        try:
            while self.running:
                msg = self.consumer.poll(timeout=1.0)
                
                if msg is None:
                    # No message, check if should flush on interval
                    if self._should_flush():
                        self._flush_buffer()
                    continue
                
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    else:
                        raise KafkaException(msg.error())
                
                # Parse and buffer message
                data = self._parse_message(msg)
                if data:
                    self.buffer.append(data)
                    
                    # Flush if buffer full
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
        """Handle shutdown signals."""
        print(f"\n⚠ Received signal {signum}")
        self.running = False
    
    def _shutdown(self):
        """Graceful shutdown with buffer flush."""
        print()
        print("=" * 60)
        print("Shutting down stream collector...")
        print("=" * 60)
        
        # Flush remaining buffer
        if self.buffer:
            print(f"  Flushing remaining {len(self.buffer)} messages...")
            self._flush_buffer()
        
        # Close writer
        if self.writer:
            print("  Closing parquet writer...")
            self.writer.close()
            self.writer = None
        
        # Close consumer
        if self.consumer:
            print("  Closing Kafka consumer...")
            self.consumer.close()
        
        # Summary
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


def main():
    parser = argparse.ArgumentParser(
        description="Streaming parquet collector for anomaly scores",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python scripts/stream_collector.py --output-file ./data/run_001/scores.parquet
  
  # Custom buffer and flush settings
  python scripts/stream_collector.py \\
      --output-file ./data/run_001/scores.parquet \\
      --buffer-size 2000 \\
      --flush-interval 10
  
  # Custom Kafka settings
  python scripts/stream_collector.py \\
      --output-file ./data/run_001/scores.parquet \\
      --bootstrap-servers kafka:9092 \\
      --topic custom-scores-topic
"""
    )
    
    parser.add_argument(
        "--output-file",
        required=True,
        help="Path to output parquet file (will be created/overwritten)"
    )
    parser.add_argument(
        "--bootstrap-servers",
        default="localhost:9092",
        help="Kafka bootstrap servers (default: localhost:9092)"
    )
    parser.add_argument(
        "--topic",
        default="anomaly-scores",
        help="Kafka topic to consume from (default: anomaly-scores)"
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=1000,
        help="Number of messages to buffer before flush (default: 1000)"
    )
    parser.add_argument(
        "--flush-interval",
        type=int,
        default=5,
        help="Seconds between forced flushes (default: 5)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.buffer_size < 1:
        print("Error: buffer-size must be >= 1", file=sys.stderr)
        sys.exit(1)
    
    if args.flush_interval < 1:
        print("Error: flush-interval must be >= 1", file=sys.stderr)
        sys.exit(1)
    
    # Create and start collector
    collector = StreamCollector(
        bootstrap_servers=args.bootstrap_servers,
        topic=args.topic,
        output_file=args.output_file,
        buffer_size=args.buffer_size,
        flush_interval=args.flush_interval,
    )
    
    collector.start()


if __name__ == "__main__":
    main()
