#!/usr/bin/env python3
"""
Collect data from Kafka topics for offline analysis.

Consumes all messages from input and output topics and saves to parquet files
for later evaluation. Useful after simulation runs to analyze detector performance.

Usage:
    python scripts/collect_data.py \\
        --input-topic normalized-features \\
        --output-topic anomaly-scores \\
        --output-dir ./data/run_001 \\
        --bootstrap-servers localhost:9092

Output:
    - <output-dir>/input_vectors.parquet
    - <output-dir>/output_scores.parquet
"""

import argparse
import json
import os
import sys
from datetime import datetime

import pandas as pd
from confluent_kafka import Consumer, KafkaError


def consume_topic(bootstrap_servers: str, topic: str, group_id: str) -> pd.DataFrame:
    """Consume all messages from a Kafka topic into a DataFrame."""
    
    config = {
        "bootstrap.servers": bootstrap_servers,
        "group.id": group_id,
        "auto.offset.reset": "earliest",
        "enable.auto.commit": False,
    }
    
    consumer = Consumer(config)
    consumer.subscribe([topic])
    
    print(f"Consuming from topic: {topic}")
    messages = []
    no_message_count = 0
    max_empty_polls = 10
    
    try:
        while True:
            msg = consumer.poll(timeout=1.0)
            
            if msg is None:
                no_message_count += 1
                if no_message_count >= max_empty_polls:
                    print(f"No new messages for {max_empty_polls} seconds, stopping...")
                    break
                continue
                
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                else:
                    print(f"Consumer error: {msg.error()}")
                    continue
            
            no_message_count = 0
            
            try:
                data = json.loads(msg.value().decode("utf-8"))
                data["_kafka_offset"] = msg.offset()
                data["_kafka_partition"] = msg.partition()
                data["_kafka_timestamp"] = msg.timestamp()[1] if msg.timestamp()[0] else None
                messages.append(data)
                
                if len(messages) % 1000 == 0:
                    print(f"  Collected {len(messages)} messages...")
                    
            except json.JSONDecodeError as e:
                print(f"Failed to decode message: {e}")
                continue
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        consumer.close()
    
    print(f"Collected {len(messages)} messages from {topic}")
    
    if not messages:
        return pd.DataFrame()
    
    return pd.DataFrame(messages)


def main():
    parser = argparse.ArgumentParser(
        description="Collect Kafka topic data for evaluation"
    )
    parser.add_argument(
        "--input-topic",
        default="normalized-features",
        help="Input topic with feature vectors"
    )
    parser.add_argument(
        "--output-topic",
        default="anomaly-scores",
        help="Output topic with anomaly scores"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save parquet files"
    )
    parser.add_argument(
        "--bootstrap-servers",
        default="localhost:9092",
        help="Kafka bootstrap servers"
    )
    parser.add_argument(
        "--skip-input",
        action="store_true",
        help="Skip collecting input topic (only collect scores)"
    )
    parser.add_argument(
        "--skip-output",
        action="store_true",
        help="Skip collecting output topic (only collect inputs)"
    )
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("Kafka Data Collection")
    print("=" * 60)
    print(f"Bootstrap servers: {args.bootstrap_servers}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    if not args.skip_input:
        print("Collecting input vectors...")
        input_df = consume_topic(
            args.bootstrap_servers,
            args.input_topic,
            f"collector-input-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        )
        
        if not input_df.empty:
            input_file = os.path.join(args.output_dir, "input_vectors.parquet")
            input_df.to_parquet(input_file, index=False)
            print(f"✓ Saved {len(input_df)} input vectors to {input_file}")
            print(f"  Columns: {list(input_df.columns)}")
            print()
        else:
            print("⚠ No input messages found")
            print()
    
    if not args.skip_output:
        print("Collecting output scores...")
        output_df = consume_topic(
            args.bootstrap_servers,
            args.output_topic,
            f"collector-output-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        )
        
        if not output_df.empty:
            output_file = os.path.join(args.output_dir, "output_scores.parquet")
            output_df.to_parquet(output_file, index=False)
            print(f"✓ Saved {len(output_df)} output scores to {output_file}")
            print(f"  Columns: {list(output_df.columns)}")
            print()
        else:
            print("⚠ No output messages found")
            print()
    
    print("=" * 60)
    print("Collection complete!")
    print(f"Data saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
