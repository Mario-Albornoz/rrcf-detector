import json
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Optional

import ciso8601
import orjson
from confluent_kafka import Consumer, KafkaError, KafkaException

from .metrics import KafkaMetrics


@dataclass
class NormalizedVectorDto:
    exchange: str
    instrument: str
    instrument_class: str
    timestamp: datetime
    model_key: str
    z_intertick_fast: float
    z_price_step_fast: float
    z_intertick_slow: float
    z_price_step_slow: float
    cusum_intertick: float
    cusum_price_step: float
    gap_flag: int
    warmup_flag: int
    session_fallback_flag: int


class NormalizedVectorConsumer:
    """High-performance Kafka consumer for normalized feature vectors."""

    def __init__(
        self,
        config: dict,
        message_handler: Callable[[NormalizedVectorDto], None],
        metrics_report_interval: int = 5,
        batch_size: int = 100,
    ):
        self.config = config
        self.consumer = Consumer(config)
        self.is_listening: bool = True
        self.message_handler = message_handler

        self.metrics = KafkaMetrics().consumer
        self.metrics_report_interval = metrics_report_interval
        self._last_metrics_report = time.time()

        self.batch_size = batch_size

        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, signum, frame):
        print(f"\nReceived signal {signum}, initiating graceful shutdown...")
        self.shutdown()

    def start(self, topics: list[str]):
        try:
            self.consumer.subscribe(topics)
            print(f"Subscribed to topics: {topics}")

            while self.is_listening:
                messages = self.consumer.consume(
                    num_messages=self.batch_size, timeout=1.0
                )

                if not messages:
                    self._maybe_report_metrics()
                    continue

                for msg in messages:
                    if msg.error():
                        if msg.error().code() == KafkaError._PARTITION_EOF:
                            print(
                                f"Reached end of partition {msg.partition()} "
                                f"at offset {msg.offset()} for topic {msg.topic()}"
                            )
                        else:
                            raise KafkaException(msg.error())
                    else:
                        try:
                            vector = self._deserialize_message(msg)
                            if vector:
                                self.message_handler.ingest_data(vector)
                                self.metrics.record_success(len(msg.value()))
                            else:
                                self.metrics.record_deserialization_error()
                        except Exception as e:
                            self.metrics.record_handler_error()
                            print(f"Error processing message: {e}", file=sys.stderr)

                self._maybe_report_metrics()

        finally:
            KafkaMetrics().report_all()
            self.consumer.close()
            print("Consumer closed")

    def _deserialize_message(self, msg) -> Optional[NormalizedVectorDto]:
        """Deserialize Kafka message to NormalizedVectorDto."""
        try:
            data = orjson.loads(msg.value().decode("utf-8"))

            timestamp = ciso8601.parse_datetime(data["timestamp"])

            return NormalizedVectorDto(
                exchange=data["exchange"],
                instrument=data["instrument"],
                instrument_class=data["class"],
                timestamp=timestamp,
                model_key=data["model_key"],
                z_intertick_fast=data["z_intertick_fast"],
                z_price_step_fast=data["z_price_step_fast"],
                z_intertick_slow=data["z_intertick_slow"],
                z_price_step_slow=data["z_price_step_slow"],
                cusum_intertick=data["cusum_intertick"],
                cusum_price_step=data["cusum_price_step"],
                gap_flag=data["gap_flag"],
                warmup_flag=data["warmup_flag"],
                session_fallback_flag=data["session_fallback_flag"],
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Failed to deserialize message: {e}", file=sys.stderr)
            return None

    def _maybe_report_metrics(self):
        now = time.time()
        if now - self._last_metrics_report >= self.metrics_report_interval:
            KafkaMetrics().report_all()
            self._last_metrics_report = now

    def shutdown(self):
        print("Shutting down consumer...")
        self.is_listening = False
