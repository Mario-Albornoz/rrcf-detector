import time
from dataclasses import dataclass
from datetime import datetime
from typing import List

import orjson
from confluent_kafka import Producer

from .metrics import KafkaMetrics


@dataclass
class AnomalyAlertDto:
    exchange: str
    instrument: str
    instrument_class: str
    timestamp: datetime
    raw_score: float
    alert_type: str


class AlertProducer:

    def __init__(self, config: dict, metrics_report_interval: int = 5):
        # Extract topic names if present (not valid Kafka producer configs)
        producer_config = {k: v for k, v in config.items() 
                          if k not in ['input_topic', 'output_topic', 'consumer_group_id', 
                                      'auto_offset_reset', 'producer']}
        
        # If there's a nested 'producer' config, merge those settings
        if 'producer' in config:
            producer_config.update(config['producer'])
        
        self.producer = Producer(producer_config)

        self.metrics = KafkaMetrics().producer
        self.metrics_report_interval = metrics_report_interval
        self._last_metrics_report = time.time()

    def publish(self, topic: str, message: dict):
        try:
            payload = self._serialize_alert(message)

            self.producer.produce(
                topic,
                value=payload,
                key=self._make_key(message),
                callback=self._delivery_callback,
            )

            self.producer.poll(0)
            self._maybe_report_metrics()

        except Exception as e:
            import traceback
            print(f"Error publishing alert: {e}")
            print(traceback.format_exc())
            self.metrics.record_failure()

    def publish_batch(self, topic: str, alerts: List[dict]):
        queued = 0

        try:
            for alert in alerts:
                try:
                    payload = self._serialize_alert(alert)

                    self.producer.produce(
                        topic,
                        value=payload,
                        key=self._make_key(alert),
                        callback=self._delivery_callback,
                    )
                    queued += 1

                except BufferError:
                    self.producer.poll(0.1)
                    try:
                        payload = self._serialize_alert(alert)
                        self.producer.produce(
                            topic,
                            value=payload,
                            key=self._make_key(alert),
                            callback=self._delivery_callback,
                        )
                        queued += 1
                    except Exception as e:
                        print(f"Failed to queue alert after retry: {e}")
                        self.metrics.record_failure()

                except Exception as e:
                    print(f"Error queuing alert: {e}")
                    self.metrics.record_failure()

            self.producer.poll(0)
            self._maybe_report_metrics()

        except Exception as e:
            print(f"Error publishing batch: {e}")

        return queued

    def _serialize_alert(self, alert: dict) -> bytes:
        # Pass through all fields from the alert
        # This supports both old format (alert_type) and new format (alert_level, model, z_score, etc.)
        # OPT_SERIALIZE_NUMPY: handle numpy.float64 and other numpy types automatically
        return orjson.dumps(alert, option=orjson.OPT_SERIALIZE_NUMPY)

    def _make_key(self, alert: dict) -> bytes:
        return f"{alert["exchange"]}:{alert["instrument"]}:{alert["instrument_class"]}".encode(
            "utf-8"
        )

    def _delivery_callback(self, err, msg):
        if err:
            print(f"Message delivery failed: {err}")
            self.metrics.record_delivery_error()
        else:
            self.metrics.record_success(len(msg.value()))

    def _maybe_report_metrics(self):
        now = time.time()
        if now - self._last_metrics_report >= self.metrics_report_interval:
            KafkaMetrics().report_all()
            self._last_metrics_report = now

    def flush(self, timeout: float = 10.0):
        remaining = self.producer.flush(timeout)
        if remaining > 0:
            print(f"Warning: {remaining} messages were not delivered")
        return remaining

    def close(self):
        print("Closing producer...")
        self.flush()
        KafkaMetrics().report_all()
        print("Producer closed")
