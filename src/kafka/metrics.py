import threading
import time
from typing import Dict, Optional


class KafkaMetrics:
    """
    Singleton metrics tracker for Kafka consumer and producer operations.
    Thread-safe implementation for use across multiple workers.
    """

    _instance: Optional["KafkaMetrics"] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self.start_time = time.time()

        # Consumer metrics
        self.consumer = MetricsSection("Consumer")

        # Producer metrics
        self.producer = MetricsSection("Producer")

        # Global metrics
        self._last_report_time = time.time()

    @classmethod
    def reset(cls):
        """Reset singleton instance (useful for testing)."""
        with cls._lock:
            cls._instance = None

    def report_all(self):
        """Print comprehensive metrics for both consumer and producer."""
        elapsed = time.time() - self.start_time

        print("\n" + "=" * 50)
        print(f"KAFKA METRICS - Uptime: {elapsed:.2f}s")
        print("=" * 50)

        self.consumer.report()
        self.producer.report()

        print("=" * 50 + "\n")

        self._last_report_time = time.time()


class MetricsSection:
    """Metrics section for either consumer or producer."""

    def __init__(self, name: str):
        self.name = name
        self._lock = threading.Lock()

        self.messages_processed = 0
        self.messages_failed = 0
        self.bytes_processed = 0

        # Consumer-specific
        self.deserialization_errors = 0
        self.handler_errors = 0

        # Producer-specific
        self.delivery_errors = 0
        self.delivery_timeouts = 0

        self._start_time = time.time()
        self._last_rate_check = time.time()
        self._last_processed_count = 0

    def record_success(self, message_size: int = 0):
        """Record successful message processing."""
        with self._lock:
            self.messages_processed += 1
            self.bytes_processed += message_size

    def record_failure(self):
        """Record general failure."""
        with self._lock:
            self.messages_failed += 1

    def record_deserialization_error(self):
        """Record deserialization error (consumer only)."""
        with self._lock:
            self.deserialization_errors += 1
            self.messages_failed += 1

    def record_handler_error(self):
        """Record handler error (consumer only)."""
        with self._lock:
            self.handler_errors += 1
            self.messages_failed += 1

    def record_delivery_error(self):
        """Record delivery error (producer only)."""
        with self._lock:
            self.delivery_errors += 1
            self.messages_failed += 1

    def record_delivery_timeout(self):
        """Record delivery timeout (producer only)."""
        with self._lock:
            self.delivery_timeouts += 1
            self.messages_failed += 1

    def get_throughput(self) -> float:
        """Returns average messages per second since start."""
        elapsed = time.time() - self._start_time
        return self.messages_processed / elapsed if elapsed > 0 else 0.0

    def get_current_rate(self) -> float:
        """Returns current messages per second since last check."""
        with self._lock:
            now = time.time()
            elapsed = now - self._last_rate_check
            if elapsed < 0.1:
                return 0.0
            rate = (self.messages_processed - self._last_processed_count) / elapsed
            self._last_rate_check = now
            self._last_processed_count = self.messages_processed
            return rate

    def report(self):
        """Print metrics summary for this section."""
        if self.messages_processed == 0 and self.messages_failed == 0:
            print(f"\n{self.name}: No activity")
            return

        throughput = self.get_throughput()
        current_rate = self.get_current_rate()

        print(f"\n{self.name}:")
        print(f"  Messages processed: {self.messages_processed:,}")
        print(f"  Messages failed: {self.messages_failed:,}")

        if self.deserialization_errors > 0:
            print(f"    - Deserialization errors: {self.deserialization_errors:,}")
        if self.handler_errors > 0:
            print(f"    - Handler errors: {self.handler_errors:,}")
        if self.delivery_errors > 0:
            print(f"    - Delivery errors: {self.delivery_errors:,}")
        if self.delivery_timeouts > 0:
            print(f"    - Delivery timeouts: {self.delivery_timeouts:,}")

        print(f"  Bytes processed: {self.bytes_processed:,}")
        print(f"  Average throughput: {throughput:.2f} msg/s")
        print(f"  Current rate: {current_rate:.2f} msg/s")
