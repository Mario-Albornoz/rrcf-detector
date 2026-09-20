"""
Generic worker for any detector model (RRCF or baselines).

Supports all models implementing BaseDetector interface:
- RRCF (via AnomalyDetector adapter)
- Z-Score Threshold
- Isolation Forest
- Half-Space Trees

Writes scores directly to parquet file for thesis evaluation.
"""

import multiprocessing as mp
import os
import queue
import signal
from pathlib import Path
from typing import Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.baselines import BaseDetector
from src.kafka.producer import AlertProducer


class ParquetWriter:
    """Buffered parquet writer for anomaly scores."""

    def __init__(self, output_file: str, buffer_size: int = 1000):
        self.output_file = output_file
        self.buffer_size = buffer_size
        self.buffer = []
        self.writer = None
        self.schema = self._create_schema()

        self.total_writes = 0
        self.total_flushes = 0
        self.total_rows_written = 0

        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    def _create_schema(self):
        """Define parquet schema matching thesis evaluation requirements."""
        return pa.schema(
            [
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
            ]
        )

    def write(self, score_dict):
        """Buffer a score for writing."""
        self.buffer.append(score_dict)
        self.total_writes += 1

        if len(self.buffer) >= self.buffer_size:
            self.flush()

    def flush(self):
        """Flush buffer to parquet file."""
        if not self.buffer:
            return

        try:
            rows_to_write = len(self.buffer)
            df = pd.DataFrame(self.buffer)
            table = pa.Table.from_pandas(df, schema=self.schema)

            if self.writer is None:
                self.writer = pq.ParquetWriter(
                    self.output_file,
                    self.schema,
                    compression="snappy",
                    use_dictionary=True,
                    write_statistics=True,
                )

            self.writer.write_table(table)
            self.total_flushes += 1
            self.total_rows_written += rows_to_write
            self.buffer.clear()

        except Exception as e:
            print(f"[ParquetWriter] Error flushing buffer: {e}")
            import traceback

            traceback.print_exc()

    def close(self):
        """Flush remaining buffer and close writer."""
        self.flush()
        if self.writer:
            self.writer.close()
            self.writer = None
            print(
                f"[ParquetWriter] Final stats: {self.total_rows_written:,} rows written in {self.total_flushes} flushes"
            )


class GenericWorker:
    def __init__(
        self,
        worker_id: int,
        detector: BaseDetector,
        input_queue: mp.Queue,
        kafka_config: dict,
        parquet_file: Optional[str] = None,
    ):
        self.worker_id = worker_id
        self.detector = detector
        self.input_queue = input_queue
        self.kafka_config = kafka_config
        self.parquet_file = parquet_file

        self.publisher: Optional[AlertProducer] = None
        self.parquet_writer: Optional[ParquetWriter] = None
        self.running = False

        self.messages_received = 0
        self.messages_processed = 0
        self.scores_written = 0
        self.last_report_time = None
        self.last_report_count = 0

    def run(self):
        if self.kafka_config.get("output_topic"):
            self.publisher = AlertProducer(config=self.kafka_config)
            print(f"[Worker {self.worker_id}] Kafka publishing ENABLED")
        else:
            print(f"[Worker {self.worker_id}] Kafka publishing DISABLED")

        if self.parquet_file:
            self.parquet_writer = ParquetWriter(self.parquet_file, buffer_size=1000)
            print(f"[Worker {self.worker_id}] Parquet output: {self.parquet_file}")

        signal.signal(signal.SIGTERM, self._shutdown_handler)
        signal.signal(signal.SIGINT, self._shutdown_handler)

        self.running = True
        model_name = self.detector.get_model_name()
        self.last_report_time = __import__("time").time()
        print(f"[Worker {self.worker_id}] Started ({model_name})")

        message_count = 0
        stride = 10
        while self.running:
            try:
                vector = self.input_queue.get(timeout=1.0)
                self.messages_received += 1
                message_count += 1

                if message_count % stride != 0:
                    continue

                if vector is None:
                    break

                self.messages_processed += 1
                result = self.detector.ingest_data(vector)

                if result is not None:
                    alert_level = self.detector.determine_alert_level(result["z_score"])

                    score_output = {
                        "exchange": vector.exchange,
                        "instrument": vector.instrument,
                        "instrument_class": vector.instrument_class,
                        "timestamp": vector.timestamp.isoformat(),
                        "timestamp_ms": int(vector.timestamp.timestamp() * 1000),
                        "model": model_name,
                        "raw_score": result["raw_score"],
                        "z_score": result["z_score"],
                        "alert_level": alert_level,
                        "stats_mean": result["stats"]["mean"],
                        "stats_std": result["stats"]["std"],
                        "stats_count": result["stats"]["count"],
                        "worker_id": self.worker_id,
                    }

                    if self.publisher:
                        self.publisher.publish(
                            topic=self.kafka_config["output_topic"],
                            message=score_output,
                        )

                    if self.parquet_writer:
                        self.parquet_writer.write(score_output)
                        self.scores_written += 1

                if self.messages_received % 10000 == 0:
                    self._report_metrics(model_name)

            except queue.Empty:
                # Normal timeout when no data is available - not an error
                continue
            except Exception as e:
                if self.running:
                    import traceback

                    print(f"[Worker {self.worker_id}] Error: {e}")
                    print(traceback.format_exc())

        print(f"\n[Worker {self.worker_id}] Final metrics ({model_name}):")
        self._report_metrics(model_name, final=True)

        print(f"[Worker {self.worker_id}] Shutting down ({model_name})")

        if self.parquet_writer:
            print(f"[Worker {self.worker_id}] Flushing Parquet buffer...")
            self.parquet_writer.close()
            print(f"[Worker {self.worker_id}] Parquet file closed: {self.parquet_file}")

        if self.publisher:
            self.publisher.close()

    def _report_metrics(self, model_name: str, final: bool = False):
        """Report worker metrics"""
        import time

        now = time.time()
        elapsed = now - self.last_report_time

        messages_since_last = self.messages_received - self.last_report_count
        receive_rate = messages_since_last / elapsed if elapsed > 0 else 0

        process_rate = self.messages_processed / (now - self.last_report_time + 0.001)

        stride_ratio = (
            (self.messages_processed / self.messages_received * 100)
            if self.messages_received > 0
            else 0
        )

        if final:
            print(f"  Total received:  {self.messages_received:,}")
            print(
                f"  Total processed: {self.messages_processed:,} ({stride_ratio:.1f}% after stride)"
            )
            print(f"  Scores written:  {self.scores_written:,}")
        else:
            print(
                f"[Worker {self.worker_id}] {model_name}: Received {self.messages_received:,} | "
                f"Processed {self.messages_processed:,} | Scores {self.scores_written:,} | "
                f"Rate: {receive_rate:.0f} msg/s in, {process_rate:.0f} scores/s out"
            )

        self.last_report_time = now
        self.last_report_count = self.messages_received

    def _shutdown_handler(self, signum, _frame):
        print(f"[Worker {self.worker_id}] Received signal {signum}")
        self.running = False

    @staticmethod
    def start_worker(
        worker_id: int,
        detector: BaseDetector,
        input_queue: mp.Queue,
        kafka_config: dict,
        parquet_file: Optional[str] = None,
    ) -> mp.Process:
        worker = GenericWorker(
            worker_id, detector, input_queue, kafka_config, parquet_file
        )
        process = mp.Process(target=worker.run, name=f"Worker-{worker_id}")
        process.start()
        return process
