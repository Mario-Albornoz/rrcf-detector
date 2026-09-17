"""
Generic worker for any detector model (RRCF or baselines).

Supports all models implementing BaseDetector interface:
- RRCF (via AnomalyDetector adapter)
- Z-Score Threshold
- Isolation Forest
- Half-Space Trees
"""

import multiprocessing as mp
import signal
from typing import Optional

from src.baselines import BaseDetector
from src.kafka.producer import AlertProducer


class GenericWorker:
    def __init__(
        self,
        worker_id: int,
        detector: BaseDetector,
        input_queue: mp.Queue,
        kafka_config: dict,
    ):
        self.worker_id = worker_id
        self.detector = detector
        self.input_queue = input_queue
        self.kafka_config = kafka_config

        self.publisher: Optional[AlertProducer] = None
        self.running = False

    def run(self):
        self.publisher = AlertProducer(config=self.kafka_config)

        signal.signal(signal.SIGTERM, self._shutdown_handler)
        signal.signal(signal.SIGINT, self._shutdown_handler)

        self.running = True
        model_name = self.detector.get_model_name()
        print(f"[Worker {self.worker_id}] Started ({model_name})")

        while self.running:
            try:
                vector = self.input_queue.get(timeout=1.0)

                if vector is None:
                    break

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

                    self.publisher.publish(
                        topic=self.kafka_config["output_topic"], message=score_output
                    )

            except Exception as e:
                if self.running:
                    import traceback
                    print(f"[Worker {self.worker_id}] Error: {e}")
                    print(traceback.format_exc())

        print(f"[Worker {self.worker_id}] Shutting down ({model_name})")

        if self.publisher:
            self.publisher.close()

    def _shutdown_handler(self, signum, _frame):
        print(f"[Worker {self.worker_id}] Received signal {signum}")
        self.running = False

    @staticmethod
    def start_worker(
        worker_id: int,
        detector: BaseDetector,
        input_queue: mp.Queue,
        kafka_config: dict,
    ) -> mp.Process:
        worker = GenericWorker(worker_id, detector, input_queue, kafka_config)
        process = mp.Process(target=worker.run, name=f"Worker-{worker_id}")
        process.start()
        return process
