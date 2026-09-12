import multiprocessing as mp
import signal
from typing import Optional

from src.detection.AnomalyDetector import AnomalyDetector, AnomalyDetectorConfig
from src.kafka.producer import AlertProducer


class Worker:
    def __init__(
        self,
        worker_id: int,
        config: AnomalyDetectorConfig,
        input_queue: mp.Queue,
        kafka_config: dict,
    ):
        self.worker_id = worker_id
        self.config: AnomalyDetectorConfig = config
        self.input_queue = input_queue
        self.kafka_config = kafka_config

        self.output_topic = kafka_config.get("output_topic", "anomaly-scores")

        self.detector: Optional[AnomalyDetector] = None
        self.publisher: Optional[AlertProducer] = None
        self.running = False

    def run(self):
        producer_config = {
            k: v for k, v in self.kafka_config.items() if k != "output_topic"
        }

        self.detector = AnomalyDetector(config=self.config)
        self.publisher = AlertProducer(config=producer_config)

        signal.signal(signal.SIGTERM, self._shutdown_handler)
        signal.signal(signal.SIGINT, self._shutdown_handler)

        self.running = True
        print(f"[Worker {self.worker_id}] Started")

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
                        "model": "rrcf",
                        "raw_score": result["raw_score"],
                        "z_score": result["z_score"],
                        "alert_level": alert_level,
                        "alert_type": f"anomaly_{alert_level}",
                        "stats_mean": result["stats"]["mean"],
                        "stats_std": result["stats"]["std"],
                        "stats_count": result["stats"]["count"],
                        "worker_id": self.worker_id,
                    }

                    self.publisher.publish(
                        topic=self.output_topic, message=score_output
                    )

            except Exception as e:
                if self.running:
                    print(f"[Worker {self.worker_id}] Error processing vector: {e}")

        print(f"[Worker {self.worker_id}] Shutting down")

        if self.publisher:
            self.publisher.close()

    def _shutdown_handler(self, signum, _frame):
        print(f"[Worker {self.worker_id}] Received signal {signum}")
        self.running = False

    @staticmethod
    def start_worker(
        worker_id: int,
        config: AnomalyDetectorConfig,
        input_queue: mp.Queue,
        kafka_config: dict,
    ) -> mp.Process:
        worker = Worker(worker_id, config, input_queue, kafka_config)
        process = mp.Process(target=worker.run, name=f"Worker-{worker_id}")
        process.start()
        return process
