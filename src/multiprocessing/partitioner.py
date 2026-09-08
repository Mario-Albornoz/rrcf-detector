import multiprocessing as mp
import signal
from typing import List

from src.config import PartitionerConfig
from src.detection.AnomalyDetector import AnomalyDetectorConfig
from src.detection.worker import Worker
from src.partitioning.strategy import hash_based_partitioner


class Partitioner:
    def __init__(self, config: PartitionerConfig):
        self.config = config
        self.num_workers = config.num_workers
        self.queue_max_size = config.queue_max_size

        self.partition_fn = hash_based_partitioner

        self.worker_queues: List[mp.Queue] = []
        self.worker_processes: List[mp.Process] = []
        self.running = False

    def start(self):
        print(f"[Partitioner] Starting {self.num_workers} workers...")

        for worker_id in range(self.num_workers):
            input_queue = mp.Queue(maxsize=self.queue_max_size)
            self.worker_queues.append(input_queue)

            detector_config = AnomalyDetectorConfig(
                window_size=self.config.detector_config.window_size,
                min_fill_threshold=self.config.detector_config.min_fill_threshold,
                normal_threshold=self.config.detector_config.normal_threshold,
                medium_threshold=self.config.detector_config.medium_threshold,
            )

            kafka_config = {
                "bootstrap_servers": self.config.kafka_config.bootstrap_servers,
                "output_topic": self.config.kafka_config.output_topic,
                "client_id": f"{self.config.kafka_config.client_id}-{worker_id}",
                "linger_ms": self.config.kafka_config.linger_ms,
                "batch_size": self.config.kafka_config.batch_size,
                "compression_type": self.config.kafka_config.compression_type,
                "acks": self.config.kafka_config.acks,
                "retries": self.config.kafka_config.retries,
            }

            process = Worker.start_worker(
                worker_id=worker_id,
                config=detector_config,
                input_queue=input_queue,
                kafka_config=kafka_config,
            )
            self.worker_processes.append(process)
            print(f"[Partitioner] Worker {worker_id} started (PID: {process.pid})")

        self.running = True

        signal.signal(signal.SIGTERM, self._shutdown_handler)
        signal.signal(signal.SIGINT, self._shutdown_handler)

        print(f"[Partitioner] All {self.num_workers} workers started")

    def route_vector(self, vector):
        worker_id = self.partition_fn(
            vector.exchange, vector.instrument_class, self.num_workers
        )

        try:
            self.worker_queues[worker_id].put(vector, block=True, timeout=1.0)
        except Exception as e:
            print(f"[Partitioner] Failed to route vector to worker {worker_id}: {e}")

    def shutdown(self):
        if not self.running:
            return

        print(f"[Partitioner] Shutting down {self.num_workers} workers...")
        self.running = False

        for worker_id, queue in enumerate(self.worker_queues):
            try:
                queue.put(None, block=False)
                print(f"[Partitioner] Sent shutdown signal to worker {worker_id}")
            except Exception as e:
                print(
                    f"[Partitioner] Failed to send shutdown to worker {worker_id}: {e}"
                )

        for worker_id, process in enumerate(self.worker_processes):
            print(f"[Partitioner] Waiting for worker {worker_id} to finish...")
            process.join(timeout=5.0)

            if process.is_alive():
                print(f"[Partitioner] Worker {worker_id} did not stop, terminating...")
                process.terminate()
                process.join(timeout=2.0)

                if process.is_alive():
                    print(f"[Partitioner] Worker {worker_id} still alive, killing...")
                    process.kill()
                    process.join()

            print(f"[Partitioner] Worker {worker_id} stopped")

        print(f"[Partitioner] Shutdown complete")

    def _shutdown_handler(self, signum, _frame):
        print(f"[Partitioner] Received signal {signum}, shutting down...")
        self.shutdown()

    def health_check(self) -> dict:
        return {
            worker_id: process.is_alive()
            for worker_id, process in enumerate(self.worker_processes)
        }

    def restart_worker(self, worker_id: int):
        old_process = self.worker_processes[worker_id]

        if old_process.is_alive():
            print(f"[Partitioner] Terminating old worker {worker_id}...")
            old_process.terminate()
            old_process.join(timeout=2.0)

        print(f"[Partitioner] Restarting worker {worker_id}...")

        detector_config = AnomalyDetectorConfig(
            window_size=self.config.detector_config.window_size,
            min_fill_threshold=self.config.detector_config.min_fill_threshold,
            normal_threshold=self.config.detector_config.normal_threshold,
            medium_threshold=self.config.detector_config.medium_threshold,
        )

        kafka_config = {
            "bootstrap_servers": self.config.kafka_config.bootstrap_servers,
            "output_topic": self.config.kafka_config.output_topic,
            "client_id": f"{self.config.kafka_config.client_id}-{worker_id}",
        }

        new_process = Worker.start_worker(
            worker_id=worker_id,
            config=detector_config,
            input_queue=self.worker_queues[worker_id],
            kafka_config=kafka_config,
        )

        self.worker_processes[worker_id] = new_process
        print(f"[Partitioner] Worker {worker_id} restarted (PID: {new_process.pid})")
