"""The worker must stop on the shutdown sentinel, and it scores every vector it receives (the
stride is applied upstream, by the runner, not here)."""

import queue
import threading

import pytest

from src.detection.generic_worker import GenericWorker


class _NeverScoresDetector:
    def ingest_data(self, data):
        return None

    def get_model_name(self):
        return "stub"

    def determine_alert_level(self, z_score):
        return "NORMAL"


@pytest.mark.parametrize("n_messages", [0, 1, 3, 9, 10, 11, 25])
def test_worker_exits_on_sentinel_for_any_message_count(n_messages, monkeypatch):
    # run() installs signal handlers, which only works on the main thread; the worker
    # itself is driven from a helper thread here, so make that a no-op.
    monkeypatch.setattr("src.detection.generic_worker.signal.signal", lambda *a, **k: None)

    input_queue = queue.Queue()
    for i in range(n_messages):
        input_queue.put(object())
    input_queue.put(None)

    worker = GenericWorker(
        worker_id=0,
        detector=_NeverScoresDetector(),
        input_queue=input_queue,
        kafka_config={},
        parquet_file=None,
    )

    thread = threading.Thread(target=worker.run, daemon=True)
    thread.start()
    thread.join(timeout=5)

    assert not thread.is_alive(), (
        f"worker did not stop after the sentinel with {n_messages} messages queued"
    )


class _CountingDetector(_NeverScoresDetector):
    def __init__(self):
        self.ingested = 0

    def ingest_data(self, data):
        self.ingested += 1
        return None


@pytest.mark.parametrize("n_messages", [1, 9, 10, 11, 25])
def test_worker_ingests_every_message_it_receives(n_messages, monkeypatch):
    monkeypatch.setattr("src.detection.generic_worker.signal.signal", lambda *a, **k: None)

    input_queue = queue.Queue()
    for i in range(n_messages):
        input_queue.put(object())
    input_queue.put(None)

    detector = _CountingDetector()
    worker = GenericWorker(worker_id=0, detector=detector, input_queue=input_queue,
                           kafka_config={}, parquet_file=None)
    thread = threading.Thread(target=worker.run, daemon=True)
    thread.start()
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert detector.ingested == n_messages  # no second stride inside the worker
