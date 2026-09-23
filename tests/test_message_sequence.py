"""The message sequence number: from the wire to the scores to the evaluation."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from src.detection.generic_worker import ParquetWriter
from src.kafka.consumer import deserialize_vector

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_thesis.py"
_spec = importlib.util.spec_from_file_location("evaluate_thesis", _SCRIPT)
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)


class _Msg:
    def __init__(self, payload: dict):
        self._payload = json.dumps(payload).encode()

    def value(self):
        return self._payload


VECTOR = {
    "exchange": "ETR", "instrument": "SAP.ETR", "class": "E", "timestamp": "2021-11-10T10:00:00.412Z",
    "model_key": "E", "z_intertick_fast": 0.1, "z_price_step_fast": 0.2, "z_intertick_slow": 0.3,
    "z_price_step_slow": 0.4, "cusum_intertick": 0.0, "cusum_price_step": 0.0, "gap_flag": 0,
    "warmup_flag": 0, "session_fallback_flag": 0,
}


def test_vector_seq_is_decoded_and_defaults_to_zero():
    assert deserialize_vector(_Msg({**VECTOR, "seq": 987654})).seq == 987654
    assert deserialize_vector(_Msg(VECTOR)).seq == 0   # an older producer sends none


def test_scores_parquet_has_a_seq_column(tmp_path):
    path = tmp_path / "scores_rrcf.parquet"
    w = ParquetWriter(str(path), buffer_size=1)
    w.write({
        "exchange": "ETR", "instrument": "SAP.ETR", "instrument_class": "E", "timestamp": "t", "timestamp_ms": 1,
        "model": "rrcf", "raw_score": 1.0, "z_score": 2.0, "alert_level": "HIGH", "stats_mean": 0.0,
        "stats_std": 1.0, "stats_count": 1, "worker_id": 0, "seq": 42,
    })
    w.close()
    table = pq.read_table(path)
    assert "seq" in table.column_names and table.column("seq").to_pylist() == [42]


EPISODE_COLUMNS = ["EpisodeID", "Phase", "AnomalyType", "Exchange", "InstrumentID", "StartMs", "EndMs",
                   "ObservedMs", "ResumeMs", "LastDeliveredMs", "Detail", "SecType", "Seq"]
T = 1_636_538_400_412


def _dataset(tmp_path, with_seq_in_scores: bool):
    """One injected spike (message 100). Its neighbour, message 99, shares its millisecond;
    only the neighbour has a scored vector, and that vector is an alert (e.g. the price
    snapping back). Matching on (instrument, time) wrongly credits the injected message."""
    episodes = pd.DataFrame([{
        "EpisodeID": 1, "Phase": "phase2", "AnomalyType": "price_spike", "Exchange": "ETR", "InstrumentID": "X.ETR",
        "StartMs": T, "EndMs": T, "ObservedMs": T, "ResumeMs": None, "LastDeliveredMs": None,
        "Detail": "multiplier=3", "SecType": "E", "Seq": 100,
    }], columns=EPISODE_COLUMNS)
    other_day = 86_400_000
    scores = pd.DataFrame({
        "exchange": "ETR", "instrument": ["X.ETR", "X.ETR", "Y.ETR"],
        "timestamp_ms": [T, T + other_day, T],
        "z_score": [5.0, 0.1, 0.1], "seq": [99, 500, 7],
    })
    if not with_seq_in_scores:
        scores = scores.drop(columns=["seq"])
    ep_path, sc_path = tmp_path / "anomaly_log_episodes.csv", tmp_path / "scores_rrcf.parquet"
    episodes.to_csv(ep_path, index=False)
    scores.to_parquet(sc_path)
    return ep_path, sc_path


def _run(tmp_path, ep_path, sc_path):
    args = ev.parse_args(["--episodes", str(ep_path), "--scores", str(sc_path), "--score-column", "z_score", "--output", str(tmp_path / "o"),
                          "--bootstrap", "0"])
    return ev.evaluate(args)["rrcf"]["phase2"]["point"]["price_spike"]


def test_exact_matching_does_not_credit_a_neighbouring_message(tmp_path):
    block = _run(tmp_path, *_dataset(tmp_path, with_seq_in_scores=True))
    assert block["strict_matching"] == "message sequence number"
    assert block["strict_exact_tick"]["scorable"] == 0   # message 100 was never scored
    assert block["strict_exact_tick"]["detected"] == 0


def test_time_matching_is_ambiguous_and_says_so(tmp_path):
    block = _run(tmp_path, *_dataset(tmp_path, with_seq_in_scores=False))
    assert "ambiguous" in block["strict_matching"]
    # the documented weakness: the neighbour at the same millisecond is taken for the injected message
    assert block["strict_exact_tick"]["scorable"] == 1 and block["strict_exact_tick"]["detected"] == 1
