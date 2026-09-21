"""The sampled vector stream: the stride is applied once (in the runner), the surviving
vectors are recorded, and a replay of the recording gives every model the same vectors in
the same order. Includes an end-to-end test through the real run_multi_model.py (no Kafka)
that runs models in separate replays and checks they score exactly the same rows."""

import datetime as dt
import importlib.util
import json
import os
import queue
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import ServiceConfig
from src.detection.vector_sample import (
    StrideSampler,
    VectorSampleWriter,
    read_vector_sample,
    sample_row_count,
)
from src.kafka.consumer import NormalizedVectorDto

ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = ROOT / "scripts" / "run_multi_model.py"
_spec = importlib.util.spec_from_file_location("run_multi_model", _SCRIPT)
rmm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rmm)

_cspec = importlib.util.spec_from_file_location("check_vector_sample", ROOT / "scripts" / "check_vector_sample.py")
cvs = importlib.util.module_from_spec(_cspec)
_cspec.loader.exec_module(cvs)

T0 = dt.datetime(2021, 11, 8, 9, 0, 0, tzinfo=dt.timezone.utc)


def make_vectors(n, start=1, seed=0):
    """n deterministic vectors; the k-th has seq=start+k and a distinct timestamp."""
    rng = np.random.RandomState(seed)
    out = []
    for k in range(n):
        f = rng.normal(size=6)
        out.append(
            NormalizedVectorDto(
                exchange="FR",
                instrument=f"INS{k % 50}.FR",
                instrument_class="E",
                timestamp=T0 + dt.timedelta(milliseconds=k, microseconds=k % 7),
                model_key="FR:E",
                z_intertick_fast=float(f[0]),
                z_price_step_fast=float(f[1]),
                z_intertick_slow=float(f[2]),
                z_price_step_slow=float(f[3]),
                cusum_intertick=float(f[4]),
                cusum_price_step=float(f[5]),
                gap_flag=int(k % 11 == 0),
                warmup_flag=int(k < 5),
                session_fallback_flag=0,
                seq=start + k,
            )
        )
    return out


def record(path, vectors, buffer_rows=50_000):
    w = VectorSampleWriter(str(path), buffer_rows=buffer_rows)
    for i, v in enumerate(vectors, start=1):
        w.write(i, v)
    w.close()


# ------------------------------------------------------------------- StrideSampler


def test_stride_keeps_every_tenth_and_the_first_kept_is_the_tenth():
    s = StrideSampler(10)
    kept = [i for i in range(1, 101) if s.take()]
    assert kept == list(range(10, 101, 10))
    assert s.seen == 100


def test_stride_one_keeps_everything():
    s = StrideSampler(1)
    assert all(s.take() for _ in range(25))


def test_stride_must_be_positive():
    with pytest.raises(ValueError):
        StrideSampler(0)


# ------------------------------------------------------------- record / read round trip


def test_round_trip_is_field_for_field_equal_and_ordered(tmp_path):
    vectors = make_vectors(1234)
    path = tmp_path / "s.parquet"
    record(path, vectors, buffer_rows=100)  # several row groups
    assert sample_row_count(str(path)) == 1234
    assert list(read_vector_sample(str(path), batch_rows=97)) == vectors


def test_timestamps_keep_microseconds_and_the_same_instant(tmp_path):
    plus2 = dt.timezone(dt.timedelta(hours=2))
    v = make_vectors(1)[0]
    v.timestamp = dt.datetime(2021, 11, 8, 11, 0, 0, 123456, tzinfo=plus2)
    path = tmp_path / "s.parquet"
    record(path, [v])
    (back,) = list(read_vector_sample(str(path)))
    assert back.timestamp == v.timestamp  # same instant (aware datetimes compare by instant)
    assert back.timestamp.microsecond == 123456
    assert int(back.timestamp.timestamp() * 1000) == int(v.timestamp.timestamp() * 1000)


def test_file_only_appears_when_complete(tmp_path):
    path = tmp_path / "vectors" / "s.parquet"
    w = VectorSampleWriter(str(path), buffer_rows=10)
    for i, v in enumerate(make_vectors(25), start=1):
        w.write(i, v)
    assert not path.exists()  # still being written: only the .partial file exists
    assert Path(w.partial_path).exists()
    w.close()
    assert path.exists() and not Path(w.partial_path).exists()


def test_empty_recording_is_a_valid_empty_file(tmp_path):
    path = tmp_path / "s.parquet"
    record(path, [])
    assert sample_row_count(str(path)) == 0
    assert list(read_vector_sample(str(path))) == []


# ------------------------------------------------------- the runner applies the stride once


class _FakeProc:
    def __init__(self, alive=True):
        self.alive = alive

    def is_alive(self):
        return self.alive


class _FullQueue:
    """A queue that is always full."""

    def put(self, item, block=True, timeout=None):
        raise queue.Full


def make_runner(tmp_path, names=("zscore", "isoforest"), record_file=None, **kw):
    r = rmm.MultiModelRunner(ServiceConfig(), model_names=list(names), record_file=record_file, **kw)
    r.models = {n: {"queue": queue.Queue(), "process": None} for n in names}
    r.dropped_counts = {n: 0 for n in names}
    return r


def drain(q):
    items = []
    while not q.empty():
        items.append(q.get_nowait())
    return items


def test_runner_sends_the_same_every_tenth_vector_to_every_model_and_records_it(tmp_path):
    rec = tmp_path / "s.parquet"
    r = make_runner(tmp_path, record_file=str(rec))
    vectors = make_vectors(95)
    kept = [r._handle_vector(v) for v in vectors]
    r.recorder.close()

    expected = [vectors[i] for i in range(9, 95, 10)]  # 10th, 20th, ... 90th
    assert sum(kept) == 9
    assert drain(r.models["zscore"]["queue"]) == expected
    assert drain(r.models["isoforest"]["queue"]) == expected
    assert list(read_vector_sample(str(rec))) == expected
    stream_index = pd.read_parquet(rec)["stream_index"].tolist()
    assert stream_index == list(range(10, 91, 10))


def test_a_full_queue_drops_for_that_model_only_and_never_changes_the_recording(tmp_path):
    rec = tmp_path / "s.parquet"
    r = make_runner(tmp_path, record_file=str(rec))
    r.models["isoforest"]["queue"] = _FullQueue()
    vectors = make_vectors(100)
    for v in vectors:
        r._handle_vector(v)
    r.recorder.close()

    expected = [vectors[i] for i in range(9, 100, 10)]
    assert r.dropped_counts == {"zscore": 0, "isoforest": 10}
    assert drain(r.models["zscore"]["queue"]) == expected  # the other model is unaffected
    assert list(read_vector_sample(str(rec))) == expected  # and so is the recording


def test_default_stride_is_ten():
    assert rmm.SAMPLE_STRIDE == 10
    assert make_runner(None).sampler.stride == 10


# ---------------------------------------------------------------------------- replay


def test_replay_feeds_every_recorded_vector_in_order_then_the_sentinel(tmp_path):
    vectors = make_vectors(300)
    path = tmp_path / "s.parquet"
    record(path, vectors)
    r = make_runner(tmp_path)
    r.running = True
    r._run_replay(str(path), 300)
    for name in ("zscore", "isoforest"):
        assert drain(r.models[name]["queue"]) == vectors + [None]


def test_replay_does_not_apply_the_stride_again(tmp_path):
    # the recording is already sampled: a replay of 40 rows must send 40 rows, not 4
    path = tmp_path / "s.parquet"
    record(path, make_vectors(40))
    r = make_runner(tmp_path, names=("zscore",))
    r.running = True
    r._run_replay(str(path), 40)
    assert len(drain(r.models["zscore"]["queue"])) == 41  # 40 vectors + sentinel


def test_replay_fails_loudly_when_a_worker_died_instead_of_dropping(tmp_path):
    r = make_runner(tmp_path, names=("zscore",))
    r.models["zscore"] = {"queue": _FullQueue(), "process": _FakeProc(alive=False)}
    with pytest.raises(RuntimeError, match="died"):
        r._put("zscore", r.models["zscore"], make_vectors(1)[0], blocking=True)


def test_replay_refuses_to_overwrite_existing_scores(tmp_path):
    out = tmp_path / "scores.parquet"
    (tmp_path / "scores_zscore.parquet").write_bytes(b"precious")
    r = make_runner(tmp_path, names=("zscore", "isoforest"), parquet_file=str(out))
    with pytest.raises(SystemExit):
        r._check_outputs_free()
    assert (tmp_path / "scores_zscore.parquet").read_bytes() == b"precious"

    r2 = make_runner(tmp_path, names=("zscore",), parquet_file=str(out), overwrite=True)
    r2._check_outputs_free()  # allowed with --overwrite
    r3 = make_runner(tmp_path, names=("isoforest",), parquet_file=str(out))
    r3._check_outputs_free()  # nothing exists for isoforest


# ------------------------------------------------------------------- model selection


def test_model_selection_precedence():
    assert rmm.select_models("rrcf, zscore", {"models": ["isoforest"]}) == ["rrcf", "zscore"]
    assert rmm.select_models(None, {"models": ["isoforest"]}) == ["isoforest"]
    assert rmm.select_models(None, {}) == rmm.DEFAULT_MODELS


def test_registry_knows_every_model():
    reg = rmm.model_registry(ServiceConfig().partitioner_config.detector_config)
    assert set(reg) == {"rrcf", "zscore", "isoforest", "halfspace", "onlineiforest"}


# ------------------------------------------------- end to end, real workers, no Kafka


def _replay(sample, models, out_dir, extra=()):
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    return subprocess.run(
        [sys.executable, "-u", str(_SCRIPT), "--from-file", str(sample), "--models", models,
         "--output", str(out_dir / "scores.parquet"), *extra],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=600,
    )


def _keys(path):
    df = pd.read_parquet(path, columns=["exchange", "instrument", "timestamp_ms", "seq"])
    return list(df.itertuples(index=False, name=None))


def test_models_replayed_in_separate_runs_score_exactly_the_same_vectors(tmp_path):
    # zscore and isoforest train on the first 20,000 vectors and score the rest
    n = 22_000
    vectors = make_vectors(n)
    sample = tmp_path / "vectors_sample.parquet"
    record(sample, vectors)

    a, b = tmp_path / "a", tmp_path / "b"
    ra = _replay(sample, "zscore", a)
    assert ra.returncode == 0, ra.stdout[-2000:] + ra.stderr[-2000:]
    rb = _replay(sample, "isoforest,zscore", b)  # a different run, different model set
    assert rb.returncode == 0, rb.stdout[-2000:] + rb.stderr[-2000:]

    z_a, z_b, iso_b = (_keys(a / "scores_zscore.parquet"), _keys(b / "scores_zscore.parquet"),
                       _keys(b / "scores_isoforest.parquet"))
    expected = [(v.exchange, v.instrument, int(v.timestamp.timestamp() * 1000), v.seq)
                for v in vectors[20_000:]]
    assert z_a == expected
    assert z_b == expected
    assert iso_b == expected

    # zscore is deterministic, so its scores must be identical across the two runs too
    sa = pd.read_parquet(a / "scores_zscore.parquet")["z_score"].to_numpy()
    sb = pd.read_parquet(b / "scores_zscore.parquet")["z_score"].to_numpy()
    assert np.array_equal(sa, sb)


def test_replay_run_refuses_to_overwrite_scores_of_a_previous_run(tmp_path):
    sample = tmp_path / "vectors_sample.parquet"
    record(sample, make_vectors(100))
    out = tmp_path / "out"
    out.mkdir()
    (out / "scores_zscore.parquet").write_bytes(b"precious")
    r = _replay(sample, "zscore", out)
    assert r.returncode == 1
    assert (out / "scores_zscore.parquet").read_bytes() == b"precious"


# ------------------------------------------- the runner's summary and the archive check


def run_and_shut_down(tmp_path, n_vectors, tamper=None):
    """A live runner that consumed n_vectors, then shut down (writes the summary)."""
    rec = tmp_path / "vectors" / "vectors_sample.parquet"
    r = make_runner(tmp_path, record_file=str(rec))
    for v in make_vectors(n_vectors):
        r._handle_vector(v)
    if tamper:
        tamper(r)
    r._shutdown()
    return rec, r


def failed(results):
    return [d for d, ok, _ in results if not ok]


def test_shutdown_prints_and_writes_a_consistent_summary(tmp_path, capsys):
    rec, _ = run_and_shut_down(tmp_path, 95)
    out = capsys.readouterr().out
    assert "Vectors consumed from Kafka:      95" in out
    assert "consumed, kept and recorded counts agree" in out and "MISMATCH" not in out
    summary = json.loads((tmp_path / "vectors" / "vectors_sample.summary.json").read_text())
    assert (summary["consumed"], summary["kept"], summary["recorded_rows"], summary["stride"]) == (95, 9, 9, 10)
    assert summary["consistent"] is True
    assert summary["models"] == ["zscore", "isoforest"]


def test_the_summary_flags_a_kept_count_that_disagrees_with_the_stride(tmp_path, capsys):
    def tamper(r):
        r.sampled_count += 1  # the runner "kept" a vector the stride did not select
    run_and_shut_down(tmp_path, 95, tamper)
    assert "MISMATCH" in capsys.readouterr().out
    summary = json.loads((tmp_path / "vectors" / "vectors_sample.summary.json").read_text())
    assert summary["consistent"] is False


def test_a_replay_prints_a_summary_but_writes_no_file(tmp_path, capsys):
    path = tmp_path / "s.parquet"
    record(path, make_vectors(30))
    r = make_runner(tmp_path, names=("zscore",))
    r.mode, r.running = "replay", True
    r._run_replay(str(path), 30)
    assert "Vectors replayed: 30" in capsys.readouterr().out
    assert not (tmp_path / "s.summary.json").exists()


def test_checker_passes_on_a_real_recording(tmp_path):
    rec, _ = run_and_shut_down(tmp_path, 1234)
    results = cvs.check(str(rec))
    assert not failed(results), results
    assert cvs.main([str(rec)]) == 0


def test_checker_records_which_models_dropped_vectors(tmp_path):
    def tamper(r):
        r.dropped_counts["isoforest"] = 7
    rec, _ = run_and_shut_down(tmp_path, 200, tamper)
    results = cvs.check(str(rec))
    assert not failed(results)  # a note, not a failure
    assert any("isoforest dropped 7" in detail for _, _, detail in results)


def test_checker_fails_when_the_file_holds_fewer_rows_than_the_runner_recorded(tmp_path):
    rec, _ = run_and_shut_down(tmp_path, 95)
    # a truncated / different file: 8 rows where the summary says 9
    vectors = list(read_vector_sample(str(rec)))[:8]
    rec.unlink()
    w = VectorSampleWriter(str(rec))
    for i, v in enumerate(vectors, start=1):
        w.write(i * 10, v)
    w.close()
    bad = failed(cvs.check(str(rec)))
    assert "rows in the file == vectors the runner recorded" in bad
    assert "rows in the file == vectors the runner kept after the stride" in bad
    assert cvs.main([str(rec)]) == 1


def test_checker_fails_on_a_gap_in_stream_index(tmp_path):
    rec, _ = run_and_shut_down(tmp_path, 95)
    vectors = list(read_vector_sample(str(rec)))
    rec.unlink()
    w = VectorSampleWriter(str(rec))
    for i, v in enumerate(vectors, start=1):
        w.write(i * 10 + (10 if i > 4 else 0), v)  # positions jump after the 4th vector
    w.close()
    assert "stream_index runs stride, 2*stride, ... without a gap" in failed(cvs.check(str(rec)))


def test_checker_fails_when_the_summary_is_missing_or_disagrees(tmp_path):
    rec, _ = run_and_shut_down(tmp_path, 95)
    summary = tmp_path / "vectors" / "vectors_sample.summary.json"

    data = json.loads(summary.read_text())
    data["consumed"] = 500  # the runner "saw" many more vectors than the file accounts for
    summary.write_text(json.dumps(data))
    assert "kept == consumed // stride" in failed(cvs.check(str(rec)))

    summary.unlink()
    assert "runner summary exists" in failed(cvs.check(str(rec)))
    assert cvs.main([str(rec)]) == 1
