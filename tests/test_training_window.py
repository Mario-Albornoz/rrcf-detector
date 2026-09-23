"""Frozen baselines train on the whole first day of data (training_days) or on a fixed
number of vectors (training_samples), with bounded memory."""

import datetime as dt

import numpy as np

from src.baselines import IsolationForestDetector, ZScoreDetector
from src.baselines.training_window import Reservoir, TrainingWindow
from src.kafka.consumer import NormalizedVectorDto

DAY1 = dt.datetime(2021, 11, 8, 9, 30, tzinfo=dt.timezone.utc)


def _vec(ts, values):
    return NormalizedVectorDto(
        exchange="ETR", instrument="SAP.ETR", instrument_class="E", timestamp=ts, model_key="E",
        z_intertick_fast=values[0], z_price_step_fast=values[1], z_intertick_slow=values[2],
        z_price_step_slow=values[3], cusum_intertick=values[4], cusum_price_step=values[5],
        gap_flag=0, warmup_flag=0, session_fallback_flag=0,
    )


def _two_days(n1=500, n2=50, seed=0):
    rng = np.random.default_rng(seed)
    day1 = [_vec(DAY1 + dt.timedelta(seconds=i), rng.normal(size=6)) for i in range(n1)]
    day2 = [_vec(DAY1 + dt.timedelta(days=1, seconds=i), rng.normal(size=6)) for i in range(n2)]
    return day1, day2


def test_window_ends_at_the_first_vector_of_the_next_day():
    w = TrainingWindow(training_days=1)
    assert not w.ends_before(DAY1)
    assert not w.ends_before(DAY1 + dt.timedelta(hours=8))
    assert w.ends_before(DAY1 + dt.timedelta(days=1))
    assert not w.ends_after(10**9)   # the count does not end a day-based window


def test_zscore_trains_on_the_whole_first_day_and_scores_from_the_next():
    day1, day2 = _two_days()
    det = ZScoreDetector({"training_days": 1})
    assert all(det.ingest_data(v) is None for v in day1)
    assert not det.is_trained
    first = det.ingest_data(day2[0])
    assert det.is_trained and first is not None          # the first vector of day 2 is scored
    m = np.array([[v.z_intertick_fast, v.z_price_step_fast, v.z_intertick_slow, v.z_price_step_slow,
                   v.cusum_intertick, v.cusum_price_step] for v in day1])
    assert np.allclose(det.feature_means, m.mean(axis=0))
    assert np.allclose(det.feature_stds, m.std(axis=0))  # population std, as before


def test_isoforest_trains_on_a_bounded_uniform_sample_of_the_day():
    day1, day2 = _two_days(n1=3000)
    det = IsolationForestDetector({"training_days": 1, "training_reservoir": 500, "n_estimators": 10})
    for v in day1:
        det.ingest_data(v)
    assert det.ingest_data(day2[0]) is not None
    assert det.sample_count == 3000


def test_reservoir_keeps_at_most_capacity_rows_and_covers_the_whole_stream():
    r = Reservoir(capacity=1000, width=1, seed=1)
    for i in range(100_000):
        r.add(np.array([i]))
    s = r.sample()[:, 0]
    assert len(s) == 1000
    # uniform over the stream: about a quarter of the sample from each quarter
    counts = np.histogram(s, bins=4, range=(0, 100_000))[0]
    assert all(200 < c < 300 for c in counts)


def test_sample_mode_is_unchanged():
    det = ZScoreDetector({"training_samples": 10})
    day1, _ = _two_days(n1=20)
    results = [det.ingest_data(v) for v in day1]
    assert all(r is None for r in results[:10]) and all(r is not None for r in results[10:])
