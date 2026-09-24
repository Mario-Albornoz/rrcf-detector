"""Ablation variants: rrcf_forest and zscore with one property changed (docs/Ablation_Plan.md)."""

import datetime as dt
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from src.baselines import RRCFForestDetector, ZScoreDetector
from src.baselines.features import FEATURE_SETS, NORMALIZED, extract, resolve
from src.config import ServiceConfig
from src.kafka.consumer import NormalizedVectorDto

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_multi_model.py"
_spec = importlib.util.spec_from_file_location("run_multi_model", _SCRIPT)
rmm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rmm)

T0 = dt.datetime(2021, 11, 8, 10, 0, tzinfo=dt.timezone.utc)


def _vec(k, instrument="A.ETR", values=None, **raw):
    rng = np.random.default_rng(k)
    f = values if values is not None else rng.normal(size=6)
    return NormalizedVectorDto(
        exchange="ETR", instrument=instrument, instrument_class="E", timestamp=T0 + dt.timedelta(seconds=k),
        model_key="E", z_intertick_fast=f[0], z_price_step_fast=f[1], z_intertick_slow=f[2],
        z_price_step_slow=f[3], cusum_intertick=f[4], cusum_price_step=f[5], gap_flag=0,
        warmup_flag=0, session_fallback_flag=0, seq=k + 1, **raw,
    )


def test_every_variant_writes_its_own_scores_file():
    reg = rmm.model_registry(ServiceConfig().partitioner_config.detector_config)
    for name, (cls, config) in reg.items():
        if cls in (RRCFForestDetector, ZScoreDetector):
            assert cls(config).get_model_name() == name


def test_variants_differ_from_the_reference_only_in_the_ablated_property():
    reg = rmm.model_registry(ServiceConfig().partitioner_config.detector_config)
    ref = reg["rrcf_forest"][1]
    for name, (cls, config) in reg.items():
        if name.startswith("rrcf_forest_"):
            extra = {k: v for k, v in config.items() if ref.get(k) != v}
            assert set(extra) <= {"name", "features", "key_by", "cold_score"}, (name, extra)


def test_default_features_are_the_six_normalized_ones():
    v = _vec(0)
    assert extract(v, resolve("all")) == [getattr(v, n) for n in NORMALIZED]


def test_raw_features_are_derived_from_the_recorded_measurements():
    v = _vec(0, intertick_ms=2000.0, has_intertick=1, price_step=0.5, has_price_step=1, ref_price=100.0)
    assert extract(v, resolve("raw2")) == [2000.0, 50.0]   # 0.5 on 100 = 50 bp
    first_of_day = _vec(1, intertick_ms=0.0, has_intertick=0, price_step=0.0, has_price_step=0, ref_price=0.0)
    assert extract(first_of_day, resolve("raw2")) == [0.0, 0.0]


def test_unknown_feature_is_rejected():
    with pytest.raises(ValueError, match="unknown features"):
        resolve(["z_intertick_fast", "no_such_feature"])


def test_feature_sets_have_the_intended_members():
    assert FEATURE_SETS["z2"] == ["z_intertick_fast", "z_price_step_fast"]
    assert all("cusum" not in f for f in FEATURE_SETS["nocusum"])
    assert all("slow" not in f for f in FEATURE_SETS["fast"])
    assert all("fast" not in f for f in FEATURE_SETS["slow"])


def test_shared_forest_is_one_per_exchange_and_instrument_forest_one_per_instrument():
    shared = RRCFForestDetector({"num_trees": 2, "min_fill_threshold": 5})
    per_inst = RRCFForestDetector({"num_trees": 2, "min_fill_threshold": 5, "key_by": "instrument"})
    for k in range(40):
        v = _vec(k, instrument=f"I{k % 4}.ETR")
        shared.ingest_data(v)
        per_inst.ingest_data(v)
    assert len(shared.forests) == 1 and len(per_inst.forests) == 4


def test_cold_forest_emits_the_cold_score_so_every_vector_stays_scorable():
    silent = RRCFForestDetector({"num_trees": 2, "min_fill_threshold": 5})
    emitting = RRCFForestDetector({"num_trees": 2, "min_fill_threshold": 5, "cold_score": 0.0})
    r_silent = [silent.ingest_data(_vec(k)) for k in range(8)]
    r_emit = [emitting.ingest_data(_vec(k)) for k in range(8)]
    assert r_silent[:4] == [None] * 4                               # cold: nothing
    assert all(r["raw_score"] == 0.0 for r in r_emit[:4])           # cold: a non-alerting 0
    assert r_silent[4:] and all(r is not None for r in r_emit)      # warm: scored as before
    assert [r["raw_score"] for r in r_silent[4:]] == [r["raw_score"] for r in r_emit[4:]]


def test_forest_on_a_feature_subset_inserts_points_of_that_width():
    det = RRCFForestDetector({"num_trees": 1, "min_fill_threshold": 3, "features": "z2"})
    for k in range(5):
        det.ingest_data(_vec(k))
    tree = next(iter(det.forests.values())).trees[0]
    assert tree.ndim == 2


def test_zscore_on_a_feature_subset():
    det = ZScoreDetector({"training_samples": 50, "features": "raw2"})
    for k in range(60):
        det.ingest_data(_vec(k, intertick_ms=1000.0 * (1 + k % 3), has_intertick=1, price_step=0.01 * (k % 2),
                             has_price_step=1, ref_price=10.0))
    assert det.is_trained and len(det.feature_means) == 2
    assert det.get_model_name() == "zscore"
