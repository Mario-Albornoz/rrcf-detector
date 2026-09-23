"""Half-Space Trees must be told the range of our features (River assumes [0, 1])."""

import random

from src.baselines.halfspace_trees_detector import DEFAULT_LIMITS, HalfSpaceTreesDetector
from src.kafka.consumer import NormalizedVectorDto

FEATURES = list(DEFAULT_LIMITS)


def _vector(values):
    v = NormalizedVectorDto.__new__(NormalizedVectorDto)
    for name, value in zip(FEATURES, values):
        setattr(v, name, value)
    return v


def _scores(detector, n, rng):
    out = []
    for _ in range(n):
        z = [rng.gauss(0, 1) for _ in range(4)]
        c = [abs(rng.gauss(0, 3)) for _ in range(2)]
        r = detector.ingest_data(_vector(z + c))
        if r is not None:
            out.append(r["raw_score"])
    return out


def test_limits_are_passed_to_river():
    det = HalfSpaceTreesDetector({"window_size": 256, "min_fill_threshold": 25})
    for name, (lo, hi) in DEFAULT_LIMITS.items():
        assert det.model.limits[name] == (lo, hi)


def test_limits_can_be_overridden():
    det = HalfSpaceTreesDetector({"limits": {"cusum_intertick": (0.0, 100.0)}})
    assert det.model.limits["cusum_intertick"] == (0.0, 100.0)
    assert det.model.limits["z_intertick_fast"] == DEFAULT_LIMITS["z_intertick_fast"]


def test_normal_traffic_is_not_squeezed_against_the_ceiling_and_outliers_stand_out():
    rng = random.Random(0)
    det = HalfSpaceTreesDetector({"window_size": 256, "min_fill_threshold": 25})
    normal = sorted(_scores(det, 3000, rng)[1000:])
    median, p99 = normal[len(normal) // 2], normal[int(0.99 * len(normal))]
    # without limits the median was ~0.98 on z-score-scale inputs
    assert median < 0.6
    outlier = det.ingest_data(_vector([0.0, 8.0, 0.0, 8.0, 0.0, 40.0]))["raw_score"]
    assert outlier > p99
