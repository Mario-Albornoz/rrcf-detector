"""Which features a model sees. Used by the ablation variants, which score the same recorded
vectors with a different feature set (see docs/Ablation_Plan.md at the repository root).

The six normalized features are the feed handler's output. The raw features are the
measurements they are computed from, before any normalization:
  intertick_ms    time since the instrument's previous message (0 when there is none, e.g.
                  the first message of a day: has_intertick = 0)
  price_step_bps  the absolute price step since the previous trade in basis points of that
                  trade's price, so steps of instruments at different price levels are in the
                  same unit (0 on messages without a price step, like the normalized price
                  features)
Recordings made before the raw fields were added carry them as 0: raw variants need a
recording of a run with feed-handler df5229b or later.
"""

from typing import Callable, Dict, List, Sequence

from src.kafka.consumer import NormalizedVectorDto

NORMALIZED = [
    "z_intertick_fast",
    "z_price_step_fast",
    "z_intertick_slow",
    "z_price_step_slow",
    "cusum_intertick",
    "cusum_price_step",
]


def _intertick_ms(v: NormalizedVectorDto) -> float:
    return float(v.intertick_ms) if v.has_intertick else 0.0


def _price_step_bps(v: NormalizedVectorDto) -> float:
    if not v.has_price_step or v.ref_price <= 0:
        return 0.0
    return 1e4 * v.price_step / v.ref_price


DERIVED: Dict[str, Callable[[NormalizedVectorDto], float]] = {
    "intertick_ms": _intertick_ms,
    "price_step_bps": _price_step_bps,
}

FEATURE_SETS: Dict[str, List[str]] = {
    "all": NORMALIZED,
    # C2 / C3: one timescale, or no cumulative sums
    "fast": ["z_intertick_fast", "z_price_step_fast", "cusum_intertick", "cusum_price_step"],
    "slow": ["z_intertick_slow", "z_price_step_slow", "cusum_intertick", "cusum_price_step"],
    "nocusum": ["z_intertick_fast", "z_price_step_fast", "z_intertick_slow", "z_price_step_slow"],
    # C4: the same two measurements, normalized (one timescale) or raw
    "z2": ["z_intertick_fast", "z_price_step_fast"],
    "raw2": ["intertick_ms", "price_step_bps"],
}


def resolve(features) -> List[str]:
    """A feature-set name or an explicit list of feature names."""
    names = FEATURE_SETS[features] if isinstance(features, str) else list(features)
    unknown = [n for n in names if n not in NORMALIZED and n not in DERIVED]
    if unknown:
        raise ValueError(f"unknown features {unknown}; known: {NORMALIZED + list(DERIVED)}")
    return names


def extract(v: NormalizedVectorDto, names: Sequence[str]) -> List[float]:
    return [DERIVED[n](v) if n in DERIVED else float(getattr(v, n)) for n in names]
