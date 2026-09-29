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
    "fast": ["z_intertick_fast", "z_price_step_fast", "cusum_intertick", "cusum_price_step"],
    "slow": ["z_intertick_slow", "z_price_step_slow", "cusum_intertick", "cusum_price_step"],
    "nocusum": ["z_intertick_fast", "z_price_step_fast", "z_intertick_slow", "z_price_step_slow"],
    "z2": ["z_intertick_fast", "z_price_step_fast"],
    "raw2": ["intertick_ms", "price_step_bps"],
}


def resolve(features) -> List[str]:
    names = FEATURE_SETS[features] if isinstance(features, str) else list(features)
    unknown = [n for n in names if n not in NORMALIZED and n not in DERIVED]
    if unknown:
        raise ValueError(f"unknown features {unknown}; known: {NORMALIZED + list(DERIVED)}")
    return names


def extract(v: NormalizedVectorDto, names: Sequence[str]) -> List[float]:
    return [DERIVED[n](v) if n in DERIVED else float(getattr(v, n)) for n in names]
