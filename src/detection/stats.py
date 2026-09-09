from dataclasses import dataclass

from numpy import sqrt


@dataclass
class Stats:
    mean: float
    std: float
    m2: float


def update_stats(state, score: float):
    """Update Welford statistics for TreeState."""
    state.score_count += 1
    delta = score - state.stats.mean
    state.stats.mean += delta / state.score_count

    delta2 = score - state.stats.mean
    state.stats.m2 += delta * delta2

    if state.score_count > 1:
        variance = state.stats.m2 / (state.score_count - 1)
        state.stats.std = sqrt(variance)
        state.z_score = (
            (score - state.stats.mean) / state.stats.std if state.stats.std > 0 else 0
        )
    else:
        state.z_score = 0

    return state
