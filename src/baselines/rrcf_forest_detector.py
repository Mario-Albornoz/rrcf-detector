from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import rrcf
from src.baselines.base_detector import BaseDetector
from src.baselines.features import extract, resolve
from src.detection.stats import Stats, update_stats
from src.detection.utils import get_instrument_key
from src.kafka.consumer import NormalizedVectorDto


@dataclass
class ForestState:
    trees: List[rrcf.RCTree]
    indices: deque
    next_index: int = 0
    is_warm: bool = False
    score_count: int = 0
    z_score: float = 0.00
    stats: Stats = field(default_factory=lambda: Stats(0, 0, 0))


class RRCFForestDetector(BaseDetector):
    def __init__(self, config: dict):
        self.window_size = config.get("window_size", 256)
        self.num_trees = config.get("num_trees", 10)
        self.min_fill_threshold = config.get("min_fill_threshold", 25)
        self.seed = config.get("seed", 42)
        self.forests: Dict[str, ForestState] = {}
        # Ablation options (defaults = the evaluated model).
        self.name = config.get("name", "rrcf_forest")
        self.features = resolve(config.get("features", "all"))
        # "exchange_class": one forest per exchange and class (shared by its instruments);
        # "instrument": one forest per instrument.
        self.key_by = config.get("key_by", "exchange_class")
        if self.key_by not in ("exchange_class", "instrument"):
            raise ValueError(f"key_by must be exchange_class or instrument, got {self.key_by}")
        # A cold forest (window below min_fill_threshold) normally emits nothing. Per-
        # instrument forests stay cold on quiet instruments; if they emitted nothing, those
        # vectors would drop out of the evaluation's scorable set and recall would be
        # computed on an easier denominator. With cold_score set, a cold forest emits that
        # (non-alerting) score instead, so every variant scores the same vectors.
        self.cold_score = config.get("cold_score")

    def _forest(self, key: str) -> ForestState:
        state = self.forests.get(key)
        if state is None:
            base = self.seed + 1000 * len(self.forests)
            state = ForestState(
                trees=[
                    rrcf.RCTree(random_state=base + i) for i in range(self.num_trees)
                ],
                indices=deque(),
            )
            self.forests[key] = state

        return state

    def _key(self, data: NormalizedVectorDto) -> str:
        if self.key_by == "instrument":
            return f"{data.exchange}:{data.instrument}"
        return get_instrument_key(data)

    def ingest_data(self, data: NormalizedVectorDto) -> Optional[Dict]:
        state = self._forest(self._key(data))

        point = extract(data, self.features)

        if len(state.indices) >= self.window_size:
            oldest = state.indices.popleft()
            for tree in state.trees:
                tree.forget_point(oldest)

        index = state.next_index
        for tree in state.trees:
            tree.insert_point(point, index=index)
        state.indices.append(index)
        state.next_index += 1

        if len(state.indices) >= self.min_fill_threshold:
            state.is_warm = True
        if not state.is_warm:
            if self.cold_score is None:
                return None
            return {
                "raw_score": float(self.cold_score),
                "z_score": 0.0,
                "stats": {"mean": 0.0, "std": 0.0, "count": 0},
            }

        raw_score = sum(tree.codisp(index) for tree in state.trees) / self.num_trees
        update_stats(state, raw_score)

        return {
            "raw_score": raw_score,
            "z_score": state.z_score,
            "stats": {
                "mean": state.stats.mean,
                "std": state.stats.std,
                "count": state.score_count,
            },
        }

    def get_model_name(self) -> str:
        return self.name
