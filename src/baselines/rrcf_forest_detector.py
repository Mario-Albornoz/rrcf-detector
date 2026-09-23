from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import rrcf
from src.baselines.base_detector import BaseDetector
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

    def ingest_data(self, data: NormalizedVectorDto) -> Optional[Dict]:
        state = self._forest(get_instrument_key(data))

        point = [
            data.z_intertick_fast,
            data.z_price_step_fast,
            data.z_intertick_slow,
            data.z_price_step_slow,
            data.cusum_intertick,
            data.cusum_price_step,
        ]

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
            return None

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
        return "rrcf_forest"
