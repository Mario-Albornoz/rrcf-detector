from collections import deque
from dataclasses import dataclass

import rrcf

from src.kafka.consumer import NormalizedVectorDto

from .utils import get_instrument_key


@dataclass
class AnomalyDetectorConfig:
    window_size: int
    min_fill_threshold: int
    normal_threshold: float = 5.0
    medium_threshold: float = 10.0


@dataclass
class TreeState:
    tree: rrcf.RCTree
    max_size: int
    current_size: int
    next_index: int
    oldest_index: int
    indices: deque
    is_warm: bool
    min_fill_threshold: int


class AnomalyDetector:
    def __init__(self, config: dict):
        self.config = config
        self.forest = {}
        
        self.normal_threshold = config.get("normal_threshold", 5.0)
        self.medium_threshold = config.get("medium_threshold", 10.0)

    def ingest_data(self, data: NormalizedVectorDto):
        self._create_tree_if_absent(data)

        feature_point = [
            data.z_intertick_fast,
            data.z_price_step_fast,
            data.z_intertick_slow,
            data.z_price_step_slow,
            data.cusum_intertick,
            data.cusum_price_step,
        ]

        tree_state: TreeState = self.forest[get_instrument_key(data)]
        index = tree_state.next_index
        self._insert_point(get_instrument_key(data), feature_point, index=index)

        if not tree_state.is_warm:
            return None

        score = self.scoreCoDisp(get_instrument_key(data), index=index)
        return score

    def evict(self, tree_key: str):
        tree_state: TreeState = self.forest[tree_key]

        if tree_state.current_size == 0:
            return

        tree_state.tree.forget_point(tree_state.oldest_index)
        tree_state.indices.popleft()
        tree_state.oldest_index = tree_state.indices[0] if tree_state.indices else 0

    def _insert_point(self, tree_key: str, point: list, index: int):
        tree_state: TreeState = self.forest[tree_key]

        if tree_state.current_size >= tree_state.max_size:
            self.evict(tree_key)

        tree_state.tree.insert_point(point, index=index)

        tree_state.indices.append(index)
        tree_state.current_size = len(tree_state.indices)
        tree_state.next_index += 1
        tree_state.oldest_index = tree_state.indices[0] if tree_state.indices else 0

        if tree_state.current_size >= tree_state.min_fill_threshold:
            tree_state.is_warm = True

    def scoreCoDisp(self, tree_key: str, index: int) -> float:
        if not self._has_tree(tree_key):
            return 0
        tree_state: TreeState = self.forest[tree_key]
        return tree_state.tree.codisp(index)

    def _create_tree_if_absent(self, data: NormalizedVectorDto):
        if not self._has_tree(get_instrument_key(data)):
            tree = TreeState(
                tree=rrcf.RCTree(),
                max_size=self.config["window_size"],
                current_size=0,
                next_index=0,
                oldest_index=0,
                indices=deque(maxlen=self.config["window_size"]),
                is_warm=False,
                min_fill_threshold=self.config["min_fill_threshold"],
            )
            self.forest[get_instrument_key(data)] = tree

    def _has_tree(self, tree_key: str) -> bool:
        return tree_key in self.forest

    def get_window_state(self, tree_key: str) -> dict:
        if tree_key not in self.forest:
            return {"error": "Tree not found", "tree_key": tree_key}

        tree_state: TreeState = self.forest[tree_key]

        return {
            "current_size": tree_state.current_size,
            "max_size": tree_state.max_size,
            "oldest_index": tree_state.indices[0] if tree_state.indices else 0,
            "newest_index": (tree_state.indices[-1] if tree_state.indices else None),
            "all_indices": list(tree_state.indices),
        }

    def get_tree_count(self) -> int:
        return len(self.forest)

    def determine_alert_level(self, score: float) -> str:
        if score < self.normal_threshold:
            return "normal"
        elif score < self.medium_threshold:
            return "medium"
        else:
            return "high"
