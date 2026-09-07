from collections import deque
from dataclasses import dataclass

import rrcf

from src.kafka.consumer import NormalizedVectorDto

from .utils import get_instrument_key


@dataclass
class TreeState:
    tree: rrcf.RCTree
    max_size: int
    current_size: int
    next_index: int
    oldest_index: int
    indices: deque


class AnomalyDetector:
    def __init__(self, config: dict):
        self.config = config
        self.forest = {}

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

        tree_state = self.forest[get_instrument_key(data)]
        index = tree_state.next_index
        self.insert_point(get_instrument_key(data), feature_point, index=index)
        score = self.scoreCoDisp(get_instrument_key(data), index=index)

        return score

    def evict(self, tree_key: str):
        tree_state: TreeState = self.forest[tree_key]

        if tree_state.current_size == 0:
            return

        tree_state.tree.forget_point(tree_state.oldest_index)
        tree_state.indices.popleft()
        tree_state.oldest_index = tree_state.indices[0] if tree_state.indices else 0

    def insert_point(self, tree_key: str, point: list, index: int):

        tree_state: TreeState = self.forest[tree_key]

        if tree_state.current_size >= tree_state.max_size:
            self.evict(tree_key)

        tree_state.tree.insert_point(point, index=index)

        tree_state.indices.append(index)
        tree_state.current_size = len(tree_state.indices)
        tree_state.next_index += 1
        tree_state.oldest_index = tree_state.indices[0] if tree_state.indices else 0

        pass

    def scoreCoDisp(self, tree_key: str, index: int) -> float:
        """
        - Return score (calibration happens in DETECT-2, not here)

        PARAMETERS:
        - tree_key: (exchange, instrument_class) identifier
        - index: Index of point to score (just inserted)

        RETURNS:
        - float: Raw CoDisp score (higher = more anomalous)

        PERFORMANCE CRITICAL:
        - CoDisp computation is expensive (tree traversal)
        - This is called after EVERY insert (3k-100k/sec)
        - Consider: batch scoring if possible? Or async?

        NOTE: Raw CoDisp scores are NOT comparable across instrument classes.
        DETECT-2 will add calibration (rolling mean/stddev per class).
        """

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
            )
            self.forest[get_instrument_key(data)] = tree

    def _has_tree(self, tree_key: str) -> bool:
        if tree_key not in self.forest:
            return False
        else:
            return True

    def get_window_state(self, tree_key: str) -> dict:
        if tree_key not in self.forest:
            return {"error": "Tree not found", "tree_key": tree_key}

        tree_state: TreeState = self.forest[tree_key]

        response = {
            "current_size": tree_state.current_size,
            "max_size": tree_state.max_size,
            "oldest_index": tree_state.indices[0] if tree_state.indices else 0,
            "newest_index": (tree_state.indices[-1] if tree_state.indices else None),
            "all_indices": list(tree_state.indices),
        }

        return response

    def get_tree_count(self) -> int:
        return len(self.forest)
