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
        """
        TODO (DETECT-1):
        - Extract window_size from config (e.g., 1000 points)
        - Initialize point_index counter (needed for RRCF insert/delete)
        - RECOMMENDED: Use ForestState dataclass approach:

        self.forest = {}  # tree_key -> ForestState

        WHY indices per-tree, not global?
        - Each tree is independent
        - tree.codisp(index) looks in THAT tree only
        - Easier to track and debug

        WHY deque for indices?
        - Auto-evicts oldest when at maxlen (append automatically drops first)
        - Always: oldest = indices[0], newest = indices[-1]
        - O(1) append/popleft operations

        PERFORMANCE NOTE:
        - self.forest will be accessed on EVERY message (hot path)
        - Consider using __slots__ if memory becomes an issue
        - Window size: larger = more memory but better gradual anomaly detection
        """
        self.config = config
        self.forest = {}  # Key: (exchange, instrument_class), Value: tree + metadata
        pass

    def ingest_data(self, data: NormalizedVectorDto):
        """
        Main entry point for processing a vector from Kafka consumer.

        ✅ GOOD: Dynamic tree creation with _create_tree_if_absent()
        ✅ GOOD: Feature extraction to 6D point

        TODO (DETECT-1):
        - Implement eviction logic (check if window full before insert)
        - Call scoreCoDisp after insertion
        - Return the score
        - Update TreeState metadata after insert

        POTENTIAL ISSUE:
        - get_current_index() might fail if tree just created (no indices yet)
        - Consider using tree_state.next_index directly instead

        PERFORMANCE CRITICAL:
        - This is called for EVERY Kafka message (3k-100k/sec)
        - Minimize allocations and copies
        - Feature extraction is inline ✅
        """
        # ✅ Create tree on-demand if new key
        self._create_tree_if_absent(data)

        # ✅ Extract 6D feature vector
        feature_point = [
            data.z_intertick_fast,
            data.z_price_step_fast,
            data.z_intertick_slow,
            data.z_price_step_slow,
            data.cusum_intertick,
            data.cusum_price_step,
        ]

        # TODO: This might fail on first insert - tree just created, no indices yet
        # Consider: tree_state = self.forest[get_instrument_key(data)]
        #           index = tree_state.next_index
        self.insert_point(
            get_instrument_key(data),
            feature_point,
            index=self.get_current_index(get_instrument_key(data)) + 1,
        )
        pass

    def evict(self, tree_key: str):
        """
        Evict oldest point from the specified tree when window is full.

        TODO (DETECT-1):
        - Get oldest_index from tree_state.indices[0]
        - Call tree_state.tree.forget_point(oldest_index)
        - Update metadata:
          * tree_state.indices.popleft() (remove oldest)
          * tree_state.current_size = len(tree_state.indices)
          * tree_state.oldest_index = tree_state.indices[0] if tree_state.indices else 0
        - Handle edge case: what if tree is empty?

        PERFORMANCE NOTE:
        - RRCF forget_point is O(log n) but can be costly at high throughput
        - This is called on EVERY insert when window is full
        - Must be efficient to maintain 100k msg/sec target

        TESTING (DETECT-1 acceptance criteria):
        - Unit test must verify window size never exceeds configured max
        - No memory growth over repeated insert/evict cycles
        """
        tree_state: TreeState = self.forest[tree_key]

        if tree_state.current_size == 0:
            return

        tree_state.tree.forget_point(tree_state.oldest_index)
        tree_state.indices.popleft()
        tree_state.oldest_index = tree_state.indices[0]

        pass

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

        tree: rrcf.RCTree = self.forest[tree_key]
        return tree.codisp(index)

    def _create_tree_if_absent(self, data: NormalizedVectorDto):
        if not self._has_tree(data):
            tree = TreeState(
                tree=rrcf.RCTree(),
                max_size=self.config["window_size"],
                current_size=0,
                next_index=0,
                oldest_index=0,
                indices=deque(maxlen=self.config["window_size"]),
            )
            self.forest[get_instrument_key(data)] = tree

    def _has_tree(self, data: NormalizedVectorDto) -> bool:
        if get_instrument_key(data) not in self.forest:
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
            "oldest_index": tree_state.indices[0],
            "newest_index": (tree_state.indices[-1] if tree_state.indices else None),
            "all_indices": list(tree_state.indices),
        }

        return response

    def get_current_index(self, tree_key):
        return self.get_window_state(tree_key)["newest_index"] + 1

    def get_tree_count(self) -> int:
        return len(self.forest)
