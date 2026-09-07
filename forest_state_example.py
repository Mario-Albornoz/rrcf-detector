# Example: Correct ForestState Tracking

from collections import deque
from dataclasses import dataclass

import rrcf


@dataclass
class ForestState:
    """
    Metadata for one RRCF tree managing a sliding window.

    Tracks:
    - The RRCF tree itself
    - Window size limits
    - Index management for insert/evict
    - Insertion order via FIFO queue
    """

    tree: rrcf.RCTree
    max_size: int  # Window size (e.g., 1000)
    current_size: int  # Current points in tree (0 to max_size)
    next_index: int  # Next index to assign (monotonically increasing)
    oldest_index: int  # Index of oldest point (first inserted, first evicted)
    indices: deque  # FIFO queue tracking insertion order


class AnomalyDetector:
    def __init__(self, config: dict):
        self.config = config
        self.window_size = config.get("window_size", 1000)
        self.forest = {}  # tree_key -> ForestState

    def _create_tree(self, tree_key: str) -> ForestState:
        """Create new tree with metadata."""
        return ForestState(
            tree=rrcf.RCTree(),
            max_size=self.window_size,
            current_size=0,
            next_index=0,  # Start at 0 for this tree
            oldest_index=0,  # Will be set on first insert
            indices=deque(maxlen=self.window_size),  # Auto-evicts when full!
        )

    def insert_point(self, tree_key: str, point: list):
        """Insert point with proper index tracking."""
        state = self.forest[tree_key]

        # Get next index for THIS tree
        index = state.next_index

        # Check if window is full
        if state.current_size >= state.max_size:
            # Evict oldest before inserting new
            self.evict(tree_key)

        # Insert into RRCF tree
        state.tree.insert_point(point, index=index)

        # Update metadata
        state.indices.append(index)  # Add to FIFO queue
        state.current_size = len(state.indices)  # Current size
        state.next_index += 1  # Increment for next insert

        # Update oldest_index (first element in deque)
        if state.indices:
            state.oldest_index = state.indices[0]

        return index

    def evict(self, tree_key: str):
        """Evict oldest point from tree."""
        state = self.forest[tree_key]

        if state.current_size == 0:
            return  # Nothing to evict

        # Oldest index is at front of deque
        oldest_idx = state.indices[0]

        # Remove from RRCF tree
        state.tree.forget_point(oldest_idx)

        # Remove from deque (already done by maxlen, but explicit for clarity)
        # Note: If using maxlen, deque auto-evicts oldest on append
        # But if evicting before full, need to explicitly popleft
        state.indices.popleft()

        # Update size
        state.current_size = len(state.indices)

        # Update oldest_index
        if state.indices:
            state.oldest_index = state.indices[0]

    def get_window_state(self, tree_key: str) -> dict:
        """
        Get current window state for testing/debugging.

        CORRECT implementation using tracked metadata.
        """
        if tree_key not in self.forest:
            return {"error": "Tree not found", "tree_key": tree_key}

        state = self.forest[tree_key]

        return {
            "current_size": state.current_size,  # ✅ From metadata
            "max_size": state.max_size,  # ✅ From metadata
            "oldest_index": state.oldest_index,  # ✅ From metadata (deque[0])
            "newest_index": (
                state.indices[-1] if state.indices else None
            ),  # ✅ Last in deque
            "next_index": state.next_index,  # Next to be assigned
            "all_indices": list(state.indices),  # For debugging
        }


# ===========================================
# EXAMPLE USAGE
# ============================================


def example_usage():
    """Demonstrate how indices work per-tree."""

    detector = AnomalyDetector(config={"window_size": 5})

    # Create two separate trees
    tree_key_1 = "binance:crypto_spot"
    tree_key_2 = "coinbase:crypto_spot"

    detector.forest[tree_key_1] = detector._create_tree(tree_key_1)
    detector.forest[tree_key_2] = detector._create_tree(tree_key_2)

    print("=== Inserting into Tree 1 ===")
    for i in range(7):  # Insert 7 points into window of size 5
        point = [1.0 + i * 0.1, 2.0 + i * 0.1]
        idx = detector.insert_point(tree_key_1, point)
        state = detector.get_window_state(tree_key_1)
        print(f"Inserted index {idx}: {state}")

    print("\n=== Inserting into Tree 2 (separate indices!) ===")
    for i in range(3):  # Insert 3 points into second tree
        point = [5.0 + i * 0.1, 6.0 + i * 0.1]
        idx = detector.insert_point(tree_key_2, point)
        state = detector.get_window_state(tree_key_2)
        print(f"Inserted index {idx}: {state}")

    print("\n=== Final State ===")
    print("Tree 1:", detector.get_window_state(tree_key_1))
    print("Tree 2:", detector.get_window_state(tree_key_2))


if __name__ == "__main__":
    example_usage()


# ============================================
# EXPECTED OUTPUT
# ============================================
"""
=== Inserting into Tree 1 ===
Inserted index 0: {'current_size': 1, 'max_size': 5, 'oldest_index': 0, 'newest_index': 0, ...}
Inserted index 1: {'current_size': 2, 'max_size': 5, 'oldest_index': 0, 'newest_index': 1, ...}
Inserted index 2: {'current_size': 3, 'max_size': 5, 'oldest_index': 0, 'newest_index': 2, ...}
Inserted index 3: {'current_size': 4, 'max_size': 5, 'oldest_index': 0, 'newest_index': 3, ...}
Inserted index 4: {'current_size': 5, 'max_size': 5, 'oldest_index': 0, 'newest_index': 4, ...}
Inserted index 5: {'current_size': 5, 'max_size': 5, 'oldest_index': 1, 'newest_index': 5, ...}  ← Evicted 0!
Inserted index 6: {'current_size': 5, 'max_size': 5, 'oldest_index': 2, 'newest_index': 6, ...}  ← Evicted 1!

=== Inserting into Tree 2 (separate indices!) ===
Inserted index 0: {'current_size': 1, 'max_size': 5, 'oldest_index': 0, 'newest_index': 0, ...}  ← Starts at 0!
Inserted index 1: {'current_size': 2, 'max_size': 5, 'oldest_index': 0, 'newest_index': 1, ...}
Inserted index 2: {'current_size': 3, 'max_size': 5, 'oldest_index': 0, 'newest_index': 2, ...}

=== Final State ===
Tree 1: {'current_size': 5, 'oldest_index': 2, 'newest_index': 6, 'all_indices': [2, 3, 4, 5, 6]}
Tree 2: {'current_size': 3, 'oldest_index': 0, 'newest_index': 2, 'all_indices': [0, 1, 2]}
"""
