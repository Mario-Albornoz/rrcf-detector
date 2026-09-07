"""
Unit tests for DETECT-1: RRCF wrapper with windowed insert/evict.

Acceptance Criteria:
1. Window size is respected (oldest point evicted on overflow)
2. Insert/score round-trip works on known synthetic sequences
3. No memory growth beyond window bound over repeated inserts
4. Dynamic tree creation works correctly for new keys arriving mid-stream
"""

from collections import deque
from datetime import datetime

import pytest

from src.detection.AnomalyDetector import AnomalyDetector, TreeState
from src.kafka.consumer import NormalizedVectorDto


@pytest.fixture
def config():
    """Test configuration with small window for fast tests."""
    return {"window_size": 5, "min_fill_threshold": 1}


@pytest.fixture
def detector(config):
    """Create a fresh detector for each test."""
    return AnomalyDetector(config)


@pytest.fixture
def sample_vector():
    """Create a sample normalized vector."""
    return NormalizedVectorDto(
        exchange="binance",
        instrument="BTC-USDT",
        instrument_class="crypto_spot",
        timestamp=datetime.now(),
        model_key="test",
        z_intertick_fast=1.5,
        z_price_step_fast=0.3,
        z_intertick_slow=1.2,
        z_price_step_slow=0.1,
        cusum_intertick=2.1,
        cusum_price_step=0.8,
        gap_flag=0,
        warmup_flag=0,
        session_fallback_flag=0,
    )


class TestTreeCreation:
    """Test dynamic tree creation."""

    def test_tree_created_on_first_vector(self, detector, sample_vector):
        """Tree should be created when first vector arrives for a new key."""
        assert detector.get_tree_count() == 0

        detector.ingest_data(sample_vector)

        assert detector.get_tree_count() == 1
        assert "binance:crypto_spot" in detector.forest

    def test_tree_not_recreated_on_subsequent_vectors(self, detector, sample_vector):
        """Same key should reuse existing tree."""
        detector.ingest_data(sample_vector)
        tree_id_1 = id(detector.forest["binance:crypto_spot"])

        detector.ingest_data(sample_vector)
        tree_id_2 = id(detector.forest["binance:crypto_spot"])

        assert tree_id_1 == tree_id_2
        assert detector.get_tree_count() == 1

    def test_multiple_keys_create_separate_trees(self, detector):
        """Different (exchange, class) keys should create separate trees."""
        vector1 = NormalizedVectorDto(
            exchange="binance",
            instrument="BTC-USDT",
            instrument_class="crypto_spot",
            timestamp=datetime.now(),
            model_key="test",
            z_intertick_fast=1.0,
            z_price_step_fast=1.0,
            z_intertick_slow=1.0,
            z_price_step_slow=1.0,
            cusum_intertick=1.0,
            cusum_price_step=1.0,
            gap_flag=0,
            warmup_flag=0,
            session_fallback_flag=0,
        )

        vector2 = NormalizedVectorDto(
            exchange="coinbase",
            instrument="BTC-USD",
            instrument_class="crypto_spot",
            timestamp=datetime.now(),
            model_key="test",
            z_intertick_fast=1.0,
            z_price_step_fast=1.0,
            z_intertick_slow=1.0,
            z_price_step_slow=1.0,
            cusum_intertick=1.0,
            cusum_price_step=1.0,
            gap_flag=0,
            warmup_flag=0,
            session_fallback_flag=0,
        )

        vector3 = NormalizedVectorDto(
            exchange="binance",
            instrument="BTC-PERP",
            instrument_class="crypto_futures",
            timestamp=datetime.now(),
            model_key="test",
            z_intertick_fast=1.0,
            z_price_step_fast=1.0,
            z_intertick_slow=1.0,
            z_price_step_slow=1.0,
            cusum_intertick=1.0,
            cusum_price_step=1.0,
            gap_flag=0,
            warmup_flag=0,
            session_fallback_flag=0,
        )

        detector.ingest_data(vector1)
        detector.ingest_data(vector2)
        detector.ingest_data(vector3)

        assert detector.get_tree_count() == 3
        assert "binance:crypto_spot" in detector.forest
        assert "coinbase:crypto_spot" in detector.forest
        assert "binance:crypto_futures" in detector.forest


class TestWindowSize:
    """Test that window size constraints are respected."""

    def test_window_size_never_exceeded(self, detector, sample_vector):
        """Window should never exceed max_size even after many inserts."""
        window_size = detector.config["window_size"]

        # Insert 3x window_size
        for i in range(window_size * 3):
            detector.ingest_data(sample_vector)

        state = detector.get_window_state("binance:crypto_spot")
        assert state["current_size"] == window_size
        assert state["current_size"] <= state["max_size"]

    def test_window_fills_gradually(self, detector, sample_vector):
        """Window should fill up to max_size before eviction starts."""
        for i in range(3):
            detector.ingest_data(sample_vector)
            state = detector.get_window_state("binance:crypto_spot")
            assert state["current_size"] == i + 1

    def test_oldest_point_evicted_on_overflow(self, detector):
        """When window is full, oldest point should be evicted on new insert."""
        window_size = detector.config["window_size"]

        # Create vectors with different values to track them
        for i in range(window_size + 2):
            vector = NormalizedVectorDto(
                exchange="binance",
                instrument="BTC-USDT",
                instrument_class="crypto_spot",
                timestamp=datetime.now(),
                model_key="test",
                z_intertick_fast=float(i),
                z_price_step_fast=float(i),
                z_intertick_slow=float(i),
                z_price_step_slow=float(i),
                cusum_intertick=float(i),
                cusum_price_step=float(i),
                gap_flag=0,
                warmup_flag=0,
                session_fallback_flag=0,
            )
            detector.ingest_data(vector)

        state = detector.get_window_state("binance:crypto_spot")

        # After inserting 7 points into window of 5:
        # Indices should be [2, 3, 4, 5, 6] (0 and 1 evicted)
        assert state["current_size"] == window_size
        assert state["oldest_index"] == 2  # 0 and 1 were evicted
        assert state["newest_index"] == 6
        assert state["all_indices"] == [2, 3, 4, 5, 6]


class TestInsertScoreRoundtrip:
    """Test insert and score operations work correctly."""

    def test_score_returned_after_insert(self, detector, sample_vector):
        """ingest_data should return a score."""
        score = detector.ingest_data(sample_vector)
        
        assert isinstance(score, (int, float))  # Accept both int and float
        assert score >= 0  # CoDisp scores are non-negative

    def test_normal_points_have_similar_scores(self, detector):
        """Similar points should have similar (low) anomaly scores."""
        scores = []

        for i in range(5):
            vector = NormalizedVectorDto(
                exchange="binance",
                instrument="BTC-USDT",
                instrument_class="crypto_spot",
                timestamp=datetime.now(),
                model_key="test",
                z_intertick_fast=1.0 + i * 0.1,
                z_price_step_fast=0.5 + i * 0.1,
                z_intertick_slow=1.0 + i * 0.1,
                z_price_step_slow=0.5 + i * 0.1,
                cusum_intertick=2.0,
                cusum_price_step=1.0,
                gap_flag=0,
                warmup_flag=0,
                session_fallback_flag=0,
            )
            score = detector.ingest_data(vector)
            scores.append(score)

        # All scores should be relatively low and similar
        # (exact values depend on RRCF internals, but should be < 10 for normal points)
        assert all(score < 10 for score in scores)

    def test_anomalous_point_has_higher_score(self, detector):
        """An outlier point should have a significantly higher score."""
        # Insert 5 normal points
        for i in range(5):
            normal_vector = NormalizedVectorDto(
                exchange="binance",
                instrument="BTC-USDT",
                instrument_class="crypto_spot",
                timestamp=datetime.now(),
                model_key="test",
                z_intertick_fast=1.0,
                z_price_step_fast=0.5,
                z_intertick_slow=1.0,
                z_price_step_slow=0.5,
                cusum_intertick=2.0,
                cusum_price_step=1.0,
                gap_flag=0,
                warmup_flag=0,
                session_fallback_flag=0,
            )
            normal_score = detector.ingest_data(normal_vector)

        # Insert anomalous point (far from normal cluster)
        anomaly_vector = NormalizedVectorDto(
            exchange="binance",
            instrument="BTC-USDT",
            instrument_class="crypto_spot",
            timestamp=datetime.now(),
            model_key="test",
            z_intertick_fast=100.0,
            z_price_step_fast=100.0,  # Extreme values
            z_intertick_slow=100.0,
            z_price_step_slow=100.0,
            cusum_intertick=100.0,
            cusum_price_step=100.0,
            gap_flag=0,
            warmup_flag=0,
            session_fallback_flag=0,
        )
        anomaly_score = detector.ingest_data(anomaly_vector)

        # Anomaly should have higher score than normal points
        assert anomaly_score > normal_score


class TestMemoryManagement:
    """Test that memory doesn't grow beyond window bounds."""

    def test_no_memory_growth_over_repeated_inserts(self, detector, sample_vector):
        """Indices deque should not grow beyond max_size."""
        window_size = detector.config["window_size"]

        # Insert many points (100x window size)
        for i in range(window_size * 100):
            detector.ingest_data(sample_vector)

        state = detector.get_window_state("binance:crypto_spot")
        tree_state = detector.forest["binance:crypto_spot"]

        # Memory should be bounded
        assert state["current_size"] == window_size
        assert len(tree_state.indices) == window_size
        assert tree_state.indices.maxlen == window_size

    def test_indices_deque_has_correct_maxlen(self, detector, sample_vector):
        """Deque should be created with correct maxlen."""
        detector.ingest_data(sample_vector)

        tree_state = detector.forest["binance:crypto_spot"]
        assert tree_state.indices.maxlen == detector.config["window_size"]


class TestWindowState:
    """Test window state introspection."""

    def test_window_state_tracks_size_correctly(self, detector, sample_vector):
        """Window state should accurately reflect current size."""
        state_0 = detector.get_window_state("binance:crypto_spot")
        assert state_0["error"] == "Tree not found"

        detector.ingest_data(sample_vector)
        state_1 = detector.get_window_state("binance:crypto_spot")
        assert state_1["current_size"] == 1

        detector.ingest_data(sample_vector)
        state_2 = detector.get_window_state("binance:crypto_spot")
        assert state_2["current_size"] == 2

    def test_window_state_tracks_indices(self, detector, sample_vector):
        """Window state should track oldest and newest indices."""
        for i in range(3):
            detector.ingest_data(sample_vector)

        state = detector.get_window_state("binance:crypto_spot")
        assert state["oldest_index"] == 0
        assert state["newest_index"] == 2
        assert state["all_indices"] == [0, 1, 2]

    def test_window_state_after_eviction(self, detector, sample_vector):
        """Window state should update correctly after evictions."""
        window_size = detector.config["window_size"]

        # Fill window + 2 more (cause 2 evictions)
        for i in range(window_size + 2):
            detector.ingest_data(sample_vector)

        state = detector.get_window_state("binance:crypto_spot")
        assert state["oldest_index"] == 2  # 0 and 1 evicted
        assert state["newest_index"] == 6  # Last inserted
        assert len(state["all_indices"]) == window_size


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_tree_state(self, detector):
        """Getting state of non-existent tree should return error."""
        state = detector.get_window_state("nonexistent:key")
        assert "error" in state
        assert state["error"] == "Tree not found"

    def test_score_on_empty_tree(self, detector):
        """Scoring on non-existent tree should return 0."""
        score = detector.scoreCoDisp("nonexistent:key", 0)
        assert score == 0

    def test_first_insert_has_index_zero(self, detector, sample_vector):
        """First point should have index 0."""
        detector.ingest_data(sample_vector)

        state = detector.get_window_state("binance:crypto_spot")
        assert state["oldest_index"] == 0
        assert state["newest_index"] == 0
        assert state["all_indices"] == [0]


class TestMetadataConsistency:
    """Test that metadata stays consistent across operations."""

    def test_next_index_increments_correctly(self, detector, sample_vector):
        """next_index should increment with each insert."""
        tree_key = "binance:crypto_spot"

        for i in range(5):
            detector.ingest_data(sample_vector)
            tree_state = detector.forest[tree_key]
            assert tree_state.next_index == i + 1

    def test_oldest_index_updates_after_eviction(self, detector, sample_vector):
        """oldest_index should update when oldest point is evicted."""
        window_size = detector.config["window_size"]
        tree_key = "binance:crypto_spot"

        # Fill window
        for i in range(window_size):
            detector.ingest_data(sample_vector)

        tree_state = detector.forest[tree_key]
        assert tree_state.oldest_index == 0

        # Insert one more to cause eviction
        detector.ingest_data(sample_vector)

        tree_state = detector.forest[tree_key]
        assert tree_state.oldest_index == 1  # 0 was evicted

    def test_current_size_matches_indices_length(self, detector, sample_vector):
        """current_size should always equal len(indices)."""
        tree_key = "binance:crypto_spot"

        for i in range(10):
            detector.ingest_data(sample_vector)
            tree_state = detector.forest[tree_key]
            assert tree_state.current_size == len(tree_state.indices)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
