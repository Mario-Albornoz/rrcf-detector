from collections import deque
from datetime import datetime

import pytest

from src.detection.AnomalyDetector import AnomalyDetector, TreeState
from src.kafka.consumer import NormalizedVectorDto


@pytest.fixture
def config():
    return {"window_size": 5, "min_fill_threshold": 1}


@pytest.fixture
def detector(config):
    return AnomalyDetector(config)


@pytest.fixture
def sample_vector():
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
    def test_tree_created_on_first_vector(self, detector, sample_vector):
        assert detector.get_tree_count() == 0

        detector.ingest_data(sample_vector)

        assert detector.get_tree_count() == 1
        assert "binance:crypto_spot" in detector.forest

    def test_tree_not_recreated_on_subsequent_vectors(self, detector, sample_vector):
        detector.ingest_data(sample_vector)
        tree_id_1 = id(detector.forest["binance:crypto_spot"])

        detector.ingest_data(sample_vector)
        tree_id_2 = id(detector.forest["binance:crypto_spot"])

        assert tree_id_1 == tree_id_2
        assert detector.get_tree_count() == 1

    def test_multiple_keys_create_separate_trees(self, detector):
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
    def test_window_size_never_exceeded(self, detector, sample_vector):
        window_size = detector.config["window_size"]

        for i in range(window_size * 3):
            detector.ingest_data(sample_vector)

        state = detector.get_window_state("binance:crypto_spot")
        assert state["current_size"] == window_size
        assert state["current_size"] <= state["max_size"]

    def test_window_fills_gradually(self, detector, sample_vector):
        for i in range(3):
            detector.ingest_data(sample_vector)
            state = detector.get_window_state("binance:crypto_spot")
            assert state["current_size"] == i + 1

    def test_oldest_point_evicted_on_overflow(self, detector):
        window_size = detector.config["window_size"]

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

        assert state["current_size"] == window_size
        assert state["oldest_index"] == 2
        assert state["newest_index"] == 6
        assert state["all_indices"] == [2, 3, 4, 5, 6]


class TestInsertScoreRoundtrip:
    def test_score_returned_after_insert(self, detector, sample_vector):
        result = detector.ingest_data(sample_vector)
        
        assert isinstance(result, dict)
        assert "raw_score" in result
        assert "z_score" in result
        assert "stats" in result
        assert isinstance(result["raw_score"], (int, float))
        assert result["raw_score"] >= 0

    def test_normal_points_have_similar_scores(self, detector):
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
            result = detector.ingest_data(vector)
            scores.append(result["raw_score"])

        assert all(score < 10 for score in scores)

    def test_anomalous_point_has_higher_score(self, detector):
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

        anomaly_vector = NormalizedVectorDto(
            exchange="binance",
            instrument="BTC-USDT",
            instrument_class="crypto_spot",
            timestamp=datetime.now(),
            model_key="test",
            z_intertick_fast=100.0,
            z_price_step_fast=100.0,
            z_intertick_slow=100.0,
            z_price_step_slow=100.0,
            cusum_intertick=100.0,
            cusum_price_step=100.0,
            gap_flag=0,
            warmup_flag=0,
            session_fallback_flag=0,
        )
        anomaly_score = detector.ingest_data(anomaly_vector)

        assert anomaly_score["raw_score"] > normal_score["raw_score"]


class TestMemoryManagement:
    def test_no_memory_growth_over_repeated_inserts(self, detector, sample_vector):
        window_size = detector.config["window_size"]

        for i in range(window_size * 100):
            detector.ingest_data(sample_vector)

        state = detector.get_window_state("binance:crypto_spot")
        tree_state = detector.forest["binance:crypto_spot"]

        assert state["current_size"] == window_size
        assert len(tree_state.indices) == window_size
        assert tree_state.indices.maxlen == window_size

    def test_indices_deque_has_correct_maxlen(self, detector, sample_vector):
        detector.ingest_data(sample_vector)

        tree_state = detector.forest["binance:crypto_spot"]
        assert tree_state.indices.maxlen == detector.config["window_size"]


class TestWindowState:
    def test_window_state_tracks_size_correctly(self, detector, sample_vector):
        state_0 = detector.get_window_state("binance:crypto_spot")
        assert state_0["error"] == "Tree not found"

        detector.ingest_data(sample_vector)
        state_1 = detector.get_window_state("binance:crypto_spot")
        assert state_1["current_size"] == 1

        detector.ingest_data(sample_vector)
        state_2 = detector.get_window_state("binance:crypto_spot")
        assert state_2["current_size"] == 2

    def test_window_state_tracks_indices(self, detector, sample_vector):
        for i in range(3):
            detector.ingest_data(sample_vector)

        state = detector.get_window_state("binance:crypto_spot")
        assert state["oldest_index"] == 0
        assert state["newest_index"] == 2
        assert state["all_indices"] == [0, 1, 2]

    def test_window_state_after_eviction(self, detector, sample_vector):
        window_size = detector.config["window_size"]

        for i in range(window_size + 2):
            detector.ingest_data(sample_vector)

        state = detector.get_window_state("binance:crypto_spot")
        assert state["oldest_index"] == 2
        assert state["newest_index"] == 6
        assert len(state["all_indices"]) == window_size


class TestEdgeCases:
    def test_empty_tree_state(self, detector):
        state = detector.get_window_state("nonexistent:key")
        assert "error" in state
        assert state["error"] == "Tree not found"

    def test_score_on_empty_tree(self, detector):
        score = detector.scoreCoDisp("nonexistent:key", 0)
        assert score == 0

    def test_first_insert_has_index_zero(self, detector, sample_vector):
        detector.ingest_data(sample_vector)

        state = detector.get_window_state("binance:crypto_spot")
        assert state["oldest_index"] == 0
        assert state["newest_index"] == 0
        assert state["all_indices"] == [0]


class TestMetadataConsistency:
    def test_next_index_increments_correctly(self, detector, sample_vector):
        tree_key = "binance:crypto_spot"

        for i in range(5):
            detector.ingest_data(sample_vector)
            tree_state = detector.forest[tree_key]
            assert tree_state.next_index == i + 1

    def test_oldest_index_updates_after_eviction(self, detector, sample_vector):
        window_size = detector.config["window_size"]
        tree_key = "binance:crypto_spot"

        for i in range(window_size):
            detector.ingest_data(sample_vector)

        tree_state = detector.forest[tree_key]
        assert tree_state.oldest_index == 0

        detector.ingest_data(sample_vector)

        tree_state = detector.forest[tree_key]
        assert tree_state.oldest_index == 1

    def test_current_size_matches_indices_length(self, detector, sample_vector):
        tree_key = "binance:crypto_spot"

        for i in range(10):
            detector.ingest_data(sample_vector)
            tree_state = detector.forest[tree_key]
            assert tree_state.current_size == len(tree_state.indices)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
