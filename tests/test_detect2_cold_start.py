"""
Unit tests for DETECT-2: Cold start handling (equities-only, no calibration needed).

Acceptance Criteria (modified for equities-only):
1. Cold-start flag correctly suppresses/marks scores below the fill threshold
2. Tree transitions from cold to warm at exactly min_fill_threshold
3. No scores returned during warmup (returns None)
4. Normal scoring resumes after warmup completes
"""

from datetime import datetime

import pytest

from src.detection.AnomalyDetector import AnomalyDetector, TreeState
from src.kafka.consumer import NormalizedVectorDto


@pytest.fixture
def config_small_threshold():
    """Config with small threshold for fast testing."""
    return {"window_size": 100, "min_fill_threshold": 5}


@pytest.fixture
def config_large_threshold():
    """Config with larger threshold to test warmup behavior."""
    return {"window_size": 100, "min_fill_threshold": 30}


@pytest.fixture
def detector_small(config_small_threshold):
    """Detector with small warmup threshold."""
    return AnomalyDetector(config_small_threshold)


@pytest.fixture
def detector_large(config_large_threshold):
    """Detector with large warmup threshold."""
    return AnomalyDetector(config_large_threshold)


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


class TestColdStartSuppression:
    """Test that scores are suppressed during cold start."""

    def test_returns_none_before_threshold(self, detector_small, sample_vector):
        """ingest_data should return None before min_fill_threshold is reached."""
        threshold = detector_small.config["min_fill_threshold"]

        # Insert threshold - 1 points
        for i in range(threshold - 1):
            score = detector_small.ingest_data(sample_vector)
            assert score is None, f"Expected None at point {i+1}/{threshold-1}, got {score}"

    def test_returns_score_at_threshold(self, detector_small, sample_vector):
        """ingest_data should return a score exactly at min_fill_threshold."""
        threshold = detector_small.config["min_fill_threshold"]

        # Insert threshold - 1 points (all return None)
        for i in range(threshold - 1):
            detector_small.ingest_data(sample_vector)

        # Threshold-th point should return a score
        result = detector_small.ingest_data(sample_vector)
        assert result is not None, f"Expected score at threshold ({threshold}), got None"
        assert isinstance(result, dict)
        assert "raw_score" in result
        assert result["raw_score"] >= 0

    def test_returns_score_after_threshold(self, detector_small, sample_vector):
        """ingest_data should continue returning scores after threshold."""
        threshold = detector_small.config["min_fill_threshold"]

        # Warmup
        for i in range(threshold):
            detector_small.ingest_data(sample_vector)

        # Insert 10 more points, all should return scores
        for i in range(10):
            result = detector_small.ingest_data(sample_vector)
            assert result is not None, f"Expected score at point {threshold + i + 1}, got None"
            assert isinstance(result, dict)
            assert "raw_score" in result

    def test_large_threshold_suppresses_longer(self, detector_large, sample_vector):
        """Larger threshold should suppress scores for more points."""
        threshold = detector_large.config["min_fill_threshold"]
        assert threshold == 30

        # Insert 29 points, all should return None
        for i in range(29):
            score = detector_large.ingest_data(sample_vector)
            assert score is None, f"Expected None at point {i+1}/29"

        # 30th point should return score
        result = detector_large.ingest_data(sample_vector)
        assert result is not None
        assert isinstance(result, dict)
        assert "raw_score" in result


class TestWarmFlagTransition:
    """Test is_warm flag transitions correctly."""

    def test_tree_starts_cold(self, detector_small, sample_vector):
        """New tree should start with is_warm=False."""
        detector_small.ingest_data(sample_vector)

        tree_key = "binance:crypto_spot"
        tree_state: TreeState = detector_small.forest[tree_key]

        # After 1 point (threshold=5), should still be cold
        assert tree_state.is_warm is False

    def test_tree_warms_at_threshold(self, detector_small, sample_vector):
        """is_warm should become True at exactly min_fill_threshold."""
        threshold = detector_small.config["min_fill_threshold"]
        tree_key = "binance:crypto_spot"

        # Insert threshold points
        for i in range(threshold):
            detector_small.ingest_data(sample_vector)

        tree_state: TreeState = detector_small.forest[tree_key]
        assert tree_state.is_warm is True
        assert tree_state.current_size == threshold

    def test_tree_stays_warm(self, detector_small, sample_vector):
        """is_warm should remain True after threshold."""
        threshold = detector_small.config["min_fill_threshold"]
        tree_key = "binance:crypto_spot"

        # Warmup + 20 more points
        for i in range(threshold + 20):
            detector_small.ingest_data(sample_vector)

        tree_state: TreeState = detector_small.forest[tree_key]
        assert tree_state.is_warm is True

    def test_tree_does_not_warm_before_threshold(self, detector_large, sample_vector):
        """is_warm should remain False before threshold."""
        threshold = detector_large.config["min_fill_threshold"]
        tree_key = "binance:crypto_spot"

        # Insert threshold - 1 points
        for i in range(threshold - 1):
            detector_large.ingest_data(sample_vector)

        tree_state: TreeState = detector_large.forest[tree_key]
        assert tree_state.is_warm is False
        assert tree_state.current_size == threshold - 1


class TestMultipleTreesColdStart:
    """Test cold start behavior with multiple trees (different keys)."""

    def test_each_tree_warms_independently(self, detector_small):
        """Each (exchange, class) key should warm up independently."""
        threshold = detector_small.config["min_fill_threshold"]

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

        # Warm up binance tree completely
        for i in range(threshold):
            detector_small.ingest_data(vector1)

        # Insert only 1 point to coinbase tree
        detector_small.ingest_data(vector2)

        # binance should be warm, coinbase should be cold
        binance_state: TreeState = detector_small.forest["binance:crypto_spot"]
        coinbase_state: TreeState = detector_small.forest["coinbase:crypto_spot"]

        assert binance_state.is_warm is True
        assert coinbase_state.is_warm is False

        # binance should return scores, coinbase should return None
        score1 = detector_small.ingest_data(vector1)
        score2 = detector_small.ingest_data(vector2)

        assert score1 is not None
        assert score2 is None


class TestThresholdConfiguration:
    """Test that threshold is correctly read from config."""

    def test_threshold_from_config(self, config_small_threshold):
        """min_fill_threshold should be read from config."""
        detector = AnomalyDetector(config_small_threshold)
        assert detector.config["min_fill_threshold"] == 5

        vector = NormalizedVectorDto(
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

        detector.ingest_data(vector)

        tree_state: TreeState = detector.forest["binance:crypto_spot"]
        assert tree_state.min_fill_threshold == 5

    def test_different_thresholds_produce_different_warmup_times(self):
        """Different threshold values should result in different warmup times."""
        detector_fast = AnomalyDetector({"window_size": 100, "min_fill_threshold": 3})
        detector_slow = AnomalyDetector({"window_size": 100, "min_fill_threshold": 10})

        vector = NormalizedVectorDto(
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

        # Insert 5 points to both
        for i in range(5):
            score_fast = detector_fast.ingest_data(vector)
            score_slow = detector_slow.ingest_data(vector)

        # Fast detector should be warm (threshold=3), slow should be cold (threshold=10)
        tree_state_fast: TreeState = detector_fast.forest["binance:crypto_spot"]
        tree_state_slow: TreeState = detector_slow.forest["binance:crypto_spot"]

        assert tree_state_fast.is_warm is True
        assert tree_state_slow.is_warm is False


class TestColdStartEdgeCases:
    """Test edge cases for cold start logic."""

    def test_threshold_equal_to_one(self):
        """Tree with threshold=1 should warm immediately on first point."""
        detector = AnomalyDetector({"window_size": 100, "min_fill_threshold": 1})

        vector = NormalizedVectorDto(
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

        result = detector.ingest_data(vector)

        # Should return a score immediately (threshold=1)
        assert result is not None
        assert isinstance(result, dict)
        assert "raw_score" in result

        tree_state: TreeState = detector.forest["binance:crypto_spot"]
        assert tree_state.is_warm is True

    def test_threshold_larger_than_window(self):
        """Threshold larger than window size should still work (tree warms when window fills)."""
        detector = AnomalyDetector({"window_size": 10, "min_fill_threshold": 50})

        vector = NormalizedVectorDto(
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

        # Insert 10 points (fill window)
        for i in range(10):
            score = detector.ingest_data(vector)
            # Tree can't reach threshold=50 since window=10
            assert score is None

        # Tree should still be cold (can't reach threshold)
        tree_state: TreeState = detector.forest["binance:crypto_spot"]
        assert tree_state.is_warm is False
        assert tree_state.current_size == 10  # Window is full


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
