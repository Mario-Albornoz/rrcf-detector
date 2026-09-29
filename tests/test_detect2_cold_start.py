from datetime import datetime

import pytest

from src.detection.AnomalyDetector import AnomalyDetector, TreeState
from src.kafka.consumer import NormalizedVectorDto


@pytest.fixture
def config_small_threshold():
    return {"window_size": 100, "min_fill_threshold": 5}


@pytest.fixture
def config_large_threshold():
    return {"window_size": 100, "min_fill_threshold": 30}


@pytest.fixture
def detector_small(config_small_threshold):
    return AnomalyDetector(config_small_threshold)


@pytest.fixture
def detector_large(config_large_threshold):
    return AnomalyDetector(config_large_threshold)


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


class TestColdStartSuppression:
    def test_returns_none_before_threshold(self, detector_small, sample_vector):
        threshold = detector_small.config["min_fill_threshold"]

        for i in range(threshold - 1):
            score = detector_small.ingest_data(sample_vector)
            assert score is None, f"Expected None at point {i+1}/{threshold-1}, got {score}"

    def test_returns_score_at_threshold(self, detector_small, sample_vector):
        threshold = detector_small.config["min_fill_threshold"]

        for i in range(threshold - 1):
            detector_small.ingest_data(sample_vector)

        result = detector_small.ingest_data(sample_vector)
        assert result is not None, f"Expected score at threshold ({threshold}), got None"
        assert isinstance(result, dict)
        assert "raw_score" in result
        assert result["raw_score"] >= 0

    def test_returns_score_after_threshold(self, detector_small, sample_vector):
        threshold = detector_small.config["min_fill_threshold"]

        for i in range(threshold):
            detector_small.ingest_data(sample_vector)

        for i in range(10):
            result = detector_small.ingest_data(sample_vector)
            assert result is not None, f"Expected score at point {threshold + i + 1}, got None"
            assert isinstance(result, dict)
            assert "raw_score" in result

    def test_large_threshold_suppresses_longer(self, detector_large, sample_vector):
        threshold = detector_large.config["min_fill_threshold"]
        assert threshold == 30

        for i in range(29):
            score = detector_large.ingest_data(sample_vector)
            assert score is None, f"Expected None at point {i+1}/29"

        result = detector_large.ingest_data(sample_vector)
        assert result is not None
        assert isinstance(result, dict)
        assert "raw_score" in result


class TestWarmFlagTransition:
    def test_tree_starts_cold(self, detector_small, sample_vector):
        detector_small.ingest_data(sample_vector)

        tree_key = "binance:crypto_spot"
        tree_state: TreeState = detector_small.forest[tree_key]

        assert tree_state.is_warm is False

    def test_tree_warms_at_threshold(self, detector_small, sample_vector):
        threshold = detector_small.config["min_fill_threshold"]
        tree_key = "binance:crypto_spot"

        for i in range(threshold):
            detector_small.ingest_data(sample_vector)

        tree_state: TreeState = detector_small.forest[tree_key]
        assert tree_state.is_warm is True
        assert tree_state.current_size == threshold

    def test_tree_stays_warm(self, detector_small, sample_vector):
        threshold = detector_small.config["min_fill_threshold"]
        tree_key = "binance:crypto_spot"

        for i in range(threshold + 20):
            detector_small.ingest_data(sample_vector)

        tree_state: TreeState = detector_small.forest[tree_key]
        assert tree_state.is_warm is True

    def test_tree_does_not_warm_before_threshold(self, detector_large, sample_vector):
        threshold = detector_large.config["min_fill_threshold"]
        tree_key = "binance:crypto_spot"

        for i in range(threshold - 1):
            detector_large.ingest_data(sample_vector)

        tree_state: TreeState = detector_large.forest[tree_key]
        assert tree_state.is_warm is False
        assert tree_state.current_size == threshold - 1


class TestMultipleTreesColdStart:
    def test_each_tree_warms_independently(self, detector_small):
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

        for i in range(threshold):
            detector_small.ingest_data(vector1)

        detector_small.ingest_data(vector2)

        binance_state: TreeState = detector_small.forest["binance:crypto_spot"]
        coinbase_state: TreeState = detector_small.forest["coinbase:crypto_spot"]

        assert binance_state.is_warm is True
        assert coinbase_state.is_warm is False

        score1 = detector_small.ingest_data(vector1)
        score2 = detector_small.ingest_data(vector2)

        assert score1 is not None
        assert score2 is None


class TestThresholdConfiguration:
    def test_threshold_from_config(self, config_small_threshold):
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

        for i in range(5):
            score_fast = detector_fast.ingest_data(vector)
            score_slow = detector_slow.ingest_data(vector)

        tree_state_fast: TreeState = detector_fast.forest["binance:crypto_spot"]
        tree_state_slow: TreeState = detector_slow.forest["binance:crypto_spot"]

        assert tree_state_fast.is_warm is True
        assert tree_state_slow.is_warm is False


class TestColdStartEdgeCases:
    def test_threshold_equal_to_one(self):
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

        assert result is not None
        assert isinstance(result, dict)
        assert "raw_score" in result

        tree_state: TreeState = detector.forest["binance:crypto_spot"]
        assert tree_state.is_warm is True

    def test_threshold_larger_than_window(self):
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

        for i in range(10):
            score = detector.ingest_data(vector)
            assert score is None

        tree_state: TreeState = detector.forest["binance:crypto_spot"]
        assert tree_state.is_warm is False
        assert tree_state.current_size == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
