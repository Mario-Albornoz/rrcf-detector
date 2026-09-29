from datetime import datetime

import numpy as np
import pytest

from src.baselines import (
    BaseDetector,
    HalfSpaceTreesDetector,
    IsolationForestDetector,
    ZScoreDetector,
)
from src.kafka.consumer import NormalizedVectorDto


@pytest.fixture
def normal_vector():
    return NormalizedVectorDto(
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


@pytest.fixture
def anomalous_vector():
    return NormalizedVectorDto(
        exchange="binance",
        instrument="BTC-USDT",
        instrument_class="crypto_spot",
        timestamp=datetime.now(),
        model_key="test",
        z_intertick_fast=15.0,
        z_price_step_fast=15.0,
        z_intertick_slow=15.0,
        z_price_step_slow=15.0,
        cusum_intertick=30.0,
        cusum_price_step=30.0,
        gap_flag=1,
        warmup_flag=0,
        session_fallback_flag=0,
    )


class TestZScoreDetector:
    def test_implements_base_detector(self):
        detector = ZScoreDetector(config={"training_samples": 100})
        assert isinstance(detector, BaseDetector)
        assert hasattr(detector, "ingest_data")
        assert hasattr(detector, "get_model_name")
        assert hasattr(detector, "determine_alert_level")
    
    def test_model_name(self):
        detector = ZScoreDetector(config={"training_samples": 100})
        assert detector.get_model_name() == "zscore"
    
    def test_training_phase(self, normal_vector):
        detector = ZScoreDetector(config={"training_samples": 10})
        
        for _ in range(9):
            result = detector.ingest_data(normal_vector)
            assert result is None
            assert not detector.is_trained
    
    def test_becomes_trained(self, normal_vector):
        detector = ZScoreDetector(config={"training_samples": 10})
        
        for _ in range(10):
            detector.ingest_data(normal_vector)
        
        assert detector.is_trained
        assert detector.feature_means is not None
        assert detector.feature_stds is not None
    
    def test_output_format(self, normal_vector):
        detector = ZScoreDetector(config={"training_samples": 10})
        
        for _ in range(10):
            detector.ingest_data(normal_vector)
        
        result = detector.ingest_data(normal_vector)
        
        assert result is not None
        assert "raw_score" in result
        assert "z_score" in result
        assert "stats" in result
        assert "mean" in result["stats"]
        assert "std" in result["stats"]
        assert "count" in result["stats"]
    
    def test_sanity_check(self, normal_vector, anomalous_vector):
        detector = ZScoreDetector(config={"training_samples": 200})
        
        normal_scores = []
        for _ in range(200):
            detector.ingest_data(normal_vector)
        
        for _ in range(10):
            result = detector.ingest_data(normal_vector)
            if result:
                normal_scores.append(result["raw_score"])
        
        anomalous_scores = []
        for _ in range(10):
            result = detector.ingest_data(anomalous_vector)
            if result:
                anomalous_scores.append(result["raw_score"])
        
        avg_normal = np.mean(normal_scores)
        avg_anomalous = np.mean(anomalous_scores)
        
        assert avg_anomalous > avg_normal, "Anomalous vectors should score higher"


class TestIsolationForestDetector:
    def test_implements_base_detector(self):
        detector = IsolationForestDetector(config={"training_samples": 100})
        assert isinstance(detector, BaseDetector)
    
    def test_model_name(self):
        detector = IsolationForestDetector(config={"training_samples": 100})
        assert detector.get_model_name() == "isoforest"
    
    def test_training_phase(self, normal_vector):
        detector = IsolationForestDetector(config={"training_samples": 10})
        
        for _ in range(9):
            result = detector.ingest_data(normal_vector)
            assert result is None
    
    def test_becomes_trained(self, normal_vector):
        detector = IsolationForestDetector(config={"training_samples": 10})
        
        for _ in range(10):
            detector.ingest_data(normal_vector)
        
        assert detector.is_trained
        assert detector.model is not None
    
    def test_output_format(self, normal_vector):
        detector = IsolationForestDetector(config={"training_samples": 10})
        
        for _ in range(10):
            detector.ingest_data(normal_vector)
        
        result = detector.ingest_data(normal_vector)
        
        assert result is not None
        assert "raw_score" in result
        assert "z_score" in result
        assert "stats" in result
    
    def test_score_is_positive(self, normal_vector):
        detector = IsolationForestDetector(config={"training_samples": 10})
        
        for _ in range(10):
            detector.ingest_data(normal_vector)
        
        result = detector.ingest_data(normal_vector)
        
        assert result["raw_score"] >= 0, "Score should be positive (negated)"
    
    def test_sanity_check(self, normal_vector, anomalous_vector):
        detector = IsolationForestDetector(config={"training_samples": 200})

        for i in range(200):
            varied_vector = NormalizedVectorDto(
                exchange="binance",
                instrument="BTC-USDT",
                instrument_class="crypto_spot",
                timestamp=datetime.now(),
                model_key="test",
                z_intertick_fast=1.0 + (i % 10) * 0.1,
                z_price_step_fast=0.5 + (i % 5) * 0.1,
                z_intertick_slow=1.0 + (i % 8) * 0.1,
                z_price_step_slow=0.5 + (i % 6) * 0.1,
                cusum_intertick=2.0 + (i % 4) * 0.1,
                cusum_price_step=1.0 + (i % 3) * 0.1,
                gap_flag=0,
                warmup_flag=0,
                session_fallback_flag=0,
            )
            detector.ingest_data(varied_vector)
        
        normal_scores = []
        for i in range(10):
            varied_vector = NormalizedVectorDto(
                exchange="binance",
                instrument="BTC-USDT",
                instrument_class="crypto_spot",
                timestamp=datetime.now(),
                model_key="test",
                z_intertick_fast=1.0 + (i % 10) * 0.1,
                z_price_step_fast=0.5 + (i % 5) * 0.1,
                z_intertick_slow=1.0 + (i % 8) * 0.1,
                z_price_step_slow=0.5 + (i % 6) * 0.1,
                cusum_intertick=2.0 + (i % 4) * 0.1,
                cusum_price_step=1.0 + (i % 3) * 0.1,
                gap_flag=0,
                warmup_flag=0,
                session_fallback_flag=0,
            )
            result = detector.ingest_data(varied_vector)
            if result:
                normal_scores.append(result["raw_score"])
        
        anomalous_scores = []
        for _ in range(10):
            result = detector.ingest_data(anomalous_vector)
            if result:
                anomalous_scores.append(result["raw_score"])
        
        avg_normal = np.mean(normal_scores)
        avg_anomalous = np.mean(anomalous_scores)
        
        assert avg_anomalous > avg_normal, f"Anomalous vectors should score higher: {avg_anomalous} vs {avg_normal}"


class TestHalfSpaceTreesDetector:
    def test_implements_base_detector(self):
        detector = HalfSpaceTreesDetector(config={"min_fill_threshold": 10})
        assert isinstance(detector, BaseDetector)
    
    def test_model_name(self):
        detector = HalfSpaceTreesDetector(config={"min_fill_threshold": 10})
        assert detector.get_model_name() == "halfspace"
    
    def test_cold_start(self, normal_vector):
        detector = HalfSpaceTreesDetector(config={"min_fill_threshold": 10})
        
        for _ in range(9):
            result = detector.ingest_data(normal_vector)
            assert result is None
    
    def test_becomes_warm(self, normal_vector):
        detector = HalfSpaceTreesDetector(config={"min_fill_threshold": 10})
        
        for _ in range(10):
            detector.ingest_data(normal_vector)
        
        result = detector.ingest_data(normal_vector)
        assert result is not None
    
    def test_output_format(self, normal_vector):
        detector = HalfSpaceTreesDetector(config={"min_fill_threshold": 10})
        
        for _ in range(10):
            detector.ingest_data(normal_vector)
        
        result = detector.ingest_data(normal_vector)
        
        assert result is not None
        assert "raw_score" in result
        assert "z_score" in result
        assert "stats" in result
    
    def test_sanity_check(self, normal_vector, anomalous_vector):
        detector = HalfSpaceTreesDetector(config={"min_fill_threshold": 50})

        for i in range(200):
            varied_vector = NormalizedVectorDto(
                exchange="binance",
                instrument="BTC-USDT",
                instrument_class="crypto_spot",
                timestamp=datetime.now(),
                model_key="test",
                z_intertick_fast=1.0 + np.random.randn() * 0.5,
                z_price_step_fast=0.5 + np.random.randn() * 0.3,
                z_intertick_slow=1.0 + np.random.randn() * 0.4,
                z_price_step_slow=0.5 + np.random.randn() * 0.2,
                cusum_intertick=2.0 + np.random.randn() * 0.6,
                cusum_price_step=1.0 + np.random.randn() * 0.3,
                gap_flag=0,
                warmup_flag=0,
                session_fallback_flag=0,
            )
            detector.ingest_data(varied_vector)

        normal_scores = []
        for i in range(20):
            varied_vector = NormalizedVectorDto(
                exchange="binance",
                instrument="BTC-USDT",
                instrument_class="crypto_spot",
                timestamp=datetime.now(),
                model_key="test",
                z_intertick_fast=1.0 + np.random.randn() * 0.5,
                z_price_step_fast=0.5 + np.random.randn() * 0.3,
                z_intertick_slow=1.0 + np.random.randn() * 0.4,
                z_price_step_slow=0.5 + np.random.randn() * 0.2,
                cusum_intertick=2.0 + np.random.randn() * 0.6,
                cusum_price_step=1.0 + np.random.randn() * 0.3,
                gap_flag=0,
                warmup_flag=0,
                session_fallback_flag=0,
            )
            result = detector.ingest_data(varied_vector)
            if result:
                normal_scores.append(result["raw_score"])
        
        anomalous_scores = []
        for _ in range(10):
            result = detector.ingest_data(anomalous_vector)
            if result:
                anomalous_scores.append(result["raw_score"])

        assert len(normal_scores) > 0, "Should produce scores for normal data"
        assert len(anomalous_scores) > 0, "Should produce scores for anomalous data"
        assert all(isinstance(s, (int, float)) for s in normal_scores)
        assert all(isinstance(s, (int, float)) for s in anomalous_scores)


class TestAlertLevels:
    def test_alert_levels(self):
        detector = ZScoreDetector(config={"training_samples": 10})
        
        assert detector.determine_alert_level(0.5) == "normal"
        assert detector.determine_alert_level(1.5) == "normal"
        assert detector.determine_alert_level(2.0) == "medium"
        assert detector.determine_alert_level(2.5) == "medium"
        assert detector.determine_alert_level(3.0) == "high"
        assert detector.determine_alert_level(5.0) == "high"
    
    def test_negative_z_scores(self):
        detector = ZScoreDetector(config={"training_samples": 10})
        
        assert detector.determine_alert_level(-0.5) == "normal"
        assert detector.determine_alert_level(-2.5) == "medium"
        assert detector.determine_alert_level(-3.5) == "high"
