"""
Isolation Forest Baseline Detector

scikit-learn IsolationForest trained on first two weeks of data, then frozen.
One instance per (exchange, instrument_class).

Represents current best-practice for unsupervised batch anomaly detection.
Used to measure concept drift - how quickly batch models degrade as markets evolve.

Training Assumption:
    First two weeks of unmodified replay data (no injected anomalies).
    Model is intentionally frozen (not retrained) to measure concept drift.
    This choice is documented in the thesis methodology section.
"""

import numpy as np
from typing import Dict, Optional
from sklearn.ensemble import IsolationForest

from src.baselines.base_detector import BaseDetector
from src.detection.stats import Stats
from src.kafka.consumer import NormalizedVectorDto


class IsolationForestDetector(BaseDetector):
    """
    Frozen batch ML baseline using Isolation Forest.
    
    Training phase: Collects samples and fits sklearn model.
    Frozen phase: Uses frozen model to score new samples.
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.training_samples = config.get("training_samples", 20000)
        self.n_estimators = config.get("n_estimators", 100)
        self.contamination = config.get("contamination", 0.1)
        
        self.is_trained = False
        self.sample_count = 0
        self.training_data = []
        
        self.model = None
        
        self.score_count = 0
        self.stats = Stats(mean=0.0, std=0.0, m2=0.0)
        self.z_score = 0.0
        
    def ingest_data(self, data: NormalizedVectorDto) -> Optional[Dict]:
        features = np.array([
            data.z_intertick_fast,
            data.z_price_step_fast,
            data.z_intertick_slow,
            data.z_price_step_slow,
            data.cusum_intertick,
            data.cusum_price_step,
        ])
        
        if not self.is_trained:
            self.training_data.append(features)
            self.sample_count += 1
            
            if self.sample_count >= self.training_samples:
                self._train()
            
            return None
        
        raw_score = self._compute_score(features)
        
        self._update_stats(raw_score)
        
        return {
            "raw_score": raw_score,
            "z_score": self.z_score,
            "stats": {
                "mean": self.stats.mean,
                "std": self.stats.std,
                "count": self.score_count,
            }
        }
    
    def _train(self):
        """Train Isolation Forest and freeze."""
        training_matrix = np.array(self.training_data)
        
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(training_matrix)
        
        self.is_trained = True
        self.training_data = []
        
        print(f"[IsolationForest] Trained on {self.sample_count} samples")
        print(f"[IsolationForest] n_estimators={self.n_estimators}, contamination={self.contamination}")
    
    def _compute_score(self, features: np.ndarray) -> float:
        """
        Compute anomaly score using Isolation Forest.
        
        Note: sklearn returns negative scores where more negative = more anomalous.
        We negate to match RRCF convention (higher = more anomalous).
        """
        score = self.model.score_samples([features])[0]
        return float(-score)
    
    def _update_stats(self, raw_score: float):
        """Update rolling statistics using Welford's algorithm."""
        self.score_count += 1
        delta = raw_score - self.stats.mean
        self.stats.mean += delta / self.score_count
        delta2 = raw_score - self.stats.mean
        self.stats.m2 += delta * delta2
        
        if self.score_count > 1:
            variance = self.stats.m2 / (self.score_count - 1)
            self.stats.std = np.sqrt(variance)
            self.z_score = (
                (raw_score - self.stats.mean) / self.stats.std
                if self.stats.std > 0 else 0
            )
        else:
            self.z_score = 0
    
    def get_model_name(self) -> str:
        return "isoforest"
