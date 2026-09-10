"""
Z-Score Threshold Baseline Detector

Per-feature statistical threshold model. Computes mean and standard deviation
for each feature over training data (first two weeks), then freezes.
Score is the maximum absolute z-score across all features.

Represents status quo production tools (e.g., Geneos-style monitoring).

Training Assumption:
    First two weeks of unmodified replay data (no injected anomalies).
    This assumption is documented in the thesis methodology section.
"""

import numpy as np
from typing import Dict, Optional

from src.baselines.base_detector import BaseDetector
from src.detection.stats import Stats, update_stats
from src.kafka.consumer import NormalizedVectorDto


class ZScoreDetector(BaseDetector):
    """
    Frozen statistical threshold baseline.
    
    Training phase: Collects statistics on first N samples.
    Frozen phase: Uses frozen statistics to score new samples.
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.training_samples = config.get("training_samples", 20000)
        
        self.is_trained = False
        self.sample_count = 0
        
        self.feature_means = None
        self.feature_stds = None
        self.training_data = []
        
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
        """Train on collected data and freeze."""
        training_matrix = np.array(self.training_data)
        
        self.feature_means = np.mean(training_matrix, axis=0)
        self.feature_stds = np.std(training_matrix, axis=0)
        
        self.feature_stds = np.where(
            self.feature_stds < 1e-8,
            1.0,
            self.feature_stds
        )
        
        self.is_trained = True
        self.training_data = []
        
        print(f"[ZScoreDetector] Trained on {self.sample_count} samples")
        print(f"[ZScoreDetector] Feature means: {self.feature_means}")
        print(f"[ZScoreDetector] Feature stds: {self.feature_stds}")
    
    def _compute_score(self, features: np.ndarray) -> float:
        """Compute max absolute z-score across features."""
        z_scores = np.abs((features - self.feature_means) / self.feature_stds)
        return float(np.max(z_scores))
    
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
        return "zscore"
