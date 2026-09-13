"""
Online Isolation Forest Baseline Detector

Online-iForest by Leveni, Cassales, Pfahringer, Bifet & Boracchi (ICML 2024).
Genuinely online streaming anomaly detection using multi-resolution histograms
with sliding window and learning/forgetting procedures.

Paper: https://proceedings.mlr.press/v235/leveni24a.html
Code: https://github.com/ineveLoppiliF/Online-Isolation-Forest

Direct competitor to RRCF - both are online tree-based methods.
Key question: which is better suited to financial feed health anomaly detection?

No Training Assumption:
    Online-iForest learns from the stream with a sliding window.
    Automatically forgets old points and learns new ones.
    No batch training phase required.
"""

import sys
import os
from pathlib import Path
import numpy as np
from typing import Dict, Optional

# Add Online-Isolation-Forest to Python path
_current_file = Path(__file__).resolve()
_baselines_dir = _current_file.parent
_online_iforest_path = _baselines_dir / "Online-Isolation-Forest"
if str(_online_iforest_path) not in sys.path:
    sys.path.insert(0, str(_online_iforest_path))

from OnlineIForest import OnlineIForest

from src.baselines.base_detector import BaseDetector
from src.detection.stats import Stats
from src.kafka.consumer import NormalizedVectorDto


class OnlineIForestDetector(BaseDetector):
    """
    Online streaming baseline using Online Isolation Forest (ICML 2024).
    
    Truly online with sliding window - learns new points, forgets old ones.
    Direct competitor to RRCF for streaming anomaly detection.
    
    Config options:
        - window_size: Sliding window size (default: 1024)
        - num_trees: Number of trees in forest (default: 32)
        - max_leaf_samples: Max samples in leaf before split (default: 32)
        - type: 'adaptive' or 'fixed' (default: 'adaptive')
        - min_fill_threshold: Warmup samples before scoring (default: 50)
    """
    
    def __init__(self, config: dict):
        self.config = config
        
        # Online-iForest parameters (following their paper defaults)
        self.window_size = config.get("window_size", 1024)
        self.num_trees = config.get("num_trees", 32)
        self.max_leaf_samples = config.get("max_leaf_samples", 32)
        self.iforest_type = config.get("type", "adaptive")  # 'adaptive' or 'fixed'
        self.min_fill_threshold = config.get("min_fill_threshold", 50)
        
        # Initialize Online-iForest
        self.model = OnlineIForest.create(
            iforest_type='boundedrandomprojectiononlineiforest',
            num_trees=self.num_trees,
            max_leaf_samples=self.max_leaf_samples,
            window_size=self.window_size,
            type=self.iforest_type,
            subsample=1.0,
            branching_factor=2,
            metric='axisparallel',
            n_jobs=1  # Single-threaded per worker (workers are already parallel)
        )
        
        # Tracking
        self.sample_count = 0
        self.is_warm = False
        
        # Welford statistics for z-score calibration
        self.score_count = 0
        self.stats = Stats(mean=0.0, std=0.0, m2=0.0)
        self.z_score = 0.0
    
    def ingest_data(self, data: NormalizedVectorDto) -> Optional[Dict]:
        """
        Process incoming normalized feature vector.
        
        Args:
            data: Normalized feature vector from aggregator
            
        Returns:
            Dict with raw_score, z_score, stats, or None if cold start
        """
        # Convert to numpy array for Online-iForest
        # Shape: (1, n_features) for single sample batch
        features = np.array([[
            data.z_intertick_fast,
            data.z_price_step_fast,
            data.z_intertick_slow,
            data.z_price_step_slow,
            data.cusum_intertick,
            data.cusum_price_step,
        ]])
        
        # Score before learning (Online-iForest uses batch API)
        # Returns array of shape (n_samples,) - higher score = more anomalous
        raw_scores = self.model.score_batch(features)
        raw_score = float(raw_scores[0])
        
        # Learn the new point (updates forest, handles sliding window automatically)
        self.model.learn_batch(features)
        
        self.sample_count += 1
        
        # Cold start: wait until minimum samples collected
        if self.sample_count >= self.min_fill_threshold:
            self.is_warm = True
        
        if not self.is_warm:
            return None
        
        # Update rolling statistics for z-score calibration
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
    
    def _update_stats(self, raw_score: float):
        """
        Update rolling statistics using Welford's online algorithm.
        
        Computes running mean and standard deviation to calibrate raw scores
        into z-scores for interpretability.
        """
        self.score_count += 1
        
        # Welford's algorithm for online variance
        delta = raw_score - self.stats.mean
        self.stats.mean += delta / self.score_count
        delta2 = raw_score - self.stats.mean
        self.stats.m2 += delta * delta2
        
        if self.score_count > 1:
            variance = self.stats.m2 / (self.score_count - 1)
            self.stats.std = np.sqrt(variance)
            
            # Compute z-score: how many standard deviations from mean
            self.z_score = (
                (raw_score - self.stats.mean) / self.stats.std
                if self.stats.std > 0 else 0
            )
        else:
            # Not enough samples for meaningful std
            self.z_score = 0
    
    def get_model_name(self) -> str:
        """Return model identifier for output tagging."""
        return "onlineiforest"
