"""
Half-Space Trees Baseline Detector

River HalfSpaceTrees - genuinely online streaming anomaly detection.
Introduced by Tan, Ting & Liu (2011) at IJCAI.

Unlike frozen baselines, HST updates incrementally with each vector.
Tree structure is fixed; only mass counts update. Constant time and memory.

No Training Assumption:
    HST learns from the stream from tick one. No batch training phase.
    This makes it a direct streaming competitor to RRCF.
    
The most interesting baseline - both are online learners, question becomes
which is better suited to financial feed health domain.
"""

import numpy as np
from typing import Dict, Optional
from river.anomaly import HalfSpaceTrees

from src.baselines.base_detector import BaseDetector
from src.detection.stats import Stats
from src.kafka.consumer import NormalizedVectorDto


class HalfSpaceTreesDetector(BaseDetector):
    """
    Online streaming baseline using River Half-Space Trees.
    
    No training phase - learns from stream immediately.
    Direct competitor to RRCF.
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.window_size = config.get("window_size", 1000)
        self.n_trees = config.get("n_trees", 25)
        self.height = config.get("height", 8)
        self.min_fill_threshold = config.get("min_fill_threshold", 50)
        
        self.model = HalfSpaceTrees(
            n_trees=self.n_trees,
            height=self.height,
            window_size=self.window_size,
            seed=42
        )
        
        self.sample_count = 0
        self.is_warm = False
        
        self.score_count = 0
        self.stats = Stats(mean=0.0, std=0.0, m2=0.0)
        self.z_score = 0.0
        
    def ingest_data(self, data: NormalizedVectorDto) -> Optional[Dict]:
        features_dict = {
            "z_intertick_fast": data.z_intertick_fast,
            "z_price_step_fast": data.z_price_step_fast,
            "z_intertick_slow": data.z_intertick_slow,
            "z_price_step_slow": data.z_price_step_slow,
            "cusum_intertick": data.cusum_intertick,
            "cusum_price_step": data.cusum_price_step,
        }
        
        raw_score = self.model.score_one(features_dict)
        
        self.model.learn_one(features_dict)
        
        self.sample_count += 1
        
        if self.sample_count >= self.min_fill_threshold:
            self.is_warm = True
        
        if not self.is_warm:
            return None
        
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
        return "halfspace"
