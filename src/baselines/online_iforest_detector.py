import sys
import os
from pathlib import Path
import numpy as np
from typing import Dict, Optional

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
    def __init__(self, config: dict):
        self.config = config
        
        # Online-iForest parameters (following their paper defaults)
        self.window_size = config.get("window_size", 1024)
        self.num_trees = config.get("num_trees", 32)
        self.max_leaf_samples = config.get("max_leaf_samples", 32)
        self.iforest_type = config.get("type", "adaptive")
        self.min_fill_threshold = config.get("min_fill_threshold", 50)

        self.model = OnlineIForest.create(
            iforest_type='boundedrandomprojectiononlineiforest',
            num_trees=self.num_trees,
            max_leaf_samples=self.max_leaf_samples,
            window_size=self.window_size,
            type=self.iforest_type,
            subsample=1.0,
            branching_factor=2,
            metric='axisparallel',
            n_jobs=1
        )

        self.sample_count = 0
        self.is_warm = False

        self.score_count = 0
        self.stats = Stats(mean=0.0, std=0.0, m2=0.0)
        self.z_score = 0.0
    
    def ingest_data(self, data: NormalizedVectorDto) -> Optional[Dict]:
        features = np.array([[
            data.z_intertick_fast,
            data.z_price_step_fast,
            data.z_intertick_slow,
            data.z_price_step_slow,
            data.cusum_intertick,
            data.cusum_price_step,
        ]])

        self.model.learn_batch(features)
        
        self.sample_count += 1

        if self.sample_count >= self.min_fill_threshold:
            self.is_warm = True
        
        if not self.is_warm:
            return None

        raw_scores = self.model.score_batch(features)
        raw_score = float(raw_scores[0])

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
        return "onlineiforest"
