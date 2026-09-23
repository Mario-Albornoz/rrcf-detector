"""
Baselines package for anomaly detection comparison.

Contains four baseline models for thesis evaluation:
1. ZScoreDetector - Statistical threshold (frozen)
2. IsolationForestDetector - Batch ML model (frozen)
3. HalfSpaceTreesDetector - Online streaming (River)
4. OnlineIForestDetector - Online Isolation Forest (ICML 2024)

Plus RRCFDetectorAdapter to make RRCF work with BaseDetector interface.
"""

from .base_detector import BaseDetector
from .halfspace_trees_detector import HalfSpaceTreesDetector
from .isolation_forest_detector import IsolationForestDetector
from .online_iforest_detector import OnlineIForestDetector
from .rrcf_adapter import RRCFDetectorAdapter
from .rrcf_forest_detector import RRCFForestDetector
from .zscore_detector import ZScoreDetector

__all__ = [
    "BaseDetector",
    "ZScoreDetector",
    "IsolationForestDetector",
    "HalfSpaceTreesDetector",
    "OnlineIForestDetector",
    "RRCFDetectorAdapter",
    "RRCFForestDetector",
]
