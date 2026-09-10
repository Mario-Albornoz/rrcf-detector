"""
Baselines package for anomaly detection comparison.

Contains three baseline models for thesis evaluation:
1. ZScoreDetector - Statistical threshold (frozen)
2. IsolationForestDetector - Batch ML model (frozen)
3. HalfSpaceTreesDetector - Online streaming (River)

Plus RRCFDetectorAdapter to make RRCF work with BaseDetector interface.
"""

from .base_detector import BaseDetector
from .zscore_detector import ZScoreDetector
from .isolation_forest_detector import IsolationForestDetector
from .halfspace_trees_detector import HalfSpaceTreesDetector
from .rrcf_adapter import RRCFDetectorAdapter

__all__ = [
    "BaseDetector",
    "ZScoreDetector",
    "IsolationForestDetector",
    "HalfSpaceTreesDetector",
    "RRCFDetectorAdapter",
]
