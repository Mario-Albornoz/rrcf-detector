"""
Base detector interface for all anomaly detection models.

All models (RRCF + 3 baselines) implement this interface for uniform scoring.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional

from src.kafka.consumer import NormalizedVectorDto


class BaseDetector(ABC):
    """
    Abstract base class for all anomaly detectors.
    
    All models return same output format:
    {
        "raw_score": float,
        "z_score": float,
        "stats": {"mean": float, "std": float, "count": int}
    }
    """
    
    @abstractmethod
    def ingest_data(self, data: NormalizedVectorDto) -> Optional[Dict]:
        """
        Process a data point and return anomaly score.
        
        Args:
            data: Normalized feature vector
            
        Returns:
            Dict with raw_score, z_score, stats, or None if not ready (cold start)
        """
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Return model identifier for output tagging."""
        pass
    
    def determine_alert_level(self, z_score: float) -> str:
        """
        Classify z-score into alert levels (shared across all models).
        
        Args:
            z_score: Calibrated z-score
            
        Returns:
            "normal", "medium", or "high"
        """
        abs_z = abs(z_score)
        
        if abs_z < 2.0:
            return "normal"
        elif abs_z < 3.0:
            return "medium"
        else:
            return "high"
