"""
Adapter to make RRCF AnomalyDetector compatible with BaseDetector interface.

Allows RRCF to be used interchangeably with baseline models.
"""

from typing import Dict, Optional

from src.baselines.base_detector import BaseDetector
from src.detection.AnomalyDetector import AnomalyDetector as RRCFDetector
from src.kafka.consumer import NormalizedVectorDto


class RRCFDetectorAdapter(BaseDetector):
    """Adapter wrapping RRCF to implement BaseDetector interface."""
    
    def __init__(self, config: dict):
        self.detector = RRCFDetector(config=config)
    
    def ingest_data(self, data: NormalizedVectorDto) -> Optional[Dict]:
        return self.detector.ingest_data(data)
    
    def get_model_name(self) -> str:
        return "rrcf"
    
    def determine_alert_level(self, z_score: float) -> str:
        return self.detector.determine_alert_level(z_score)
