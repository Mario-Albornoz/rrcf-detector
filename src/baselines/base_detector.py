from abc import ABC, abstractmethod
from typing import Dict, Optional

from src.kafka.consumer import NormalizedVectorDto


class BaseDetector(ABC):
    @abstractmethod
    def ingest_data(self, data: NormalizedVectorDto) -> Optional[Dict]:
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        pass
    
    def determine_alert_level(self, z_score: float) -> str:
        abs_z = abs(z_score)
        
        if abs_z < 2.0:
            return "normal"
        elif abs_z < 3.0:
            return "medium"
        else:
            return "high"
