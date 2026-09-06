"""
Kafka module for RRCF anomaly detector.
Provides high-performance consumer and producer with integrated metrics.
"""

from .consumer import NormalizedVectorConsumer, NormalizedVectorDto
from .producer import AlertProducer, AnomalyAlertDto
from .metrics import KafkaMetrics

__all__ = [
    'NormalizedVectorConsumer',
    'NormalizedVectorDto',
    'AlertProducer',
    'AnomalyAlertDto',
    'KafkaMetrics',
]
