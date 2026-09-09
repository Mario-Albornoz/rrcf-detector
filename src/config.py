"""Configuration classes for RRCF detection service."""

"""Configuration classes for RRCF detection service."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DetectorConfig:
    window_size: int = 1000
    min_fill_threshold: int = 50


@dataclass
class KafkaConfig:
    bootstrap_servers: str = "localhost:9092"
    
    input_topic: str = "normalized-features"
    output_topic: str = "anomaly-scores"
    
    consumer_group_id: str = "rrcf-detector-consumer"
    auto_offset_reset: str = "earliest"
    enable_auto_commit: bool = True
    auto_commit_interval_ms: int = 5000
    session_timeout_ms: int = 30000
    max_poll_records: int = 500
    
    client_id: str = "rrcf-detector-producer"
    linger_ms: int = 10
    batch_size: int = 65536
    compression_type: str = "lz4"
    acks: int = 1
    retries: int = 3


@dataclass
class WorkerConfig:
    detector_config: DetectorConfig
    kafka_config: KafkaConfig


@dataclass
class PartitionerConfig:
    num_workers: int = 4
    queue_max_size: int = 10000
    
    detector_config: DetectorConfig = field(default_factory=DetectorConfig)
    kafka_config: KafkaConfig = field(default_factory=KafkaConfig)
    
    def to_worker_config(self) -> WorkerConfig:
        return WorkerConfig(
            detector_config=self.detector_config,
            kafka_config=self.kafka_config,
        )


@dataclass
class RedisConfig:
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    socket_timeout: int = 5
    ttl_seconds: int = 604800


@dataclass
class ServiceConfig:
    partitioner_config: PartitionerConfig = field(default_factory=PartitionerConfig)
    log_level: str = "INFO"
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "ServiceConfig":
        detector_config = DetectorConfig(
            window_size=config_dict.get("rrcf_window_size", 1000),
            min_fill_threshold=config_dict.get("min_fill_threshold", 50),
        )
        
        kafka_config = KafkaConfig(
            bootstrap_servers=config_dict.get("kafka_bootstrap_servers", "localhost:9092"),
            input_topic=config_dict.get("kafka_input_topic", "normalized-features"),
            output_topic=config_dict.get("kafka_output_topic", "anomaly-scores"),
            consumer_group_id=config_dict.get("kafka_consumer_group_id", "rrcf-detector-consumer"),
        )
        
        partitioner_config = PartitionerConfig(
            num_workers=config_dict.get("num_workers", 4),
            queue_max_size=config_dict.get("queue_max_size", 10000),
            detector_config=detector_config,
            kafka_config=kafka_config,
        )
        
        return cls(
            partitioner_config=partitioner_config,
            log_level=config_dict.get("log_level", "INFO"),
        )
