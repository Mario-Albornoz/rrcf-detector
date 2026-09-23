"""Configuration classes for RRCF detection service."""

from dataclasses import dataclass, field


@dataclass
class DetectorConfig:
    window_size: int = 1000
    min_fill_threshold: int = 50
    rrcf_forest: dict = field(default_factory=dict)


@dataclass
class KafkaConfig:
    bootstrap_servers: str = "localhost:9092"

    input_topic: str = "normalized-vectors"
    output_topic: str = "anomaly-scores"

    consumer_group_id: str = "rrcf-detector-consumer"
    auto_offset_reset: str = "earliest"
    enable_auto_commit: bool = True
    auto_commit_interval_ms: int = 5000
    session_timeout_ms: int = 30000

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
class ServiceConfig:
    partitioner_config: PartitionerConfig = field(default_factory=PartitionerConfig)
    log_level: str = "INFO"

    @classmethod
    def from_dict(cls, config_dict: dict) -> "ServiceConfig":
        # Extract nested sections with defaults
        detector_dict = config_dict.get("detector", {})
        kafka_dict = config_dict.get("kafka", {})
        service_dict = config_dict.get("service", {})
        
        # Build detector config
        detector_config = DetectorConfig(
            window_size=detector_dict.get("window_size", 1000),
            min_fill_threshold=detector_dict.get("min_fill_threshold", 50),
            rrcf_forest=detector_dict.get("rrcf_forest") or {},
        )

        # Build kafka config with ALL properties from YAML
        kafka_config = KafkaConfig(
            bootstrap_servers=kafka_dict.get("bootstrap_servers", "localhost:9092"),
            input_topic=kafka_dict.get("input_topic", "normalized-vectors"),
            output_topic=kafka_dict.get("output_topic", "anomaly-scores"),
            consumer_group_id=kafka_dict.get("consumer_group_id", "rrcf-detector-consumer"),
            auto_offset_reset=kafka_dict.get("auto_offset_reset", "earliest"),
            enable_auto_commit=kafka_dict.get("enable_auto_commit", True),
            auto_commit_interval_ms=kafka_dict.get("auto_commit_interval_ms", 5000),
            session_timeout_ms=kafka_dict.get("session_timeout_ms", 30000),
            client_id=kafka_dict.get("client_id", "rrcf-detector-producer"),
            linger_ms=kafka_dict.get("linger_ms", 10),
            batch_size=kafka_dict.get("batch_size", 65536),
            compression_type=kafka_dict.get("compression_type", "lz4"),
            acks=kafka_dict.get("acks", 1),
            retries=kafka_dict.get("retries", 3),
        )

        # Build partitioner config
        partitioner_config = PartitionerConfig(
            num_workers=service_dict.get("num_workers", 4),
            queue_max_size=service_dict.get("queue_max_size", 10000),
            detector_config=detector_config,
            kafka_config=kafka_config,
        )

        return cls(
            partitioner_config=partitioner_config,
            log_level=service_dict.get("log_level", "INFO"),
        )
