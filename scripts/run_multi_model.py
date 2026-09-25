#!/usr/bin/env python3
"""
Run multi-model comparison service.

Runs the selected models in parallel, each with a dedicated worker process. Each model
writes its scores to its own parquet file (scores_<model>.parquet).

Usage:
    python scripts/run_multi_model.py --config config/baselines.yaml
    python scripts/run_multi_model.py --record data/vectors/vectors_sample.parquet
    python scripts/run_multi_model.py --from-file <vectors_sample.parquet> --models rrcf \\
        --output <dir>/scores.parquet

Architecture:
    One Kafka consumer (this process, the "runner") feeds all models:
    - The runner applies the stride: only every STRIDE-th vector (10 by default) goes on;
      it is chosen here, once, so every model gets the same vectors in the same order
    - With --record the runner writes those vectors to a parquet file
    - Each model has its own worker process and its own input queue
    - The workers score everything they receive (they have no stride of their own)

Replay (--from-file):
    Instead of Kafka the runner reads a recorded vectors_sample.parquet, in order, and feeds
    it to the selected models without dropping anything. Models can then be run one after
    the other, in separate runs, and still score exactly the same vectors.
"""

import argparse
import json
import multiprocessing as mp
import os
import queue
import signal
import sys
import time
from pathlib import Path

# Ignore SIGHUP to prevent termination when parent process exits
signal.signal(signal.SIGHUP, signal.SIG_IGN)

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml
from confluent_kafka import Consumer, KafkaError, KafkaException

from src.baselines import (
    HalfSpaceTreesDetector,
    IsolationForestDetector,
    OnlineIForestDetector,
    RRCFDetectorAdapter,
    RRCFForestDetector,
    ZScoreDetector,
)
from src.config import ServiceConfig
from src.detection.generic_worker import GenericWorker
from src.detection.vector_sample import (
    StrideSampler,
    VectorSampleWriter,
    read_vector_sample,
    sample_row_count,
    summary_path,
)
from src.kafka import deserialize_vector

# One vector in SAMPLE_STRIDE is scored. This is a deliberate CPU workaround: do not lower it.
SAMPLE_STRIDE = 10

# Longest a live shutdown waits for a worker to score its backlog (make stop-all allows 120 s).
LIVE_STOP_WAIT_SEC = 90.0

# Models used when neither --models nor the `models:` list of the config selects any.
DEFAULT_MODELS = ["rrcf", "isoforest"]


def model_registry(detector_config) -> dict:
    """{name: (detector class, its config)} for every model this service can run."""
    forest = detector_config.rrcf_forest
    forest_config = {
        "window_size": forest.get("window_size", detector_config.window_size),
        "num_trees": forest.get("num_trees", 5),
        "min_fill_threshold": forest.get(
            "min_fill_threshold", detector_config.min_fill_threshold
        ),
        "seed": forest.get("seed", 42),
    }
    zscore_config = {"training_days": 1}
    registry = {
        "rrcf": (
            RRCFDetectorAdapter,
            {
                "window_size": detector_config.window_size,
                "min_fill_threshold": detector_config.min_fill_threshold,
            },
        ),
        "rrcf_forest": (RRCFForestDetector, forest_config),
        # The frozen models learn from the whole first day of data (the warm-up day, which
        # the evaluation excludes), not from its first 20,000 vectors (~90 s of trading).
        "zscore": (ZScoreDetector, zscore_config),
        "isoforest": (
            IsolationForestDetector,
            {
                "training_days": 1,
                "training_reservoir": 200_000,
                "n_estimators": 100,
                "contamination": "auto",
            },
        ),
        "halfspace": (
            HalfSpaceTreesDetector,
            {
                "window_size": detector_config.window_size,
                "n_trees": 25,
                "height": 8,
                "min_fill_threshold": detector_config.min_fill_threshold,
            },
        ),
        "onlineiforest": (
            OnlineIForestDetector,
            {
                "window_size": 256,
                "num_trees": 15,
                "max_leaf_samples": 25,
                "type": "adaptive",
                "min_fill_threshold": detector_config.min_fill_threshold,
            },
        ),
    }
    registry.update(ablation_variants(forest_config, zscore_config))
    return registry


def ablation_variants(forest_config: dict, zscore_config: dict) -> dict:
    """Variants of rrcf_forest and zscore that differ from them in one property only, for the
    ablation in docs/Ablation_Plan.md. They are replayed on a recorded vector sample
    (make replay-models MODELS=<name>), never run live.

      C1 sharing:       *_inst          one forest per instrument instead of per exchange
      C2 timescales:    *_fast, *_slow  one timescale (plus the CUSUMs)
      C3 CUSUM:         *_nocusum       the four z-scores only
      C4 normalization: *_z2 vs *_raw2  the same two measurements, normalized or raw

    The raw variants need a recording with the raw fields (feed-handler df5229b or later).
    Per-instrument forests emit a score of 0 while cold (see RRCFForestDetector.cold_score);
    they hold up to one forest per instrument (about 2.8 GB for 5 trees x 2,700 instruments),
    so replay them alone.
    """

    def forest(name, **extra):
        return (RRCFForestDetector, {**forest_config, "name": name, **extra})

    def zscore(name, **extra):
        return (ZScoreDetector, {**zscore_config, "name": name, **extra})

    return {
        "rrcf_forest_inst": forest(
            "rrcf_forest_inst", key_by="instrument", cold_score=0.0
        ),
        "rrcf_forest_fast": forest("rrcf_forest_fast", features="fast"),
        "rrcf_forest_slow": forest("rrcf_forest_slow", features="slow"),
        "rrcf_forest_nocusum": forest("rrcf_forest_nocusum", features="nocusum"),
        "rrcf_forest_z2": forest("rrcf_forest_z2", features="z2"),
        "rrcf_forest_raw2": forest("rrcf_forest_raw2", features="raw2"),
        "rrcf_forest_inst_raw2": forest(
            "rrcf_forest_inst_raw2",
            features="raw2",
            key_by="instrument",
            cold_score=0.0,
        ),
        "zscore_fast": zscore("zscore_fast", features="fast"),
        "zscore_slow": zscore("zscore_slow", features="slow"),
        "zscore_nocusum": zscore("zscore_nocusum", features="nocusum"),
        "zscore_z2": zscore("zscore_z2", features="z2"),
        "zscore_raw2": zscore("zscore_raw2", features="raw2"),
    }


class MultiModelRunner:
    """
    Run the selected models in parallel.

    Each model gets:
    - One dedicated worker process
    - One input queue for message routing
    """

    def __init__(
        self,
        config: ServiceConfig,
        parquet_file: str = None,
        model_names=None,
        stride: int = SAMPLE_STRIDE,
        record_file: str = None,
        overwrite: bool = False,
    ):
        self.config = config
        self.parquet_file = parquet_file or "./data/scores.parquet"
        self.model_names = list(model_names or DEFAULT_MODELS)
        self.sampler = StrideSampler(stride)
        self.recorder = VectorSampleWriter(record_file) if record_file else None
        self.overwrite = overwrite
        self.models = (
            {}
        )  # {model_name: {"detector": class, "queue": Queue, "process": Process}}
        self.consumer = None
        self.running = False
        self.sampled_count = 0
        self.dropped_counts = {}
        self.mode = "live"

    def start(self):
        print("=" * 60)
        print("Multi-Model Anomaly Detection Service")
        print("=" * 60)
        print()

        self._init_models()
        self._start_workers()
        self._start_consumer()

        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

        self.running = True
        self._run_loop()

    def start_replay(self, sample_file: str):
        """Feed a recorded vectors_sample.parquet to the models instead of reading Kafka."""
        print("=" * 60)
        print("Multi-Model Anomaly Detection Service (replay)")
        print("=" * 60)
        print()

        self.mode = "replay"
        total = sample_row_count(sample_file)
        print(f"Replaying {total:,} vectors from {sample_file}")
        print()

        self._init_models()
        self._check_outputs_free()
        self._start_workers()

        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

        self.running = True
        self._run_replay(sample_file, total)

    def _init_models(self):
        """Initialize model configurations."""
        registry = model_registry(self.config.partitioner_config.detector_config)

        print("Initializing models:")
        for model_name in self.model_names:
            detector_class, config = registry[model_name]
            self.models[model_name] = {
                "detector_class": detector_class,
                "config": config,
                "queue": None,
                "process": None,
            }
            self.dropped_counts[model_name] = 0
            print(f"  ✓ {model_name}: {detector_class.__name__}")
        print()

    def _score_file(self, model_name: str) -> str:
        return str(Path(self.parquet_file).parent / f"scores_{model_name}.parquet")

    def _check_outputs_free(self):
        """A replay writes scores into an archive: never silently overwrite one."""
        existing = [
            self._score_file(m)
            for m in self.models
            if os.path.exists(self._score_file(m))
        ]
        if existing and not self.overwrite:
            print("✗ Refusing to overwrite existing scores (pass --overwrite):")
            for f in existing:
                print(f"    {f}")
            sys.exit(1)

    def _start_workers(self):
        """Start one worker process per model."""
        kafka_config = self.config.partitioner_config.kafka_config

        print("Starting workers:")
        for model_name, model_info in self.models.items():
            queue_ = mp.Queue(maxsize=10000)
            model_info["queue"] = queue_

            detector = model_info["detector_class"](model_info["config"])

            # Disable Kafka publishing - only write to Parquet for thesis
            worker_kafka_config = {
                "bootstrap.servers": kafka_config.bootstrap_servers,
                "output_topic": None,  # DISABLED - only write to Parquet
                "client.id": f"multi-model-{model_name}",
                "linger.ms": kafka_config.linger_ms,
                "batch.size": kafka_config.batch_size,
                "compression.type": kafka_config.compression_type,
                "acks": kafka_config.acks,
                "retries": kafka_config.retries,
            }

            process = GenericWorker.start_worker(
                worker_id=0,  # Single worker per model
                detector=detector,
                input_queue=queue_,
                kafka_config=worker_kafka_config,
                parquet_file=self._score_file(model_name),
            )

            model_info["process"] = process
            print(f"  ✓ {model_name} worker (PID: {process.pid})")

        print(f"\n✓ Started {len(self.models)} model workers")
        print()

    def _start_consumer(self):
        """Start Kafka consumer for input vectors."""
        kafka_config = self.config.partitioner_config.kafka_config

        consumer_config = {
            "bootstrap.servers": kafka_config.bootstrap_servers,
            "group.id": kafka_config.consumer_group_id + "-multi",
            "auto.offset.reset": kafka_config.auto_offset_reset,
            "enable.auto.commit": True,
            "auto.commit.interval.ms": 5000,
        }

        self.consumer = Consumer(consumer_config)
        self.consumer.subscribe([kafka_config.input_topic])
        print(f"✓ Subscribed to input topic: {kafka_config.input_topic}")
        if self.recorder:
            print(f"✓ Recording the sampled vectors to: {self.recorder.path}")
        print(f"✓ Stride: every {self.sampler.stride}th vector is scored")
        print()

    # ------------------------------------------------------------- vector routing

    def _put(self, model_name: str, model_info: dict, item, blocking: bool) -> bool:
        """Put an item on a model's queue. In live mode a full queue for 5 s drops it; in
        replay nothing may be dropped, so wait (and fail loudly if the worker died)."""
        while True:
            try:
                model_info["queue"].put(item, block=True, timeout=5.0)
                return True
            except queue.Full:
                if not blocking:
                    self.dropped_counts[model_name] += 1
                    return False
                process = model_info["process"]
                if process is not None and not process.is_alive():
                    raise RuntimeError(f"worker of {model_name} died during the replay")

    def _handle_vector(self, vector, blocking: bool = False) -> bool:
        """Live path for one consumed vector: apply the stride, record, fan out.
        Returns True if the vector was kept."""
        if not self.sampler.take():
            return False
        if self.recorder:
            self.recorder.write(self.sampler.seen, vector)
        self._fan_out(vector, blocking)
        return True

    def _fan_out(self, vector, blocking: bool):
        self.sampled_count += 1
        for model_name, model_info in self.models.items():
            self._put(model_name, model_info, vector, blocking)

    # ----------------------------------------------------------------- main loops

    def _run_loop(self):
        """Main consumption loop - stride, record and fan vectors out to all models."""
        print("=" * 60)
        print("Starting message consumption")
        print("Press Ctrl+C to stop")
        print("=" * 60)
        print()

        last_report = time.time()

        try:
            while self.running:
                msg = self.consumer.poll(timeout=1.0)

                if msg is None:
                    continue

                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    else:
                        raise KafkaException(msg.error())

                vector = deserialize_vector(msg)
                if vector:
                    self._handle_vector(vector)
                    message_count = self.sampler.seen

                    if message_count % 10000 == 0:
                        elapsed = time.time() - last_report
                        rate = 10000 / elapsed if elapsed > 0 else 0

                        drop_summary = " | ".join(
                            [
                                f"{name}: {count:,} dropped"
                                for name, count in self.dropped_counts.items()
                                if count > 0
                            ]
                        )

                        if drop_summary:
                            print(
                                f"Processed {message_count:,} messages | Rate: {rate:.0f} msg/s | {drop_summary}"
                            )
                        else:
                            print(
                                f"Processed {message_count:,} messages | Rate: {rate:.0f} msg/s | No drops"
                            )

                        last_report = time.time()

        except KeyboardInterrupt:
            print("\n⚠ Shutdown signal received")
        except Exception as e:
            print(f"\n✗ Error in consumption loop: {e}")
            import traceback

            traceback.print_exc()
        finally:
            self._shutdown()

    def _run_replay(self, sample_file: str, total: int):
        """Feed the recorded vectors to the workers, in order, dropping nothing."""
        print("=" * 60)
        print("Starting replay")
        print("=" * 60)
        print()

        finished = False
        last_report = time.time()
        try:
            for vector in read_vector_sample(sample_file):
                if not self.running:
                    break
                self._fan_out(vector, blocking=True)
                n = self.sampled_count
                if n % 100000 == 0:
                    elapsed = time.time() - last_report
                    rate = 100000 / elapsed if elapsed > 0 else 0
                    print(
                        f"Replayed {n:,} / {total:,} vectors | Rate: {rate:.0f} vec/s"
                    )
                    last_report = time.time()
            finished = self.running
        except Exception as e:
            print(f"\n✗ Error in replay: {e}")
            import traceback

            traceback.print_exc()
        finally:
            if finished:
                print(f"\n✓ Replay complete: {self.sampled_count:,} vectors sent")
            else:
                print(f"\n⚠ Replay stopped early after {self.sampled_count:,} vectors")
            # on a complete replay wait for the workers to finish their backlog
            self._shutdown(drain=finished)

    def _report_summary(self):
        """Always printed at shutdown. In live mode it also checks the bookkeeping (every
        kept vector was recorded, and kept == consumed // stride) and, if a recording was
        made, writes it next to the file as vectors_sample.summary.json so that
        `make archive-run` can cross-check the archived sample against it."""
        print("\n" + "=" * 60)
        print("Run summary")
        print("=" * 60)
        if self.mode == "replay":
            print(f"Vectors replayed: {self.sampled_count:,}")
            print("=" * 60)
            return

        stride = self.sampler.stride
        consumed, kept = self.sampler.seen, self.sampled_count
        recorded = self.recorder.rows_written if self.recorder else None
        print(f"Vectors consumed from Kafka:      {consumed:,}")
        print(
            f"Vectors kept (stride {stride}):         {kept:,}  (consumed // {stride} = {consumed // stride:,})"
        )
        consistent = kept == consumed // stride
        if self.recorder:
            print(f"Vectors recorded to the file:     {recorded:,}")
            consistent = consistent and recorded == kept
        print(
            ("✓ " if consistent else "✗ MISMATCH: ")
            + "consumed, kept and recorded counts agree"
        )
        total_dropped = sum(self.dropped_counts.values())
        print(f"Vectors dropped by full queues:   {total_dropped:,}")
        for model_name, count in self.dropped_counts.items():
            if count > 0:
                rate = (count / kept) * 100 if kept else 0
                print(
                    f"  {model_name:15s}: {count:,} ({rate:.2f}% of kept vectors; replay this model from the sample)"
                )
        print("=" * 60)

        if self.recorder:
            summary = {
                "stride": stride,
                "consumed": consumed,
                "kept": kept,
                "recorded_rows": recorded,
                "consistent": consistent,
                "models": self.model_names,
                "dropped": self.dropped_counts,
            }
            with open(summary_path(self.recorder.path), "w") as f:
                json.dump(summary, f, indent=2)
            print(f"Summary written to {summary_path(self.recorder.path)}")

    def _signal_handler(self, signum, _frame):
        print(f"\n⚠ Received signal {signum}")
        self.running = False

    def _shutdown(self, drain: bool = False):
        """Gracefully shutdown all workers and consumer.

        drain=True (a completed replay): wait for every worker to score its whole backlog
        before it is asked to stop, however long that takes."""
        print()
        print("=" * 60)
        print("Shutting down multi-model service")
        print("=" * 60)

        # Send shutdown signal to all workers (None message)
        print("Stopping workers:")
        for model_name, model_info in self.models.items():
            try:
                if model_info["queue"]:
                    if drain:
                        self._put(model_name, model_info, None, blocking=True)
                    else:
                        # the queue may be full of vectors still to be scored
                        model_info["queue"].put(
                            None, block=True, timeout=LIVE_STOP_WAIT_SEC
                        )
                print(f"  ✓ Sent shutdown to {model_name}")
            except Exception as e:
                print(f"  ✗ Failed to signal {model_name}: {e}")

        print("\nWaiting for workers to finish:")
        for model_name, model_info in self.models.items():
            process = model_info["process"]
            if process and process.is_alive():
                print(f"  Waiting for {model_name}...")
                if drain:
                    process.join()
                else:
                    # a worker scores its whole backlog (up to a full queue of vectors)
                    # before it reaches the sentinel; `make stop-all` waits 120 s for us
                    process.join(timeout=LIVE_STOP_WAIT_SEC)

                if process.is_alive():
                    print(f"  ⚠ {model_name} did not stop, terminating...")
                    process.terminate()
                    process.join(timeout=2.0)

                if process.is_alive():
                    print(f"  ⚠ {model_name} still alive, killing...")
                    process.kill()
                    process.join()

                print(f"  ✓ {model_name} stopped")

        if self.recorder:
            print("\nClosing the vector recording...")
            self.recorder.close()

        self._report_summary()

        if self.consumer:
            print("\nClosing consumer...")
            self.consumer.close()
            print("  ✓ Consumer closed")

        print()
        print("=" * 60)
        print("Shutdown complete")
        print("=" * 60)


def select_models(cli_models, config_dict) -> list:
    """--models beats the `models:` list of the config beats DEFAULT_MODELS."""
    if cli_models:
        return [m.strip() for m in cli_models.split(",") if m.strip()]
    return list(config_dict.get("models") or DEFAULT_MODELS)


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Model Anomaly Detection Service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Live: read Kafka, score, and record the sampled vectors so other models can be run later
  python scripts/run_multi_model.py --record data/vectors/vectors_sample.parquet

  # Live with the models of the config (`models:` list) or explicitly
  python scripts/run_multi_model.py --config config/baselines.yaml --models zscore,isoforest

  # Replay a recorded sample through one more model (writes <dir of --output>/scores_rrcf.parquet)
  python scripts/run_multi_model.py --from-file data/vectors/vectors_sample.parquet \\
      --models rrcf --output results/thesis_x/inputs/scores.parquet

Models:
  rrcf, rrcf_forest, zscore, isoforest, halfspace, onlineiforest, and the ablation
  variants (rrcf_forest_{inst,fast,slow,nocusum,z2,raw2,inst_raw2}, zscore_{fast,slow,nocusum,z2,raw2})

Output:
  One parquet file per model next to --output: scores_<model>.parquet
""",
    )
    parser.add_argument(
        "--config",
        default="config/baselines.yaml",
        help="Path to YAML config file (default: config/baselines.yaml)",
    )
    parser.add_argument(
        "--output",
        default="./data/scores.parquet",
        help="Scores go to scores_<model>.parquet in the directory of this path "
        "(default: ./data/scores.parquet)",
    )
    parser.add_argument(
        "--models",
        help="Comma-separated models to run (default: the `models:` list of the config)",
    )
    parser.add_argument(
        "--record",
        metavar="FILE",
        help="Live mode: write the vectors that pass the stride to this parquet file",
    )
    parser.add_argument(
        "--from-file",
        metavar="FILE",
        help="Replay a recorded vectors_sample.parquet instead of reading Kafka",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help=f"Live mode: score every Nth vector (default {SAMPLE_STRIDE}; do not lower)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replay mode: allow overwriting existing scores_<model>.parquet files",
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"✗ Error: Config not found: {args.config}", file=sys.stderr)
        sys.exit(1)
    if args.from_file and (args.record or args.stride is not None):
        parser.error(
            "--from-file replays an already sampled file: no --record or --stride"
        )
    if args.from_file and not os.path.exists(args.from_file):
        print(f"✗ Error: sample file not found: {args.from_file}", file=sys.stderr)
        sys.exit(1)

    print("Loading configuration...")
    with open(args.config, "r") as f:
        config_dict = yaml.safe_load(f)

    service_config = ServiceConfig.from_dict(config_dict)
    print(f"✓ Loaded config from {args.config}")
    print(f"✓ Output file: {args.output}")

    model_names = select_models(args.models, config_dict)
    registry = model_registry(service_config.partitioner_config.detector_config)
    unknown = [m for m in model_names if m not in registry]
    if unknown:
        print(
            f"✗ Error: unknown model(s) {unknown}; available: {sorted(registry)}",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"✓ Models: {', '.join(model_names)}")
    print()

    runner = MultiModelRunner(
        service_config,
        parquet_file=args.output,
        model_names=model_names,
        stride=args.stride if args.stride is not None else SAMPLE_STRIDE,
        record_file=args.record,
        overwrite=args.overwrite,
    )
    if args.from_file:
        runner.start_replay(args.from_file)
    else:
        runner.start()


if __name__ == "__main__":
    main()
