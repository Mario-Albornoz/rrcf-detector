"""
The sampled vector stream: choose it once, record it, replay it.

The models do not ingest every normalized vector, only every STRIDE-th one (a CPU
workaround, see scripts/run_multi_model.py). Which vectors those are depends on the order
in which one consumer sees the messages of the topic's partitions, and that order changes
from one consumption to the next. Two runs of the pipeline therefore score different
samples of the same stream, and their scores cannot be compared row by row.

The fix is to make that choice exactly once. The runner applies the stride to the stream it
reads from Kafka (StrideSampler), hands the surviving vectors to every model, and can record
them (VectorSampleWriter). A later run reads the recorded file instead of Kafka
(read_vector_sample) and feeds the models the same vectors in the same order, so any model,
in any run, scores the same rows.

File layout (Parquet, zstd): one row per vector, in ingestion order. `stream_index` is the
1-based position of the vector in the full consumed stream (a multiple of the stride).
Timestamps are stored as integer microseconds since the epoch (UTC), which is what
datetime holds, so a vector read back is field-for-field equal to the one that was written.
"""

import datetime as dt
import os
from pathlib import Path
from typing import Iterator, Optional

import pyarrow as pa
import pyarrow.parquet as pq

from src.kafka.consumer import NormalizedVectorDto

SAMPLE_SCHEMA = pa.schema(
    [
        ("stream_index", pa.int64()),
        ("exchange", pa.string()),
        ("instrument", pa.string()),
        ("instrument_class", pa.string()),
        ("timestamp_us", pa.int64()),
        ("model_key", pa.string()),
        ("z_intertick_fast", pa.float64()),
        ("z_price_step_fast", pa.float64()),
        ("z_intertick_slow", pa.float64()),
        ("z_price_step_slow", pa.float64()),
        ("cusum_intertick", pa.float64()),
        ("cusum_price_step", pa.float64()),
        ("gap_flag", pa.int64()),
        ("warmup_flag", pa.int64()),
        ("session_fallback_flag", pa.int64()),
        ("seq", pa.int64()),
    ]
)

_EPOCH = dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc)
_ONE_US = dt.timedelta(microseconds=1)


def _to_us(ts: dt.datetime) -> int:
    if ts.tzinfo is None:  # the handler always sends a zone; a naive time is taken as UTC
        ts = ts.replace(tzinfo=dt.timezone.utc)
    return (ts - _EPOCH) // _ONE_US  # exact integer arithmetic, no float rounding


def _from_us(us: int) -> dt.datetime:
    return _EPOCH + dt.timedelta(microseconds=us)


class StrideSampler:
    """Keeps every `stride`-th item of a stream: the 1st..(stride-1)-th are skipped, the
    stride-th is kept, and so on. `seen` counts every item offered, kept or not."""

    def __init__(self, stride: int = 10):
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")
        self.stride = stride
        self.seen = 0

    def take(self) -> bool:
        self.seen += 1
        return self.seen % self.stride == 0


class VectorSampleWriter:
    """Buffered Parquet writer for the sampled vectors.

    Rows go to `<path>.partial` and the file is renamed to `<path>` only by close(), once
    the Parquet footer is written. A file at `path` is therefore always complete; a run that
    was killed leaves only the .partial file behind."""

    def __init__(self, path: str, buffer_rows: int = 50_000):
        self.path = str(path)
        self.partial_path = self.path + ".partial"
        self.buffer_rows = buffer_rows
        self.rows_written = 0
        self._writer: Optional[pq.ParquetWriter] = None
        self._cols = self._empty()
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _empty() -> dict:
        return {name: [] for name in SAMPLE_SCHEMA.names}

    def write(self, stream_index: int, v: NormalizedVectorDto) -> None:
        c = self._cols
        c["stream_index"].append(stream_index)
        c["exchange"].append(v.exchange)
        c["instrument"].append(v.instrument)
        c["instrument_class"].append(v.instrument_class)
        c["timestamp_us"].append(_to_us(v.timestamp))
        c["model_key"].append(v.model_key)
        c["z_intertick_fast"].append(v.z_intertick_fast)
        c["z_price_step_fast"].append(v.z_price_step_fast)
        c["z_intertick_slow"].append(v.z_intertick_slow)
        c["z_price_step_slow"].append(v.z_price_step_slow)
        c["cusum_intertick"].append(v.cusum_intertick)
        c["cusum_price_step"].append(v.cusum_price_step)
        c["gap_flag"].append(v.gap_flag)
        c["warmup_flag"].append(v.warmup_flag)
        c["session_fallback_flag"].append(v.session_fallback_flag)
        c["seq"].append(v.seq)
        if len(c["seq"]) >= self.buffer_rows:
            self.flush()

    def flush(self) -> None:
        n = len(self._cols["seq"])
        if n == 0:
            return
        if self._writer is None:
            self._writer = pq.ParquetWriter(
                self.partial_path, SAMPLE_SCHEMA, compression="zstd", use_dictionary=True
            )
        self._writer.write_table(pa.Table.from_pydict(self._cols, schema=SAMPLE_SCHEMA))
        self.rows_written += n
        self._cols = self._empty()

    def close(self) -> None:
        self.flush()
        if self._writer is None:  # nothing was recorded: still leave a valid, empty file
            self._writer = pq.ParquetWriter(self.partial_path, SAMPLE_SCHEMA, compression="zstd")
        self._writer.close()
        self._writer = None
        os.replace(self.partial_path, self.path)
        print(f"[VectorSampleWriter] {self.rows_written:,} vectors recorded to {self.path}")


def summary_path(sample_path: str) -> str:
    """Where the runner writes its own account of a recording: vectors_sample.summary.json
    next to vectors_sample.parquet (what it consumed, kept and recorded)."""
    return str(Path(sample_path).with_suffix(".summary.json"))


def sample_row_count(path: str) -> int:
    return pq.ParquetFile(path).metadata.num_rows


def read_vector_sample(path: str, batch_rows: int = 50_000) -> Iterator[NormalizedVectorDto]:
    """Yield the recorded vectors in the order they were written."""
    pf = pq.ParquetFile(path)
    if pf.metadata.num_row_groups == 0:  # an empty recording; iter_batches raises on it
        return
    for batch in pf.iter_batches(batch_size=batch_rows):
        c = batch.to_pydict()
        for i in range(batch.num_rows):
            yield NormalizedVectorDto(
                exchange=c["exchange"][i],
                instrument=c["instrument"][i],
                instrument_class=c["instrument_class"][i],
                timestamp=_from_us(c["timestamp_us"][i]),
                model_key=c["model_key"][i],
                z_intertick_fast=c["z_intertick_fast"][i],
                z_price_step_fast=c["z_price_step_fast"][i],
                z_intertick_slow=c["z_intertick_slow"][i],
                z_price_step_slow=c["z_price_step_slow"][i],
                cusum_intertick=c["cusum_intertick"][i],
                cusum_price_step=c["cusum_price_step"][i],
                gap_flag=c["gap_flag"][i],
                warmup_flag=c["warmup_flag"][i],
                session_fallback_flag=c["session_fallback_flag"][i],
                seq=c["seq"][i],
            )
