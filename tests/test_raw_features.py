"""The raw measurements behind the z-scores travel from the feed handler's JSON through the
recorded vector sample, so an ablation can run models on un-normalized features. Recordings
and messages from before they were added still work, with the raw fields at their defaults."""

import dataclasses
import datetime as dt

import orjson
import pyarrow as pa
import pyarrow.parquet as pq

from src.detection.vector_sample import (
    RAW_DEFAULTS,
    SAMPLE_SCHEMA,
    VectorSampleWriter,
    read_vector_sample,
)
from src.kafka.consumer import NormalizedVectorDto, deserialize_vector

VECTOR = {
    "exchange": "ETR", "instrument": "SAP.ETR", "class": "E", "timestamp": "2021-11-10T10:00:00.412Z",
    "model_key": "E", "z_intertick_fast": 0.1, "z_price_step_fast": 0.2, "z_intertick_slow": 0.3,
    "z_price_step_slow": 0.4, "cusum_intertick": 0.0, "cusum_price_step": 0.0, "gap_flag": 0,
    "warmup_flag": 0, "session_fallback_flag": 0, "seq": 42,
}
RAW = {"has_trade": 1, "intertick_ms": 2000.0, "has_intertick": 1, "price_step": 0.5,
       "has_price_step": 1, "ref_price": 100.0}


class _Msg:
    def __init__(self, payload):
        self._v = orjson.dumps(payload)

    def value(self):
        return self._v


def _dto(k, **raw):
    return NormalizedVectorDto(
        exchange="FR", instrument=f"INS{k}.FR", instrument_class="E",
        timestamp=dt.datetime(2021, 11, 8, 9, 0, k, tzinfo=dt.timezone.utc), model_key="FR:E",
        z_intertick_fast=0.1 * k, z_price_step_fast=0.2, z_intertick_slow=0.3, z_price_step_slow=0.4,
        cusum_intertick=1.0, cusum_price_step=2.0, gap_flag=0, warmup_flag=0,
        session_fallback_flag=0, seq=k + 1, **raw,
    )


def test_raw_fields_are_decoded_from_the_handler_message():
    v = deserialize_vector(_Msg({**VECTOR, **RAW}))
    for name, value in RAW.items():
        assert getattr(v, name) == value


def test_a_message_from_an_older_handler_decodes_with_defaults():
    v = deserialize_vector(_Msg(VECTOR))
    assert v is not None and v.seq == 42
    for name, value in RAW_DEFAULTS.items():
        assert getattr(v, name) == value


def test_raw_fields_are_recorded_and_replayed_exactly(tmp_path):
    vectors = [_dto(k, **{**RAW, "intertick_ms": 1000.0 * k, "ref_price": 100.0 + k}) for k in range(20)]
    path = tmp_path / "s.parquet"
    w = VectorSampleWriter(str(path), buffer_rows=7)
    for i, v in enumerate(vectors, start=1):
        w.write(i, v)
    w.close()
    assert set(RAW_DEFAULTS) <= set(pq.ParquetFile(path).schema_arrow.names)
    assert list(read_vector_sample(str(path), batch_rows=6)) == vectors


def test_a_recording_without_raw_columns_still_replays(tmp_path):
    """Recordings of earlier runs have no raw columns; replaying them must keep working."""
    vectors = [_dto(k) for k in range(10)]
    old_schema = pa.schema([f for f in SAMPLE_SCHEMA if f.name not in RAW_DEFAULTS])
    rows = {name: [] for name in old_schema.names}
    for i, v in enumerate(vectors, start=1):
        d = dataclasses.asdict(v)
        d["stream_index"] = i
        d["timestamp_us"] = int(v.timestamp.timestamp() * 1_000_000)
        for name in old_schema.names:
            rows[name].append(d[name])
    path = tmp_path / "old.parquet"
    pq.write_table(pa.Table.from_pydict(rows, schema=old_schema), path)

    replayed = list(read_vector_sample(str(path)))
    assert replayed == vectors   # the raw fields come back at their defaults
