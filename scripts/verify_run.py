#!/usr/bin/env python3
"""
Verify a pipeline run: what went right, what went wrong, and where.

Reads whatever a run produced (the simulator's ground truth and manifest, the two Kafka
topics, the feed-handler's evaluation logs, the detector's scores) and checks each stage
against what the design says should have happened. Each check prints PASS, WARN or FAIL
with what was observed, what was expected, and a hint about the usual cause. Exit status
is 1 if anything failed, so it can gate a long evaluation.

It is meant to be run right after a run and before the evaluation: a broken input makes
every later number meaningless, and this says so in a minute rather than after days.

    venv/bin/python scripts/verify_run.py  # from rrcf-detector/ \\
        --manifest price-feed-simulator/data/injection_manifest.json \\
        --episodes price-feed-simulator/anomaly_log_episodes.csv \\
        --instruments price-feed-simulator/anomaly_log_instruments.csv \\
        --silence-log feed-handler/data/eval/silence_alerts.csv \\
        --validation-log feed-handler/data/eval/validation_alerts.csv \\
        --scores rrcf-detector/data/scores_rrcf.parquet \\
        --kafka localhost:9092 --raw-topic raw-ticks --vector-topic normalized-vectors

Every input is optional; checks whose input is missing are reported as SKIP.
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

STRIDE = 10  # the worker scores one vector in ten

PASS, WARN, FAIL, SKIP = "PASS", "WARN", "FAIL", "SKIP"


class Report:
    def __init__(self):
        self.rows: List[Dict] = []

    def add(self, stage: str, name: str, status: str, observed: str, expected: str = "", hint: str = ""):
        self.rows.append({"stage": stage, "check": name, "status": status,
                          "observed": observed, "expected": expected, "hint": hint})

    def within(self, stage, name, value, ok, warn, expected, hint, fmt="{:.4g}", value_text=None):
        """PASS if ok(value), WARN if warn(value), else FAIL."""
        text = value_text if value_text is not None else fmt.format(value)
        status = PASS if ok(value) else (WARN if warn(value) else FAIL)
        self.add(stage, name, status, text, expected, "" if status == PASS else hint)

    def failed(self) -> int:
        return sum(1 for r in self.rows if r["status"] == FAIL)

    def print(self):
        icon = {PASS: "PASS", WARN: "WARN", FAIL: "FAIL", SKIP: "skip"}
        stage = None
        for r in self.rows:
            if r["stage"] != stage:
                stage = r["stage"]
                print(f"\n== {stage}")
            print(f"  [{icon[r['status']]}] {r['check']}: {r['observed']}")
            if r["status"] in (WARN, FAIL):
                if r["expected"]:
                    print(f"         expected: {r['expected']}")
                if r["hint"]:
                    print(f"         hint:     {r['hint']}")
        counts = {s: sum(1 for r in self.rows if r["status"] == s) for s in (PASS, WARN, FAIL, SKIP)}
        print(f"\n{counts[PASS]} passed, {counts[WARN]} warnings, {counts[FAIL]} FAILED, {counts[SKIP]} skipped")


# ------------------------------------------------------------------------ ground truth


def check_ground_truth(rep: Report, manifest: Optional[dict], episodes: Optional[pd.DataFrame],
                       instruments: Optional[pd.DataFrame]) -> None:
    stage = "Ground truth (simulator)"

    if manifest is None:
        rep.add(stage, "injection manifest", SKIP, "not given")
    else:
        stats = manifest.get("stats", {})
        rep.within(stage, "messages processed by the simulator", stats.get("TotalProcessed", 0),
                   lambda v: v > 0, lambda v: False, "> 0",
                   "the simulator read no ticks: check data_dir / file_pattern", fmt="{:,.0f}")

    if episodes is None:
        rep.add(stage, "episode file", SKIP, "not given")
        return

    rep.within(stage, "episode file has rows", len(episodes), lambda v: v > 0, lambda v: False, "> 0",
               "no anomaly was injected at all: date_filter, window, exchange_filter or probabilities",
               fmt="{:,.0f}")

    if not len(episodes) or "Phase" not in episodes.columns:
        return

    phases = {1: "phase1", 2: "phase2", 3: "phase3", 4: "phase4"}
    enabled = {}
    if manifest is not None:
        for p, key in phases.items():
            info = manifest.get("phases", {}).get(key, {})
            enabled[p] = bool(info.get("enabled"))
    counts = episodes["Phase"].str.extract(r"(\d+)", expand=False).astype(int).value_counts().to_dict()
    for p, key in phases.items():
        n = int(counts.get(p, 0))
        if manifest is not None and not enabled.get(p, False):
            rep.add(stage, f"{key} episodes", SKIP, f"{n:,} (phase disabled)")
            continue
        hint = {
            1: "no instrument was selected or no tick fell in the window on the date filter",
            2: "check date_filter/window, and that prices are on trade rows (Last > 0)",
            3: "check exchange_filter: Exchange is the ID suffix (ETR, FR, NL), not the venue name",
            4: "check date_filter/window and the strategy probabilities",
        }[p]
        rep.within(stage, f"{key} episodes injected", n, lambda v: v > 0, lambda v: False, "> 0", hint, fmt="{:,.0f}")

    if "SecType" in episodes.columns:
        eq = float((episodes["SecType"] == "E").mean())
        rep.within(stage, "share of episodes on equities", eq, lambda v: v > 0.5, lambda v: v > 0.2,
                   "most episodes on equities (indices never reach the detector)",
                   "the injector is hitting index rows: check the instrument selection", fmt="{:.1%}")
    else:
        rep.add(stage, "episode file has SecType", FAIL, "column missing",
                "SecType column", "the evaluation cannot drop index episodes: update the simulator")

    if instruments is None:
        rep.add(stage, "instrument-day summary", SKIP, "not given (results cannot be stratified by activity)")
    else:
        eq = instruments[instruments["SecType"] == "E"]
        rows, trades = eq["Rows"].sum(), eq["Trades"].sum()
        share = trades / rows if rows else 0.0
        rep.within(stage, "trades as a share of equity rows", share, lambda v: 0.02 <= v <= 0.08,
                   lambda v: 0 < v < 0.2, "about 4% (4.2% measured on the DEBS files)",
                   "0% means prices are lost before the simulator's injector (parser column); "
                   "far above 4% means the parser reads the wrong column", fmt="{:.2%}")
        rep.add(stage, "days in the run", PASS, f"{sorted(eq['Date'].unique())}")


# ------------------------------------------------------------------------------ Kafka


def _consumer(brokers: str):
    from confluent_kafka import Consumer

    return Consumer({"bootstrap.servers": brokers, "group.id": "verify-run", "enable.auto.commit": False,
                     "auto.offset.reset": "earliest"})


def topic_offsets(brokers: str, topic: str) -> Dict[int, tuple]:
    """partition -> (low, high) offsets."""
    c = _consumer(brokers)
    try:
        md = c.list_topics(topic, timeout=10)
        if topic not in md.topics or md.topics[topic].error is not None:
            return {}
        out = {}
        from confluent_kafka import TopicPartition

        for p in md.topics[topic].partitions:
            low, high = c.get_watermark_offsets(TopicPartition(topic, p), timeout=10)
            out[p] = (low, high)
        return out
    finally:
        c.close()


def sample_topic(brokers: str, topic: str, per_partition: int = 4000) -> List[dict]:
    """The last `per_partition` messages of each partition, decoded as JSON, in offset order."""
    from confluent_kafka import TopicPartition

    offsets = topic_offsets(brokers, topic)
    c = _consumer(brokers)
    rows: List[dict] = []
    try:
        for p, (low, high) in offsets.items():
            if high <= low:
                continue
            start = max(low, high - per_partition)
            c.assign([TopicPartition(topic, p, start)])
            got = 0
            idle = 0
            while got < high - start and idle < 5:
                m = c.poll(1.0)
                if m is None:
                    idle += 1
                    continue
                if m.error():
                    continue
                idle = 0
                try:
                    d = json.loads(m.value())
                except ValueError:
                    continue
                d["_partition"], d["_offset"] = p, m.offset()
                rows.append(d)
                got += 1
    finally:
        c.close()
    return rows


def check_kafka(rep: Report, brokers: str, raw_topic: Optional[str], vector_topic: Optional[str],
                manifest: Optional[dict]) -> Dict[str, int]:
    stage = "Kafka"
    totals: Dict[str, int] = {}
    try:
        if raw_topic:
            offs = topic_offsets(brokers, raw_topic)
            if not offs:
                rep.add(stage, f"topic {raw_topic}", FAIL, "does not exist", "the simulator's topic",
                        "was the simulator run against this topic name?")
            else:
                raw_count = sum(h - l for l, h in offs.values())
                totals["raw"] = raw_count
                if manifest is not None:
                    s = manifest.get("stats", {})
                    expected = s.get("TotalProcessed", 0) - s.get("Phase1Dropped", 0) - s.get("Phase3Dropped", 0)
                    ratio = raw_count / expected if expected else float("nan")
                    if ratio > 1.001:
                        rep.add(stage, f"messages on {raw_topic}", WARN,
                                f"{raw_count:,} vs {expected:,} expected ({ratio:.2f}x)", "about 1.0x",
                                "the topic holds messages from earlier runs: delete it before a run "
                                "(they inflate every count and replay old data)")
                    else:
                        rep.within(stage, f"messages on {raw_topic} vs simulator", ratio,
                                   lambda v: 0.999 <= v <= 1.001, lambda v: v >= 0.99,
                                   "1.000 (published = processed - dropped by injection)",
                                   "messages were lost between the simulator and Kafka (the writer is asynchronous: "
                                   "check the simulator's 'acknowledged by Kafka' count)",
                                   fmt="{:.4f}", value_text=f"{raw_count:,} of {expected:,} ({ratio:.4f})")
                else:
                    rep.add(stage, f"messages on {raw_topic}", PASS, f"{raw_count:,}")
                _check_raw_sample(rep, brokers, raw_topic)
        if vector_topic:
            offs = topic_offsets(brokers, vector_topic)
            if not offs:
                rep.add(stage, f"topic {vector_topic}", FAIL, "does not exist", "the feed-handler's output topic",
                        "did the feed-handler start with this output_topic?")
            else:
                totals["vectors"] = sum(h - l for l, h in offs.values())
                if "raw" in totals and totals["raw"]:
                    r = totals["vectors"] / totals["raw"]
                    rep.within(stage, "vectors per raw message", r, lambda v: 0.65 <= v <= 0.95,
                               lambda v: 0.4 <= v <= 1.0,
                               "about 0.8 (equities are about 82% of rows; index rows and rejected messages emit nothing)",
                               "far below: the handler is lagging or dropping (it may still be running), or the wrong "
                               "SecType filter; above 0.95: index rows are not being filtered",
                               fmt="{:.3f}", value_text=f"{totals['vectors']:,} / {totals['raw']:,} = {r:.3f}")
                else:
                    rep.add(stage, f"vectors on {vector_topic}", PASS, f"{totals['vectors']:,}")
                _check_vector_sample(rep, brokers, vector_topic)
    except Exception as e:  # noqa: BLE001 - a broker problem is itself the finding
        rep.add(stage, "Kafka reachable", FAIL, f"{type(e).__name__}: {e}", "a broker at --kafka",
                "is docker compose up?")
    return totals


def _check_raw_sample(rep: Report, brokers: str, topic: str) -> None:
    stage = "Kafka: raw messages (simulator's wire format)"
    rows = sample_topic(brokers, topic)
    if not rows:
        rep.add(stage, "sample", SKIP, "no messages")
        return
    df = pd.DataFrame(rows)
    eq = df[df.get("SecType", "E") == "E"]
    if "Last" not in df.columns:
        rep.add(stage, "price field 'Last'", FAIL, "absent from the messages", "a 'Last' field",
                "the simulator changed its wire format: the feed-handler reads 'Last'")
        return
    share = float((eq["Last"] > 0).mean()) if len(eq) else 0.0
    rep.within(stage, "equity messages with a last price", share, lambda v: 0.01 <= v <= 0.12, lambda v: v > 0,
               "about 4% of equity rows are trades",
               "0% means the price is lost in the simulator; the parser may be reading the wrong column",
               fmt="{:.2%}", value_text=f"{share:.2%} of {len(eq):,} sampled")

    # per-instrument order inside a partition: a producer that reorders breaks the validator
    if {"ID", "Time", "_partition", "_offset"} <= set(df.columns):
        d = df.sort_values(["_partition", "_offset"]).copy()
        d["t"] = pd.to_datetime(d["Time"], utc=True, errors="coerce")
        back = d.groupby("ID")["t"].diff().dt.total_seconds()
        rate = float((back < -1.0).mean())
        rep.within(stage, "per-instrument time going back by more than 1 s", rate, lambda v: v < 1e-4,
                   lambda v: v < 1e-3, "about 0 (one producer worker per instrument)",
                   "the producer is reordering an instrument's messages; the validator will reject valid data",
                   fmt="{:.4%}")


def _check_vector_sample(rep: Report, brokers: str, topic: str) -> None:
    stage = "Kafka: feature vectors"
    rows = sample_topic(brokers, topic)
    if not rows:
        rep.add(stage, "sample", SKIP, "no messages")
        return
    df = pd.DataFrame(rows)
    need = ["has_trade", "z_price_step_fast", "z_price_step_slow", "z_intertick_fast", "z_intertick_slow",
            "cusum_intertick", "cusum_price_step", "warmup_flag"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        rep.add(stage, "vector fields", FAIL, f"missing {missing}", "the fields the detector reads",
                "the feed-handler and detector disagree on the vector format")
        return

    trades = df[df["has_trade"] == 1]
    share = len(trades) / len(df)
    rep.within(stage, "vectors flagged as trades", share, lambda v: 0.01 <= v <= 0.12, lambda v: v > 0,
               "about 4%",
               "0% means no price reaches the features: the handler read every Last as 0 "
               "(wire-format field name?)", fmt="{:.2%}", value_text=f"{share:.2%} of {len(df):,} sampled")

    quotes = df[df["has_trade"] == 0]
    if len(quotes):
        leak = float(((quotes["z_price_step_fast"] != 0) | (quotes["z_price_step_slow"] != 0)).mean())
        rep.within(stage, "quote rows with a price z-score", leak, lambda v: v == 0, lambda v: v < 0.01,
                   "0 (a quote update says nothing about price)", "price statistics are updating on quote rows",
                   fmt="{:.3%}")
    if len(trades) >= 50:
        warm = trades[trades["z_price_step_slow"].abs() > 0]
        rep.within(stage, "trade rows with a non-zero price z-score", len(warm) / len(trades),
                   lambda v: v >= 0.4, lambda v: v >= 0.1,
                   "most trades (after the first two of each instrument)",
                   "price z-scores are constant zero: the price never varies (or is 0)", fmt="{:.1%}")

    numeric = df[[c for c in need if c != "warmup_flag"]].astype(float)
    bad = int((~np.isfinite(numeric.to_numpy())).sum())
    rep.within(stage, "non-finite feature values (NaN/Inf)", bad, lambda v: v == 0, lambda v: False, "0",
               "a division or a statistic is producing NaN", fmt="{:,.0f}")
    z = numeric["z_intertick_slow"].abs()
    rep.within(stage, "slow timing z-score beyond 3", float((z > 3).mean()), lambda v: v < 0.05, lambda v: v < 0.15,
               "a few percent (about 1-2% measured)",
               "far more: the statistics are mis-calibrated (warm-up, overnight gaps, clock)", fmt="{:.1%}")


# ------------------------------------------------------------------ feed-handler logs


def check_handler_logs(rep: Report, silence: Optional[pd.DataFrame], validation: Optional[pd.DataFrame],
                       episodes: Optional[pd.DataFrame], vectors: Optional[int]) -> None:
    stage = "Feed-handler evaluation logs"
    if silence is None:
        rep.add(stage, "silence log", SKIP, "not given")
    else:
        if vectors:
            per_1000 = 1000 * len(silence) / vectors
            rep.within(stage, "silence alerts per 1000 vectors", per_1000, lambda v: v <= 5, lambda v: v <= 20,
                       "about 1 (the quantile rule)",
                       "far more: the rule is firing on ordinary gaps (mean-multiple rule? clock jitter? overnight gap?)",
                       fmt="{:.2f}", value_text=f"{len(silence):,} alerts, {per_1000:.2f} per 1000 vectors")
        else:
            rep.add(stage, "silence alerts", PASS, f"{len(silence):,}")
        if len(silence) and "Trigger" in silence.columns:
            t = silence["Trigger"].value_counts().to_dict()
            rep.add(stage, "silence triggers", PASS, str(t))
            flush = t.get("flush", 0)
            rep.within(stage, "end-of-stream (flush) alerts", flush,
                       lambda v: v <= max(50, 0.05 * len(silence)), lambda v: False,
                       "few (silences still open at shutdown)",
                       "many instruments were silent at the end: the run may have stopped mid-day", fmt="{:,.0f}")

    if validation is None:
        rep.add(stage, "validation log", SKIP, "not given")
    else:
        by_type = validation["AlertType"].value_counts().to_dict() if len(validation) else {}
        rep.add(stage, "validation alerts by type", PASS, str(by_type))
        if vectors and len(validation):
            inj = 0
            if episodes is not None:
                inj = int((episodes["AnomalyType"] == "timestamp_inversion").sum())
            unexplained = max(len(validation) - inj, 0)
            rate = unexplained / vectors
            rep.within(stage, "validation alerts beyond the injected inversions", rate,
                       lambda v: v < 1e-4, lambda v: v < 1e-3,
                       "about 0 on real data (natural backward steps are under 1 s)",
                       "valid messages are being rejected: producer reordering, or a clock/tolerance problem",
                       fmt="{:.4%}", value_text=f"{unexplained:,} of {len(validation):,} alerts ({rate:.4%} of vectors)")


# -------------------------------------------------------------------------- detector


def check_scores(rep: Report, scores: Optional[pd.DataFrame], vectors: Optional[int],
                 episodes: Optional[pd.DataFrame]) -> None:
    stage = "Detector scores"
    if scores is None:
        rep.add(stage, "scores file", SKIP, "not given")
        return

    rep.within(stage, "score rows", len(scores), lambda v: v > 0, lambda v: False, "> 0",
               "the detector wrote nothing: did it consume the vector topic (group offsets, topic name)?",
               fmt="{:,.0f}")
    if not len(scores):
        return

    need = {"exchange", "instrument", "timestamp_ms", "z_score"}
    if not need <= set(scores.columns):
        rep.add(stage, "score columns", FAIL, f"missing {sorted(need - set(scores.columns))}", str(sorted(need)),
                "the evaluation needs these")
        return

    if vectors:
        expected = vectors / STRIDE
        r = len(scores) / expected
        rep.within(stage, "scores vs vectors / stride", r, lambda v: 0.85 <= v <= 1.05, lambda v: 0.5 <= v <= 1.2,
                   "about 1.0 (one vector in ten is scored, after each model's warm-up)",
                   "far below: the detector lagged, was stopped early, or workers crashed; above: duplicates",
                   fmt="{:.3f}", value_text=f"{len(scores):,} vs {expected:,.0f} ({r:.3f})")

    z = scores["z_score"].to_numpy(dtype=float)
    rep.within(stage, "non-finite scores", int((~np.isfinite(z)).sum()), lambda v: v == 0, lambda v: False, "0",
               "the detector emitted NaN/Inf", fmt="{:,.0f}")
    alert_rate = float((z >= 2.0).mean())
    rep.within(stage, "share of scores with z >= 2", alert_rate, lambda v: 0.005 <= v <= 0.08, lambda v: 0 < v < 0.2,
               "a few percent (about 2.3% for a normal z)",
               "0: scores are constant (features all zero?); very high: the score scaling is off", fmt="{:.2%}")
    rep.within(stage, "spread of z-scores", float(np.nanstd(z)), lambda v: v > 0.3, lambda v: v > 0.05,
               "a standard deviation near 1", "constant scores carry no information", fmt="{:.3f}")

    if episodes is not None and len(episodes):
        eq = episodes[episodes.get("SecType", "E") == "E"] if "SecType" in episodes.columns else episodes
        ep_keys = set(eq["Exchange"].astype(str) + "|" + eq["InstrumentID"].astype(str))
        sc_keys = set(scores["exchange"].astype(str) + "|" + scores["instrument"].astype(str))
        overlap = len(ep_keys & sc_keys) / len(ep_keys) if ep_keys else 0.0
        rep.within(stage, "episode instruments present in the scores", overlap, lambda v: v >= 0.8, lambda v: v >= 0.3,
                   "most (instruments that never produced a scored vector are missing)",
                   "0 means the naming differs ('exchange|instrument' keys) or the detector saw other data",
                   fmt="{:.1%}")


def check_join(rep: Report, scores: Optional[pd.DataFrame], episodes: Optional[pd.DataFrame]) -> None:
    """Do the scores line up with the ground truth in time? The exact-tick match is the most
    fragile join in the evaluation."""
    stage = "Scores meet ground truth"
    if scores is None or episodes is None or not len(scores) or not len(episodes):
        rep.add(stage, "exact-tick scorable share", SKIP, "needs scores and episodes")
        return

    point = episodes[episodes["AnomalyType"].isin(["price_spike", "price_deviation", "implausible_price"])]
    if "SecType" in point.columns:
        point = point[point["SecType"] == "E"]
    if not len(point):
        rep.add(stage, "exact-tick scorable share", SKIP, "no point episodes")
        return
    if len(point) > 200_000:
        point = point.sample(200_000, random_state=0)

    # exact join on the message sequence number when both sides carry it
    if "seq" in scores.columns and scores["seq"].gt(0).any() and "Seq" in point.columns and point["Seq"].notna().all():
        share = float(point["Seq"].astype("int64").isin(set(scores["seq"].astype("int64"))).mean())
        rep.within(stage, "injected messages whose own vector was scored (by sequence number)", share,
                   lambda v: 0.05 <= v <= 0.2, lambda v: 0 < v < 0.5, "about 10% (the stride)",
                   "0% means the sequence numbers are lost between the simulator and the scores "
                   "(feed-handler vector 'seq', detector DTO and parquet column)",
                   fmt="{:.1%}", value_text=f"{share:.1%} of {len(point):,} episodes")
        return

    s_key = (scores["exchange"].astype(str) + "|" + scores["instrument"].astype(str)).to_numpy()
    s_ts = scores["timestamp_ms"].to_numpy(dtype=np.int64)
    scored = set(zip(s_key.tolist(), s_ts.tolist()))
    e_key = (point["Exchange"].astype(str) + "|" + point["InstrumentID"].astype(str)).to_numpy()
    e_ts = point["ObservedMs"].fillna(point["StartMs"]).to_numpy(dtype=np.int64)
    hit = sum(1 for k, t in zip(e_key.tolist(), e_ts.tolist())
              if (k, t) in scored or (k, t - 1) in scored or (k, t + 1) in scored)
    share = hit / len(point)
    rep.within(stage, "injected trades whose own vector was scored", share, lambda v: 0.05 <= v <= 0.2,
               lambda v: 0 < v < 0.5, "about 10% (the stride)",
               "0% means the timestamps do not line up (score timestamp_ms vs the episode's ObservedMs): "
               "the evaluation's exact-tick matching would find nothing",
               fmt="{:.1%}", value_text=f"{share:.1%} of {len(point):,} episodes")


# ---------------------------------------------------------------------------- driver


def _read_csv(path: Optional[str]) -> Optional[pd.DataFrame]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return pd.read_csv(p, dtype={"Detail": "string", "Date": str})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", help="injection_manifest.json from the simulator")
    ap.add_argument("--episodes", help="anomaly_log_episodes.csv")
    ap.add_argument("--instruments", help="anomaly_log_instruments.csv")
    ap.add_argument("--silence-log")
    ap.add_argument("--validation-log")
    ap.add_argument("--scores", help="scores parquet")
    ap.add_argument("--kafka", help="bootstrap servers, e.g. localhost:9092 (enables the Kafka checks)")
    ap.add_argument("--raw-topic", default="raw-ticks")
    ap.add_argument("--vector-topic", default="normalized-vectors")
    ap.add_argument("--json", help="also write the report as JSON")
    args = ap.parse_args(argv)

    manifest = None
    if args.manifest and Path(args.manifest).exists():
        manifest = json.loads(Path(args.manifest).read_text())
    episodes = _read_csv(args.episodes)
    instruments = _read_csv(args.instruments)
    silence = _read_csv(args.silence_log)
    validation = _read_csv(args.validation_log)
    scores = None
    if args.scores and Path(args.scores).exists():
        import pyarrow.parquet as pq

        wanted = ["exchange", "instrument", "timestamp_ms", "z_score"]
        if "seq" in pq.ParquetFile(args.scores).schema_arrow.names:
            wanted.append("seq")
        scores = pd.read_parquet(args.scores, columns=wanted)

    rep = Report()
    check_ground_truth(rep, manifest, episodes, instruments)

    totals: Dict[str, int] = {}
    if args.kafka:
        totals = check_kafka(rep, args.kafka, args.raw_topic, args.vector_topic, manifest)
    else:
        rep.add("Kafka", "topic checks", SKIP, "pass --kafka to enable")

    check_handler_logs(rep, silence, validation, episodes, totals.get("vectors"))
    check_scores(rep, scores, totals.get("vectors"), episodes)
    check_join(rep, scores, episodes)

    rep.print()
    if args.json:
        Path(args.json).write_text(json.dumps(rep.rows, indent=2))
    return 1 if rep.failed() else 0


if __name__ == "__main__":
    sys.exit(main())
