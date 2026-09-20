#!/usr/bin/env python3
"""
Thesis evaluation: RQ1 (four-phase feed-degradation detection) for one detector.

Ground truth is the simulator's episode file (one row per injected anomaly episode,
not per tick). Each phase is scored against its own episodes and by the component
responsible for it:

  phase 1  gradual tick-rate decline   RRCF scores      window detection + control group
  phase 2  contextual price anomalies  RRCF scores      exact-tick and short-window recall
  phase 3  feed silence                silence log      match by the silence's last tick
  phase 4  point failures              implausible price: RRCF scores
                                       malformed ISIN / timestamp inversion: validation log

RRCF and the baselines are scored on the same protocol: pass any scores parquet with
exchange, instrument, timestamp_ms, z_score. The silence and validation logs are rule
based and produced by the feed-handler; they are not part of the RRCF-vs-baseline
comparison, so omit them when evaluating a baseline.

Key points of the protocol (see docs/EVALUATION_METHODOLOGY.md at the repo root):

  * Episode level, not tick level. Every metric is per episode (or per instrument),
    never per injected row, so different phases cannot share or inflate counts.
  * Scorable episodes. The worker scores one vector in ten, so an injected tick is
    often never scored. Recall is reported over scorable episodes (headline) and over
    all episodes. Scorable depends only on the stride, never on whether the detector
    fired, and is defined on the score records, not on the alerts.
  * One-sided alerts: an alert is z_score >= threshold (CoDisp is one-sided).
  * Controls instead of a time-shift null: unaffected instruments in the same window
    (phase 1, 3), untouched ticks (phase 2, 4) and clean days give the false-alarm
    rate under identical market conditions.
  * The operating threshold can be fixed from the clean days (--target-far) so it is
    never tuned on the injected data.
  * Confidence intervals come from a bootstrap over instruments (episodes of one
    instrument are correlated).

Usage:
    python scripts/evaluate_thesis.py \\
        --episodes ../price-feed-simulator/anomaly_log_episodes.csv \\
        --scores ./data/scores_rrcf.parquet \\
        --silence-log ../feed-handler/data/eval/silence_alerts.csv \\
        --validation-log ../feed-handler/data/eval/validation_alerts.csv \\
        --output ../results/thesis_YYYYMMDD_HHMMSS
"""

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Composite sort key = (instrument code << TS_BITS) + timestamp_ms. 2**42 ms is the
# year 2109, so timestamps never spill into the instrument bits.
TS_BITS = 42
TS_LIMIT = (1 << TS_BITS) - 1

PHASE_NAMES = {
    1: "tick_rate_decline",
    2: "contextual_price",
    3: "feed_silence",
    4: "point_failures",
}


# --------------------------------------------------------------------------- loading


def _fingerprint(path: Path) -> Dict:
    """Size, sha256 and row count of an input, so a result can be traced to its data."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            digest.update(chunk)
    info = {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        info["rows"] = pq.ParquetFile(path).metadata.num_rows
    else:
        with open(path, "rb") as f:
            info["rows"] = max(sum(1 for _ in f) - 1, 0)
    return info


def load_scores(path: Path) -> pd.DataFrame:
    wanted = ["exchange", "instrument", "timestamp_ms", "z_score"]
    import pyarrow.parquet as pq

    if "seq" in pq.ParquetFile(path).schema_arrow.names:
        wanted.append("seq")
    df = pd.read_parquet(path, columns=wanted)
    df["key"] = df["exchange"].astype(str) + "|" + df["instrument"].astype(str)
    df["ts"] = df["timestamp_ms"].astype(np.int64)
    df["seq"] = df["seq"].astype(np.int64) if "seq" in df.columns else 0
    return df[["key", "ts", "z_score", "seq"]]


def _extract_int(detail: pd.Series, name: str) -> pd.Series:
    return pd.to_numeric(detail.str.extract(rf"(?:^|;){name}=(-?\d+)", expand=False))


def load_episodes(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"Detail": "string"})
    df["Detail"] = df["Detail"].fillna("")

    if "SecType" in df.columns:
        df = df[df["SecType"] == "E"].copy()  # the feed-handler only forwards equities
    else:
        print("  WARNING: episode file has no SecType column; index instruments are included")

    df["phase"] = df["Phase"].str.extract(r"(\d+)", expand=False).astype(int)
    df["key"] = df["Exchange"].astype(str) + "|" + df["InstrumentID"].astype(str)
    df["start"] = df["StartMs"].astype(np.int64)
    df["end"] = df["EndMs"].astype(np.int64)
    df["observed"] = df["ObservedMs"].fillna(df["StartMs"]).astype(np.int64)
    df["resume"] = df["ResumeMs"]
    # The silence detector runs on the whole-second update time, so a silence's LastSeen is
    # on that clock; the ground truth records the last delivered message on it too.
    ld_clock = _extract_int(df["Detail"], "last_delivered_time_ms")
    df["last_delivered"] = ld_clock.where(ld_clock.notna(), df["LastDeliveredMs"])
    df["type"] = df["AnomalyType"]
    df["seq"] = pd.to_numeric(df["Seq"], errors="coerce").fillna(0).astype(np.int64) if "Seq" in df.columns else 0
    df["changed_ticks"] = _extract_int(df["Detail"], "changed_ticks")
    df["prev_ms"] = _extract_int(df["Detail"], "prev_ms")
    df["delivered_before"] = _extract_int(df["Detail"], "delivered_before")
    return df.reset_index(drop=True)


def load_silence(path: Path, exclude_flush: bool) -> pd.DataFrame:
    df = pd.read_csv(path)
    if exclude_flush and "Trigger" in df.columns:
        df = df[df["Trigger"] != "flush"]  # end-of-stream artefacts, not live detections
    df["key"] = df["Exchange"].astype(str) + "|" + df["Instrument"].astype(str)
    # how many times its threshold (the instrument's own gap quantile) the silence lasted;
    # alerts are logged at the configured multiplier, stricter ones are subsets
    df["mult"] = df["ElapsedMs"] / df["ThresholdMs"].replace(0, np.nan)
    return df.reset_index(drop=True)


# Strata. Recall is not one number: an instrument with 10 trades a day and one with 10,000
# are different detection problems, so results are reported by activity. Price anomalies
# (phases 2, 4) by the instrument's trades that day; feed-level effects (phases 1, 3) by its
# messages that day.
TRADE_BANDS = ([-1, 0, 19, 49, 199, np.inf], ["0", "1-19", "20-49", "50-199", "200+"])
ROW_TIERS = ([-1, 99, 999, 9999, np.inf], ["<100", "100-999", "1k-9,999", "10k+"])


def _band(values: pd.Series, spec) -> pd.Series:
    return pd.cut(values, bins=spec[0], labels=spec[1])


def load_instruments(path: Path) -> pd.DataFrame:
    """Per-instrument-day rows and trades, written by the simulator before any injection."""
    df = pd.read_csv(path, dtype={"Date": str})
    df = df[df["SecType"] == "E"].copy()
    df["key"] = df["Exchange"].astype(str) + "|" + df["InstrumentID"].astype(str)
    df["day"] = df["Date"]
    return df.reset_index(drop=True)


def stratify(ep, scorable, detected, column, args, rng):
    """Recall per stratum (None when the strata are not available)."""
    if column not in ep.columns or ep[column].isna().all():
        return None
    codes = ep["code"].to_numpy()
    scorable = np.asarray(scorable, dtype=bool)
    detected = np.asarray(detected, dtype=bool)
    out = {}
    for label in ep[column].cat.categories:
        m = (ep[column] == label).to_numpy()
        if m.any():
            out[str(label)] = recall_block(codes[m], scorable[m], detected[m], None, args.bootstrap, rng)
    unknown = int(ep[column].isna().sum())
    if unknown:
        out["unknown"] = {"episodes": unknown, "note": "instrument-day not in the instruments file"}
    return out


def load_validation(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["key"] = df["Exchange"].astype(str) + "|" + df["Instrument"].astype(str)
    return df.reset_index(drop=True)


# ----------------------------------------------------------------------- key/time index


class KeyIndex:
    """Maps the 'exchange|instrument' strings of every input onto one integer code."""

    def __init__(self, *series: pd.Series):
        self.index = pd.Index(pd.concat(series, ignore_index=True).unique())

    def codes(self, keys: pd.Series) -> np.ndarray:
        return self.index.get_indexer(keys).astype(np.int64)


def composite(codes: np.ndarray, ts: np.ndarray) -> np.ndarray:
    return (np.asarray(codes, dtype=np.int64) << TS_BITS) + np.clip(np.asarray(ts, dtype=np.int64), 0, TS_LIMIT)


class Timeline:
    """Events sorted by (instrument, time), answering 'any event in [lo, hi]?' for many
    episodes at once with binary search."""

    def __init__(self, codes: np.ndarray, ts: np.ndarray, values: Optional[np.ndarray] = None,
                 seqs: Optional[np.ndarray] = None):
        keys = composite(codes, ts)
        order = np.argsort(keys, kind="stable")
        self.keys = keys[order]
        self.values = None if values is None else np.asarray(values)[order]
        # message sequence numbers (0 = unknown), when the scores carry them
        self.seqs = None if seqs is None or not np.any(seqs) else np.asarray(seqs)[order]

    def __len__(self) -> int:
        return len(self.keys)

    def where(self, mask: np.ndarray) -> "Timeline":
        sub = Timeline.__new__(Timeline)
        sub.keys = self.keys[mask]
        sub.values = None if self.values is None else self.values[mask]
        sub.seqs = None if self.seqs is None else self.seqs[mask]
        return sub

    def count(self, codes: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        left = np.searchsorted(self.keys, composite(codes, lo), side="left")
        right = np.searchsorted(self.keys, composite(codes, hi), side="right")
        return right - left

    def first(self, codes: np.ndarray, lo: np.ndarray, hi: np.ndarray):
        """First event time in [lo, hi] per episode, with a mask of which had one."""
        if len(self.keys) == 0:
            return np.zeros(len(codes), dtype=bool), np.full(len(codes), -1, dtype=np.int64)
        left = np.searchsorted(self.keys, composite(codes, lo), side="left")
        inside = np.minimum(left, max(len(self.keys) - 1, 0))
        found = (left < len(self.keys)) & (self.keys[inside] <= composite(codes, hi))
        times = np.where(found, self.keys[inside] - (np.asarray(codes, dtype=np.int64) << TS_BITS), -1)
        return found, times


# --------------------------------------------------------------------------- statistics


def bootstrap_ci(codes: np.ndarray, hits: np.ndarray, n: int, rng) -> Optional[List[float]]:
    """95% CI of mean(hits), resampling whole instruments (episodes of one instrument
    are correlated, so resampling episodes would be too optimistic)."""
    if len(hits) == 0 or n <= 0:
        return None
    hits = np.asarray(hits, dtype=float)
    _, inverse = np.unique(codes, return_inverse=True)
    sums = np.bincount(inverse, weights=hits)
    counts = np.bincount(inverse).astype(float)
    m = len(sums)
    means = np.empty(n)
    for i in range(n):
        pick = rng.integers(0, m, m)
        means[i] = sums[pick].sum() / counts[pick].sum()
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def bootstrap_diff_ci(a, b, n: int, rng) -> Optional[List[float]]:
    """95% CI of mean(a) - mean(b) with independent instrument resampling. a and b are
    (codes, hits) pairs."""
    if len(a[1]) == 0 or len(b[1]) == 0 or n <= 0:
        return None

    def prepare(codes, hits):
        _, inv = np.unique(codes, return_inverse=True)
        return np.bincount(inv, weights=np.asarray(hits, float)), np.bincount(inv).astype(float)

    sa, ca = prepare(*a)
    sb, cb = prepare(*b)
    diffs = np.empty(n)
    for i in range(n):
        pa = rng.integers(0, len(sa), len(sa))
        pb = rng.integers(0, len(sb), len(sb))
        diffs[i] = sa[pa].sum() / ca[pa].sum() - sb[pb].sum() / cb[pb].sum()
    return [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]


def _rate(x) -> Optional[float]:
    return float(np.mean(x)) if len(x) else None


def _quantiles(values: np.ndarray) -> Dict:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return {"median": None, "p90": None}
    return {"median": float(np.median(values)), "p90": float(np.percentile(values, 90))}


def recall_block(codes, scorable, detected, latency_ms, n_boot, rng) -> Dict:
    """Recall over scorable episodes (headline) and over all episodes."""
    codes = np.asarray(codes)
    scorable = np.asarray(scorable, dtype=bool)
    detected = np.asarray(detected, dtype=bool)
    hit = detected & scorable
    return {
        "episodes": int(len(codes)),
        "scorable": int(scorable.sum()),
        "scorable_fraction": _rate(scorable),
        "detected": int(hit.sum()),
        "recall_scorable": _rate(detected[scorable]),
        "recall_scorable_ci95": bootstrap_ci(codes[scorable], detected[scorable], n_boot, rng),
        "recall_all": _rate(hit),
        "latency_ms": _quantiles(np.asarray(latency_ms)[hit]) if latency_ms is not None else None,
    }


# ------------------------------------------------------------------------ RRCF phases


def eval_point_episodes(ep, scores_tl, alerts_tl, args, rng, label) -> Dict:
    """Single-tick anomalies: is the injected tick's own vector an alert (strict), or
    is there an alert on the instrument within point_window_ms afterwards (lenient)?"""
    if len(ep) == 0:
        return {"episodes": 0}
    codes = ep["code"].to_numpy()
    obs = ep["observed"].to_numpy()
    tol = args.match_tol_ms

    seq = ep["seq"].to_numpy()
    if scores_tl.seqs is not None and len(seq) and (seq > 0).all():
        # exact identity: the injected message's own vector, however many of the instrument's
        # messages share its millisecond
        scorable = np.isin(seq, scores_tl.seqs)
        strict = np.isin(seq, alerts_tl.seqs)
        matching = "message sequence number"
    else:
        scorable = scores_tl.count(codes, obs - tol, obs + tol) > 0
        strict = alerts_tl.count(codes, obs - tol, obs + tol) > 0
        matching = "instrument and millisecond (ambiguous when messages share a millisecond)"
    lenient = alerts_tl.count(codes, obs - tol, obs + args.point_window_ms) > 0
    found, first = alerts_tl.first(codes, obs - tol, obs + args.point_window_ms)
    latency = np.where(found, first - ep["start"].to_numpy(), 0)

    return {
        "label": label,
        "strict_matching": matching,
        "strict_exact_tick": recall_block(codes, scorable, strict, None, args.bootstrap, rng),
        "strict_by_trades_that_day": stratify(ep, scorable, strict, "trades_band", args, rng),
        "lenient_within_window": {
            **recall_block(codes, scorable, lenient, latency, args.bootstrap, rng),
            "window_ms": args.point_window_ms,
        },
    }


def eval_stale_runs(ep, scores_tl, alerts_tl, args, rng) -> Dict:
    """Stale-price runs. Only runs in which the frozen price actually differed from the
    true one at least once can be detected, so those are the effective episodes."""
    if len(ep) == 0:
        return {"episodes": 0}
    effective = ep[ep["changed_ticks"].fillna(1) > 0]
    out = {"episodes_total": int(len(ep)), "episodes_effective": int(len(effective))}
    if len(effective) == 0:
        return out

    codes = effective["code"].to_numpy()
    lo = effective["start"].to_numpy()
    hi = effective["end"].to_numpy()
    scorable = scores_tl.count(codes, lo, hi) > 0
    found, first = alerts_tl.first(codes, lo, hi + args.stale_window_ms)
    out.update(
        recall_block(codes, scorable, found, first - lo, args.bootstrap, rng),
    )
    out["window_after_run_ms"] = args.stale_window_ms
    out["by_trades_that_day"] = stratify(effective, scorable, found, "trades_band", args, rng)
    return out


def eval_phase1(ep, scores_df, scores_tl, alerts_tl, args, rng, activity=None) -> Dict:
    """Tick-rate decline. Affected instruments are compared with unaffected instruments
    that were active in the same window, so market conditions are identical."""
    if len(ep) == 0:
        return {"episodes": 0}

    aff_codes, aff_scorable, aff_detected, aff_alerts, aff_scores = [], [], [], [], []
    ctl_codes, ctl_detected, ctl_alerts, ctl_scores = [], [], [], []
    first_alert_delay = []
    aff_tier, ctl_tier = [], []

    for (start, end), group in ep.groupby(["start", "end"]):
        codes = group["code"].to_numpy()
        lo = np.full(len(codes), start, dtype=np.int64)
        hi = np.full(len(codes), end, dtype=np.int64)
        n_scores = scores_tl.count(codes, lo, hi)
        n_alerts = alerts_tl.count(codes, lo, hi)
        found, first = alerts_tl.first(codes, lo, hi)

        aff_codes.append(codes)
        aff_scorable.append(n_scores > 0)
        aff_detected.append(n_alerts > 0)
        aff_alerts.append(n_alerts)
        aff_scores.append(n_scores)
        first_alert_delay.append((first - start)[found])
        if "rows_band" in group.columns:
            aff_tier.append(group["rows_band"].astype(object).to_numpy())

        # control group: instruments scored in this window that had no episode
        window = scores_df[(scores_df["ts"] >= start) & (scores_df["ts"] <= end)]
        active = np.unique(window["code"].to_numpy())
        control = np.setdiff1d(active, codes)
        lo_c = np.full(len(control), start, dtype=np.int64)
        hi_c = np.full(len(control), end, dtype=np.int64)
        c_alerts = alerts_tl.count(control, lo_c, hi_c)
        ctl_codes.append(control)
        ctl_detected.append(c_alerts > 0)
        ctl_alerts.append(c_alerts)
        ctl_scores.append(scores_tl.count(control, lo_c, hi_c))
        if activity is not None:
            day = str(_utc_day(np.array([start]))[0])
            rows = activity.reindex(pd.MultiIndex.from_arrays([control, np.full(len(control), day)]))["Rows"]
            ctl_tier.append(_band(rows.reset_index(drop=True), ROW_TIERS).astype(object).to_numpy())

    aff_codes = np.concatenate(aff_codes)
    aff_scorable = np.concatenate(aff_scorable)
    aff_detected = np.concatenate(aff_detected)
    ctl_codes = np.concatenate(ctl_codes)
    ctl_detected = np.concatenate(ctl_detected)

    def density(alerts, scores):
        total = int(np.concatenate(scores).sum())
        return 1000.0 * int(np.concatenate(alerts).sum()) / total if total else None

    a = (aff_codes[aff_scorable], aff_detected[aff_scorable])
    b = (ctl_codes, ctl_detected)

    by_tier = None
    if aff_tier and ctl_tier:
        at, ct = np.concatenate(aff_tier), np.concatenate(ctl_tier)
        by_tier = {}
        for label in ROW_TIERS[1]:
            am, cm = at == label, ct == label
            if am.any() or cm.any():
                by_tier[label] = {
                    "affected_instruments": int(am.sum()),
                    "affected_detection_rate": _rate(aff_detected[am & aff_scorable]) if (am & aff_scorable).any() else None,
                    "control_instruments": int(cm.sum()),
                    "control_detection_rate": _rate(ctl_detected[cm]) if cm.any() else None,
                }
    return {
        "affected": {
            **recall_block(aff_codes, aff_scorable, aff_detected, None, args.bootstrap, rng),
            "alerts_per_1000_vectors": density(aff_alerts, aff_scores),
        },
        "control_unaffected": {
            "instruments": int(len(ctl_codes)),
            "detection_rate": _rate(ctl_detected),
            "detection_rate_ci95": bootstrap_ci(ctl_codes, ctl_detected, args.bootstrap, rng),
            "alerts_per_1000_vectors": density(ctl_alerts, ctl_scores),
        },
        "detection_rate_difference_ci95": bootstrap_diff_ci(a, b, args.bootstrap, rng),
        "by_messages_that_day": by_tier,
        "first_alert_delay_ms": _quantiles(np.concatenate(first_alert_delay)),
        "note": (
            "A gradual decline has no sharp onset, so delay to the first alert mostly "
            "reflects the background alert rate; use the affected-vs-control difference."
        ),
    }


# ------------------------------------------------------------ silence (phase 3)


def eval_phase3(ep, silence, index, scores_df, args, rng) -> Dict:
    """Feed silence, against the feed-handler's silence log. An alert answers an
    episode when it is about the silence that began at the blackout's previous tick
    (LastSeen == LastDelivered), else when it was detected inside the blackout."""
    if len(ep) == 0:
        return {"episodes": 0}
    if silence is None:
        return {"episodes": int(len(ep)), "note": "no --silence-log given; not evaluated"}

    silence = silence.copy()
    silence["code"] = index.codes(silence["key"])

    codes = ep["code"].to_numpy()
    start = ep["start"].to_numpy()
    scheduled_end = ep["end"].to_numpy()
    resume = ep["resume"].fillna(ep["end"]).to_numpy().astype(np.int64)
    last_delivered = ep["last_delivered"]
    has_ld = last_delivered.notna().to_numpy()
    ld = last_delivered.fillna(0).to_numpy().astype(np.int64)
    warm = (ep["delivered_before"].fillna(np.inf) >= args.min_observations).to_numpy()

    scope_lo = int(min(start.min(), ld[has_ld].min() if has_ld.any() else start.min()))
    scope_hi = int(resume.max())
    exchanges = set(ep["Exchange"])

    result = {"episodes": int(len(ep)), "warm_episodes": int(warm.sum()), "by_multiplier": {}}
    for mult in args.mult_thresholds:
        sub = silence[silence["mult"] >= mult]
        sub_codes = sub["code"].to_numpy()

        by_last_seen = Timeline(sub_codes, sub["LastSeenMs"].to_numpy())
        by_detected = Timeline(sub_codes, sub["DetectedAtMs"].to_numpy())
        match_last = by_last_seen.count(codes, ld - args.match_tol_ms, ld + args.match_tol_ms) > 0
        match_window = by_detected.count(codes, start, resume) > 0
        detected = np.where(has_ld, match_last, match_window)

        # detection time of the matching alert, for latency
        det_times = _detected_time(sub, index, codes, np.where(has_ld, ld, start), has_ld, args, resume)
        latency_start = np.where(detected, det_times - start, 0)
        latency_last = np.where(detected & has_ld, det_times - ld, 0)

        block = recall_block(codes[warm], np.ones(int(warm.sum()), bool), detected[warm], latency_start[warm], args.bootstrap, rng)
        block["recall_all_episodes"] = _rate(detected)
        block["by_messages_that_day"] = stratify(ep[warm], np.ones(int(warm.sum()), bool), detected[warm], "rows_band", args, rng)
        block["latency_from_last_tick_ms"] = _quantiles(latency_last[warm & detected & has_ld])
        block["latency_from_blackout_start_ms"] = _quantiles(latency_start[warm & detected])

        # precision inside the phase-3 scope: alerts on the affected exchange(s)
        in_scope = sub[
            sub["Exchange"].isin(exchanges)
            & (sub["ObservedAtMs"] >= scope_lo)
            & (sub["DetectedAtMs"] <= scope_hi)
        ]
        matched = _alerts_matching_episodes(in_scope, codes, start, resume, ld, has_ld, args)
        block["alert_precision_in_scope"] = {
            "alerts": int(len(in_scope)),
            "matched_to_blackouts": int(matched.sum()),
            "precision": _rate(matched),
        }

        # control: instruments on the same exchange(s) that had no blackout
        window = scores_df[(scores_df["ts"] >= scope_lo) & (scores_df["ts"] <= scope_hi)]
        active = np.unique(window["code"].to_numpy())
        control = np.setdiff1d(active, codes)
        ctl_alerted = by_detected.count(control, np.full(len(control), scope_lo), np.full(len(control), scope_hi)) > 0
        block["control_unaffected"] = {
            "instruments": int(len(control)),
            "alert_rate": _rate(ctl_alerted),
        }
        result["by_multiplier"][str(mult)] = block
    result["note"] = (
        "detection time is LastSeen + threshold (event-time virtual timer), so latency "
        "from the last tick equals the threshold by construction; the informative "
        "quantities are recall, precision and the control alert rate. Episodes on "
        "instruments with fewer than min_observations delivered ticks are excluded: the "
        "detector has no baseline for them."
    )
    return result


def _detected_time(sub, index, codes, anchor, has_ld, args, resume) -> np.ndarray:
    """DetectedAt of the alert answering each episode (0 where none)."""
    out = np.zeros(len(codes), dtype=np.int64)
    if len(sub) == 0:
        return out
    tl = Timeline(sub["code"].to_numpy(), sub["LastSeenMs"].to_numpy(), sub["DetectedAtMs"].to_numpy())
    left = np.searchsorted(tl.keys, composite(codes, anchor - args.match_tol_ms), side="left")
    inside = np.minimum(left, len(tl.keys) - 1)
    ok = has_ld & (left < len(tl.keys)) & (tl.keys[inside] <= composite(codes, anchor + args.match_tol_ms))
    out[ok] = tl.values[inside][ok]

    tl2 = Timeline(sub["code"].to_numpy(), sub["DetectedAtMs"].to_numpy(), sub["DetectedAtMs"].to_numpy())
    found, first = tl2.first(codes, anchor, resume)
    fallback = (~has_ld) & found
    out[fallback] = first[fallback]
    return out


def _alerts_matching_episodes(alerts, codes, start, resume, ld, has_ld, args) -> np.ndarray:
    """For each alert: does it answer some episode?"""
    if len(alerts) == 0:
        return np.zeros(0, dtype=bool)
    a_codes = alerts["code"].to_numpy()
    by_ld = Timeline(codes[has_ld], ld[has_ld])
    by_window_start = Timeline(codes[~has_ld], start[~has_ld])
    m1 = by_ld.count(a_codes, alerts["LastSeenMs"].to_numpy() - args.match_tol_ms,
                     alerts["LastSeenMs"].to_numpy() + args.match_tol_ms) > 0
    # alerts about instruments without LastDelivered: detected inside a blackout window
    m2 = np.zeros(len(alerts), dtype=bool)
    if (~has_ld).any():
        c2, s2, r2 = codes[~has_ld], start[~has_ld], resume[~has_ld]
        order = np.argsort(composite(c2, s2))
        lo_keys, hi_keys = composite(c2, s2)[order], composite(c2, r2)[order]
        run_max = np.maximum.accumulate(hi_keys)
        det = composite(a_codes, alerts["DetectedAtMs"].to_numpy())
        pos = np.searchsorted(lo_keys, det, side="right") - 1
        m2 = (pos >= 0) & (run_max[np.maximum(pos, 0)] >= det)
    return m1 | m2


# ------------------------------------------------------------ validator (phase 4)


def eval_validation(ep, validation, index, args, rng) -> Dict:
    """Malformed ISIN and timestamp inversion, against the validator log. An alert
    answers an episode when it is on the same instrument at the tick time the feed
    handler saw (ObservedMs)."""
    out = {}
    for ep_type, alert_type in (
        ("malformed_isin", "MALFORMED_ISIN"),
        ("timestamp_inversion", "TIMESTAMP_INVERSION"),
    ):
        sub = ep[ep["type"] == ep_type]
        if len(sub) == 0:
            continue
        if validation is None:
            out[ep_type] = {"episodes": int(len(sub)), "note": "no --validation-log given; not evaluated"}
            continue

        val = validation[validation["AlertType"] == alert_type].copy()
        val["code"] = index.codes(val["key"])
        codes = sub["code"].to_numpy()
        obs = sub["observed"].to_numpy()
        tol = args.match_tol_ms

        detected = Timeline(val["code"].to_numpy(), val["TickTimeMs"].to_numpy()).count(codes, obs - tol, obs + tol) > 0
        block = recall_block(codes, np.ones(len(codes), bool), detected, None, args.bootstrap, rng)

        if ep_type == "timestamp_inversion":
            # a rewind is only visible to a per-instrument monotonicity check if it lands
            # before the instrument's previous tick (by more than the tolerance)
            prev = sub["prev_ms"].to_numpy(dtype=float)
            detectable = np.where(np.isnan(prev), False, obs < prev - args.validator_tolerance_ms)
            block["detectable_episodes"] = int(detectable.sum())
            block["recall_detectable"] = _rate(detected[detectable])
            block["recall_detectable_ci95"] = bootstrap_ci(codes[detectable], detected[detectable], args.bootstrap, rng)

        # false positives: validator alerts that answer no episode of this type
        answered = Timeline(codes, obs).count(val["code"].to_numpy(),
                                              val["TickTimeMs"].to_numpy() - tol,
                                              val["TickTimeMs"].to_numpy() + tol) > 0
        block["alerts"] = int(len(val))
        block["false_alerts"] = int((~answered).sum())
        out[ep_type] = block
    return out


# -------------------------------------------------------------- false-alarm rates


def _utc_day(ms: np.ndarray) -> np.ndarray:
    return pd.to_datetime(ms, unit="ms", utc=True).strftime("%Y-%m-%d").to_numpy()


def clean_days(scores_df, episodes, warmup_days: int, explicit: Optional[List[str]]) -> Dict:
    all_days = sorted(set(_utc_day(scores_df["ts"].to_numpy())))
    injected = set(_utc_day(episodes["start"].to_numpy())) if len(episodes) else set()
    clean = [d for d in all_days if d not in injected]
    warm = set(all_days[:warmup_days])
    headline = explicit if explicit else [d for d in clean if d not in warm]
    return {"all": all_days, "injected": sorted(injected), "clean": clean,
            "warmup_excluded": sorted(warm), "headline": headline}


def false_alarm_rate(scores_df, days: Dict, thresholds: List[float]) -> Dict:
    """Alerts per 1000 scored vectors on the clean days, per threshold and per day."""
    day_of = _utc_day(scores_df["ts"].to_numpy())
    z = scores_df["z_score"].to_numpy()
    codes = scores_df["code"].to_numpy()
    out = {"days": days, "by_threshold": {}}
    for thr in thresholds:
        per_day = {}
        for day in days["clean"]:
            m = day_of == day
            n = int(m.sum())
            alerted = z[m] >= thr
            per_day[day] = {
                "vectors": n,
                "alerts": int(alerted.sum()),
                "alerts_per_1000_vectors": 1000.0 * alerted.sum() / n if n else None,
                "instruments_alerting": float(len(np.unique(codes[m][alerted])) / max(len(np.unique(codes[m])), 1)),
            }
        headline = [per_day[d] for d in days["headline"] if d in per_day]
        vectors = sum(d["vectors"] for d in headline)
        alerts = sum(d["alerts"] for d in headline)
        out["by_threshold"][str(thr)] = {
            "headline_alerts_per_1000_vectors": 1000.0 * alerts / vectors if vectors else None,
            "per_day": per_day,
        }
    return out


def silence_false_alarms(silence, scores_df, days: Dict, mults: List[float]) -> Dict:
    if silence is None:
        return {}
    day_of_alert = _utc_day(silence["ObservedAtMs"].to_numpy())
    day_of_score = _utc_day(scores_df["ts"].to_numpy())
    out = {}
    for mult in mults:
        sel = (silence["mult"] >= mult).to_numpy()
        per_day = {}
        for day in days["clean"]:
            active = len(np.unique(scores_df["code"].to_numpy()[day_of_score == day]))
            n = int((sel & (day_of_alert == day)).sum())
            per_day[day] = {"alerts": n, "active_instruments": int(active),
                            "alerts_per_active_instrument": n / active if active else None}
        out[str(mult)] = per_day
    return out


def validation_false_alarms(validation, days: Dict) -> Dict:
    if validation is None:
        return {}
    day_of = _utc_day(validation["TickTimeMs"].to_numpy())
    return {day: int((day_of == day).sum()) for day in days["clean"]}


# ------------------------------------------------------------------ precision (RRCF)


def cluster_precision(alerts_tl, windows_codes, windows_lo, windows_hi, gap_ms) -> Dict:
    """Alert clusters (alerts on one instrument less than gap_ms apart) that overlap an
    episode window. Only meaningful at the injected density."""
    if len(windows_codes) == 0 or len(alerts_tl) == 0:
        return {"clusters": 0, "precision": None}
    lo_keys = composite(windows_codes, windows_lo)
    order = np.argsort(lo_keys)
    lo_sorted = lo_keys[order]
    run_max = np.maximum.accumulate(composite(windows_codes, windows_hi)[order])

    scope_lo, scope_hi = int(windows_lo.min()), int(windows_hi.max())
    keys = alerts_tl.keys
    code_of = keys >> TS_BITS
    ts_of = keys & TS_LIMIT
    in_scope = (ts_of >= scope_lo) & (ts_of <= scope_hi)
    keys, code_of, ts_of = keys[in_scope], code_of[in_scope], ts_of[in_scope]
    if len(keys) == 0:
        return {"clusters": 0, "precision": None}

    explained = (np.searchsorted(lo_sorted, keys, side="right") - 1)
    explained = (explained >= 0) & (run_max[np.maximum(explained, 0)] >= keys)

    new_cluster = np.ones(len(keys), dtype=bool)
    new_cluster[1:] = (code_of[1:] != code_of[:-1]) | ((ts_of[1:] - ts_of[:-1]) > gap_ms)
    cluster_id = np.cumsum(new_cluster) - 1
    hit = np.bincount(cluster_id, weights=explained.astype(float)) > 0
    return {"alerts": int(len(keys)), "clusters": int(len(hit)), "precision": float(hit.mean()),
            "alert_level_precision": float(explained.mean())}


# --------------------------------------------------------------------------- driver


def evaluate(args) -> Dict:
    rng = np.random.default_rng(args.seed)

    print("Loading inputs...")
    episodes = load_episodes(Path(args.episodes))
    scores = load_scores(Path(args.scores))
    silence = load_silence(Path(args.silence_log), not args.include_flush) if args.silence_log else None
    validation = load_validation(Path(args.validation_log)) if args.validation_log else None
    instruments = load_instruments(Path(args.instruments)) if args.instruments else None
    print(f"  episodes (equities): {len(episodes):,}  scores: {len(scores):,}"
          f"  silence alerts: {0 if silence is None else len(silence):,}"
          f"  validation alerts: {0 if validation is None else len(validation):,}")

    index = KeyIndex(episodes["key"], scores["key"],
                     *(x["key"] for x in (silence, validation, instruments) if x is not None))
    episodes["code"] = index.codes(episodes["key"])
    scores["code"] = index.codes(scores["key"])

    activity = None
    if instruments is not None:
        instruments["code"] = index.codes(instruments["key"])
        activity = instruments.set_index(["code", "day"])[["Rows", "Trades"]]
        episodes["day"] = _utc_day(episodes["start"].to_numpy())
        episodes = episodes.merge(instruments[["code", "day", "Rows", "Trades"]], on=["code", "day"], how="left")
        episodes["trades_band"] = _band(episodes["Trades"], TRADE_BANDS)
        episodes["rows_band"] = _band(episodes["Rows"], ROW_TIERS)
        print(f"  activity attached to {int(episodes['Rows'].notna().sum()):,} of {len(episodes):,} episodes")

    overlap = float(np.isin(np.unique(episodes["code"]), np.unique(scores["code"])).mean()) if len(episodes) else 0.0
    if len(episodes) and overlap == 0.0:
        raise SystemExit("No episode instrument appears in the scores: check the exchange/instrument "
                         "naming of the inputs (expected 'exchange|instrument' keys to match).")

    scores_tl = Timeline(scores["code"].to_numpy(), scores["ts"].to_numpy(), scores["z_score"].to_numpy(),
                         scores["seq"].to_numpy())
    days = clean_days(scores, episodes, args.warmup_days, args.clean_days)

    thresholds = sorted(set(args.thresholds + [args.alert_threshold]))
    far = false_alarm_rate(scores, days, thresholds)

    operating = args.alert_threshold
    if args.target_far is not None:
        ok = [t for t in thresholds
              if far["by_threshold"][str(t)]["headline_alerts_per_1000_vectors"] is not None
              and far["by_threshold"][str(t)]["headline_alerts_per_1000_vectors"] <= args.target_far]
        if not ok:
            raise SystemExit(f"No threshold in {thresholds} reaches {args.target_far} alerts per 1000 "
                             f"vectors on the clean days; widen --thresholds.")
        operating = min(ok)
        print(f"  operating threshold {operating} chosen from clean days (target {args.target_far}/1000)")

    def phase_eval(thr: float, args) -> Dict:
        alerts_tl = scores_tl.where(scores_tl.values >= thr)
        p1 = episodes[episodes["phase"] == 1]
        p2 = episodes[episodes["phase"] == 2]
        p4 = episodes[episodes["phase"] == 4]
        p2_point = p2[p2["type"].isin(["price_spike", "price_deviation"])]
        p2_stale = p2[p2["type"] == "stale_price"]
        p4_price = p4[p4["type"] == "implausible_price"]
        # null_price zeroes the last price; with the last traded price carried forward
        # through quote-only rows a nulled trade looks like a quote update, so it cannot
        # be detected and is not evaluated (older runs may still contain it)
        p4_null = p4[p4["type"] == "null_price"]

        result = {
            "phase1": eval_phase1(p1, scores, scores_tl, alerts_tl, args, rng, activity),
            "phase2": {
                "point": {t: eval_point_episodes(p2_point[p2_point["type"] == t], scores_tl, alerts_tl, args, rng, t)
                          for t in sorted(p2_point["type"].unique())},
                "stale_price": eval_stale_runs(p2_stale, scores_tl, alerts_tl, args, rng),
            },
            "phase4_implausible_price": {
                **eval_point_episodes(p4_price, scores_tl, alerts_tl, args, rng, "implausible_price"),
                "not_evaluated_null_price_episodes": int(len(p4_null)),
            },
        }
        # alert-level precision at the injected density, per family
        families = {
            "phase1": p1[["code", "start", "end"]].rename(columns={"start": "lo", "end": "hi"}),
            "phase2": pd.concat([
                p2_point.assign(lo=p2_point["observed"] - args.match_tol_ms, hi=p2_point["observed"] + args.point_window_ms),
                p2_stale.assign(lo=p2_stale["start"], hi=p2_stale["end"] + args.stale_window_ms),
            ]),
            "phase4_implausible_price": p4_price.assign(lo=p4_price["observed"] - args.match_tol_ms,
                                                        hi=p4_price["observed"] + args.point_window_ms),
        }
        result["cluster_precision_at_injected_density"] = {
            name: cluster_precision(alerts_tl, fam["code"].to_numpy(), fam["lo"].to_numpy(),
                                    fam["hi"].to_numpy(), args.cluster_gap_ms)
            for name, fam in families.items() if len(fam)
        }
        return result

    print(f"Evaluating at threshold {operating}...")
    headline = phase_eval(operating, args)

    sweep_rows = []
    if args.thresholds:
        print("Threshold sweep...")
        sweep_args = argparse.Namespace(**{**vars(args), "bootstrap": 0})  # point estimates only
        for thr in thresholds:
            res = headline if thr == operating else phase_eval(thr, sweep_args)
            row = {"threshold": thr,
                   "clean_alerts_per_1000_vectors": far["by_threshold"][str(thr)]["headline_alerts_per_1000_vectors"]}
            row["phase1_affected_detection_rate"] = res["phase1"].get("affected", {}).get("recall_scorable")
            row["phase1_control_detection_rate"] = res["phase1"].get("control_unaffected", {}).get("detection_rate")
            for t, block in res["phase2"]["point"].items():
                row[f"phase2_{t}_strict_recall"] = block["strict_exact_tick"]["recall_scorable"]
            row["phase2_stale_recall"] = res["phase2"]["stale_price"].get("recall_scorable")
            row["phase4_implausible_price_strict_recall"] = res["phase4_implausible_price"].get("strict_exact_tick", {}).get("recall_scorable")
            sweep_rows.append(row)

    results = {
        "method": args.method_name,
        "operating_threshold": operating,
        "alert_rule": "z_score >= threshold (one-sided)",
        "inputs": {name: _fingerprint(Path(p)) for name, p in
                   (("episodes", args.episodes), ("scores", args.scores),
                    ("silence_log", args.silence_log), ("validation_log", args.validation_log),
                    ("instruments", args.instruments)) if p},
        "data_checks": {"episode_instruments_present_in_scores": overlap,
                        "episodes_by_phase": {str(p): int((episodes["phase"] == p).sum()) for p in sorted(episodes["phase"].unique())}},
        "parameters": {k: v for k, v in vars(args).items()
                       if k not in ("episodes", "scores", "silence_log", "validation_log", "output",
                                    "ground_truth_csv", "ground_truth_manifest")},
        "rrcf": headline,
        "phase3_feed_silence": eval_phase3(episodes[episodes["phase"] == 3], silence, index, scores, args, rng),
        "phase4_validator": eval_validation(episodes[episodes["phase"] == 4], validation, index, args, rng),
        "false_alarms": {
            "rrcf": far,
            "silence": silence_false_alarms(silence, scores, days, args.mult_thresholds),
            "validation_alerts_per_clean_day": validation_false_alarms(validation, days),
        },
        "sweep": sweep_rows,
    }
    return results


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"not JSON serialisable: {type(obj)}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Evaluate feed-degradation detection against the simulator's ground truth")
    p.add_argument("--episodes", help="anomaly_log_episodes.csv from the simulator")
    # Deprecated: the previous interface. The tick-level log is no longer the ground
    # truth; its sibling <name>_episodes.csv is used instead, so existing make targets
    # keep working (without the silence/validation logs, phases 3-4 are then skipped).
    p.add_argument("--ground-truth-csv", help=argparse.SUPPRESS)
    p.add_argument("--ground-truth-manifest", help=argparse.SUPPRESS)
    p.add_argument("--scores", required=True, help="scores parquet (exchange, instrument, timestamp_ms, z_score)")
    p.add_argument("--silence-log", help="silence_alerts.csv from the feed-handler (phase 3)")
    p.add_argument("--validation-log", help="validation_alerts.csv from the feed-handler (phase 4)")
    p.add_argument("--instruments", help="anomaly_log_instruments.csv from the simulator (rows and trades per "
                                          "instrument and day); enables results by activity. Default: the file "
                                          "next to --episodes, if present")
    p.add_argument("--output", required=True, help="output directory")
    p.add_argument("--method-name", default="rrcf", help="label stored in the results")

    p.add_argument("--alert-threshold", type=float, default=2.0, help="z_score >= threshold is an alert (default 2.0)")
    p.add_argument("--thresholds", type=lambda s: [float(x) for x in s.split(",") if x], default=[],
                   help="comma-separated thresholds for the sweep, e.g. 1,2,3,4,5")
    p.add_argument("--target-far", type=float, default=None,
                   help="choose the smallest swept threshold whose clean-day false alarms are at most this many "
                        "alerts per 1000 vectors (fixes the operating point without using injected data)")
    p.add_argument("--mult-thresholds", type=lambda s: [float(x) for x in s.split(",") if x], default=[1, 2, 3, 5, 10],
                   help="silence alerts are evaluated at these multiples of their own threshold (1 = as logged)")
    p.add_argument("--include-flush", action="store_true", help="keep end-of-stream (flush) silence alerts")

    p.add_argument("--match-tol-ms", type=int, default=2, help="tolerance for exact timestamp matches")
    p.add_argument("--point-window-ms", type=int, default=5000, help="lenient window after a point anomaly")
    p.add_argument("--stale-window-ms", type=int, default=5000, help="allowance after the end of a stale run")
    p.add_argument("--cluster-gap-ms", type=int, default=5000, help="alerts closer than this form one cluster")
    p.add_argument("--min-observations", type=int, default=50, help="warm-up ticks before the silence detector can alert")
    p.add_argument("--validator-tolerance-ms", type=int, default=1000, help="timestamp tolerance of the validator")

    p.add_argument("--warmup-days", type=int, default=1, help="leading days excluded from the clean-day false-alarm headline")
    p.add_argument("--clean-days", type=lambda s: [x for x in s.split(",") if x], default=None,
                   help="explicit clean days (YYYY-MM-DD) for the headline; default: days without episodes")
    p.add_argument("--bootstrap", type=int, default=1000, help="bootstrap resamples over instruments (0 disables)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--archive-inputs", action="store_true", help="copy the small input files into the output directory")
    args = p.parse_args(argv)
    if args.episodes is None:
        if not args.ground_truth_csv:
            p.error("--episodes is required")
        gt = Path(args.ground_truth_csv)
        args.episodes = str(gt.with_name(gt.stem + "_episodes.csv"))
        print(f"NOTE: --ground-truth-csv is deprecated; using {args.episodes}")
    if args.instruments is None:
        sibling = Path(args.episodes).with_name(Path(args.episodes).name.replace("_episodes", "_instruments"))
        if sibling != Path(args.episodes) and sibling.exists():
            args.instruments = str(sibling)
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    results = evaluate(args)

    with open(out / "evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2, default=_json_default)
    if results["sweep"]:
        pd.DataFrame(results["sweep"]).to_csv(out / "sweep.csv", index=False)
    if args.archive_inputs:
        inputs = out / "inputs"
        inputs.mkdir(exist_ok=True)
        for p in (args.episodes, args.instruments, args.silence_log, args.validation_log):
            if p:
                shutil.copy2(p, inputs / Path(p).name)

    print(f"Results written to {out / 'evaluation_results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
