"""Tests for scripts/evaluate_thesis.py on synthetic data with known outcomes."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_thesis.py"
_spec = importlib.util.spec_from_file_location("evaluate_thesis", _SCRIPT)
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)

EPISODE_COLUMNS = [
    "EpisodeID", "Phase", "AnomalyType", "Exchange", "InstrumentID", "StartMs", "EndMs",
    "ObservedMs", "ResumeMs", "LastDeliveredMs", "Detail", "SecType",
]


def ms(day: str, hh=0, mm=0, ss=0, extra_ms=0) -> int:
    midnight = int(pd.Timestamp(day, tz="UTC").value // 1_000_000)
    return midnight + ((hh * 60 + mm) * 60 + ss) * 1000 + extra_ms


D_WARM, D_P1, D_P23, D_P4, D_CLEAN = "2021-11-08", "2021-11-09", "2021-11-10", "2021-11-11", "2021-11-12"


def episode(id_, phase, typ, inst, start, end=None, observed=None, resume=None, last_delivered=None,
            detail="", sectype="E"):
    return {
        "EpisodeID": id_, "Phase": f"phase{phase}", "AnomalyType": typ, "Exchange": "ETR", "InstrumentID": inst,
        "StartMs": start, "EndMs": start if end is None else end,
        "ObservedMs": observed if observed is not None else (start if phase in (2, 4) else None),
        "ResumeMs": resume, "LastDeliveredMs": last_delivered, "Detail": detail, "SecType": sectype,
    }


def build_dataset(tmp_path):
    ep, scores = [], []

    def add_scores(inst, times, z):
        z = np.broadcast_to(np.asarray(z, dtype=float), (len(times),))
        for t, zz in zip(times, z):
            scores.append(("ETR", inst, int(t), float(zz)))

    # --- clean days: day 1 is the warm-up (excluded), day 5 is the headline.
    add_scores("W1.ETR", [ms(D_WARM, 10, 0, i) for i in range(100)], [3.0] * 5 + [0.1] * 95)
    add_scores("W1.ETR", [ms(D_CLEAN, 10, 0, 0, extra_ms=i * 100) for i in range(1000)],
               [3.0] * 4 + [0.1] * 996)

    # --- phase 1 (day 2): one affected instrument and three controls in the same window
    start, end = ms(D_P1, 9, 30), ms(D_P1, 14, 0)
    ep.append(episode(1, 1, "tick_rate_decline", "P.ETR", start, end,
                      detail="ticks_seen=20;ticks_dropped=8"))
    times = [start + 60_000 * i for i in range(1, 21)]
    add_scores("P.ETR", times, [3.0] + [0.1] * 19)
    add_scores("C1.ETR", times, [3.0] + [0.1] * 19)
    add_scores("C2.ETR", times, 0.1)
    add_scores("C3.ETR", times, 0.1)

    # --- phase 2 (day 3): point anomalies on X.ETR
    t1, t2, t3, t4 = (ms(D_P23, 10, 0, s) for s in (0, 10, 20, 30))
    for i, t in enumerate((t1, t2, t3, t4), start=10):
        ep.append(episode(i, 2, "price_spike", "X.ETR", t, detail="multiplier=3"))
    add_scores("X.ETR", [t1, t2], [5.0, 0.5])         # t3 never scored
    add_scores("X.ETR", [t4 + 1000], [6.0])           # t4 unscored, alert one second later

    # stale runs on S.ETR: one detected, one with no visible effect, one never scored
    s1 = ms(D_P23, 11, 0, 0)
    ep.append(episode(20, 2, "stale_price", "S.ETR", s1, s1 + 3000, detail="changed_ticks=3"))
    add_scores("S.ETR", [s1 + 1000], [4.0])
    s2 = ms(D_P23, 11, 5, 0)
    ep.append(episode(21, 2, "stale_price", "S.ETR", s2, s2 + 3000, detail="changed_ticks=0"))
    s3 = ms(D_P23, 11, 10, 0)
    ep.append(episode(22, 2, "stale_price", "S2.ETR", s3, s3 + 3000, detail="changed_ticks=2"))

    # --- phase 3 (day 3, afternoon): blackouts and silence alerts
    l1 = ms(D_P23, 15, 29, 59)
    b_start = ms(D_P23, 15, 30, 1)
    # the last delivered message on the update clock is l1 (what the silence alert reports as
    # LastSeen); its millisecond TradingTime is 300 ms later
    ep.append(episode(30, 3, "feed_silence", "Q1.ETR", b_start, b_start + 120_000,
                      resume=b_start + 123_000, last_delivered=l1 + 300,
                      detail=f"blackout_s=120;delivered_before=100;last_delivered_time_ms={l1}"))
    ep.append(episode(31, 3, "feed_silence", "Q2.ETR", b_start, b_start + 120_000,
                      resume=b_start + 123_000, last_delivered=l1, detail="blackout_s=120;delivered_before=10"))
    ep.append(episode(32, 3, "feed_silence", "Q3.ETR", b_start, b_start + 120_000,
                      resume=b_start + 123_000, detail="blackout_s=120;delivered_before=100"))
    ep.append(episode(33, 3, "feed_silence", "IDX", b_start, b_start + 120_000,
                      resume=b_start + 123_000, last_delivered=l1, detail="delivered_before=100", sectype="I"))
    for inst in ("R.ETR", "K1.ETR", "K2.ETR", "Q1.ETR", "Q3.ETR"):
        add_scores(inst, [b_start + 30_000], 0.1)

    silence = pd.DataFrame([
        # answers Q1: about the silence that began at its last delivered tick
        dict(Exchange="ETR", Instrument="Q1.ETR", AlertType="SILENCE", LastSeenMs=l1, DetectedAtMs=l1 + 500,
             ObservedAtMs=b_start + 123_000, ElapsedMs=125_000, ThresholdMs=500, ExpectedIntervalMs=100,
             Level="SEVERE", Trigger="resume", WallTimeMs=0),
        # answers Q3 (no LastDelivered): detected inside the blackout
        dict(Exchange="ETR", Instrument="Q3.ETR", AlertType="SILENCE", LastSeenMs=l1 - 5000,
             DetectedAtMs=b_start + 4000, ObservedAtMs=b_start + 123_000, ElapsedMs=130_000, ThresholdMs=500,
             ExpectedIntervalMs=100, Level="SEVERE", Trigger="resume", WallTimeMs=0),
        # a natural silence on an unaffected instrument: a false alert
        dict(Exchange="ETR", Instrument="R.ETR", AlertType="SILENCE", LastSeenMs=b_start,
             DetectedAtMs=b_start + 1000, ObservedAtMs=b_start + 9000, ElapsedMs=9000, ThresholdMs=500,
             ExpectedIntervalMs=100, Level="SEVERE", Trigger="resume", WallTimeMs=0),
        # end-of-stream artefact, excluded by default
        dict(Exchange="ETR", Instrument="K1.ETR", AlertType="SILENCE", LastSeenMs=b_start,
             DetectedAtMs=b_start + 1000, ObservedAtMs=b_start + 9000, ElapsedMs=9000, ThresholdMs=500,
             ExpectedIntervalMs=100, Level="SEVERE", Trigger="flush", WallTimeMs=0),
    ])

    # --- phase 4 (day 4): validator episodes
    o1, o2 = ms(D_P4, 10, 0, 0), ms(D_P4, 10, 1, 0)
    ep.append(episode(40, 4, "malformed_isin", "M.ETR", o1, detail="corruption=random_chars"))
    ep.append(episode(41, 4, "malformed_isin", "M.ETR", o2, detail="corruption=random_chars"))
    i1, i2, i3 = (ms(D_P4, 11, 0, s) for s in (0, 10, 20))
    ep.append(episode(42, 4, "timestamp_inversion", "T.ETR", i1 - 60_000, observed=i1 - 60_000,
                      detail=f"rewind_s=60;prev_ms={i1}"))            # detectable, detected
    ep.append(episode(43, 4, "timestamp_inversion", "T.ETR", i2, observed=i2 - 60_000,
                      detail=f"rewind_s=60;prev_ms={i2 - 60_100}"))   # rewound to before prev? no: undetectable
    ep.append(episode(44, 4, "timestamp_inversion", "T.ETR", i3, observed=i3 - 60_000,
                      detail=f"rewind_s=60;prev_ms={i3}"))            # detectable, missed
    n1 = ms(D_P4, 12, 0, 0)
    ep.append(episode(45, 4, "implausible_price", "N.ETR", n1, detail="factor=50.00;direction=up"))
    ep.append(episode(46, 4, "null_price", "N.ETR", n1 + 5000, detail="field=Last"))  # legacy, undetectable
    add_scores("N.ETR", [n1], [9.0])

    validation = pd.DataFrame([
        dict(Exchange="ETR", Instrument="M.ETR", AlertType="MALFORMED_ISIN", TickTimeMs=o1, ReferenceTimeMs=None,
             Detail="isin=XXX", WallTimeMs=0),
        dict(Exchange="ETR", Instrument="M.ETR", AlertType="MALFORMED_ISIN", TickTimeMs=o2 + 500_000,
             ReferenceTimeMs=None, Detail="isin=XXX", WallTimeMs=0),   # stray: no such episode
        dict(Exchange="ETR", Instrument="T.ETR", AlertType="TIMESTAMP_INVERSION", TickTimeMs=i1 - 60_000,
             ReferenceTimeMs=i1, Detail="backward_ms=60000", WallTimeMs=0),
    ])

    # rows and trades per instrument and day, as the simulator writes them (before injection)
    instruments = [
        (D_P1, "P.ETR", 20000, 900), (D_P1, "C1.ETR", 50, 2), (D_P1, "C2.ETR", 500, 30), (D_P1, "C3.ETR", 5000, 300),
        (D_P23, "X.ETR", 30000, 300), (D_P23, "S.ETR", 800, 5), (D_P23, "S2.ETR", 300, 0),
        (D_P23, "Q1.ETR", 5000, 200), (D_P23, "Q2.ETR", 50, 3), (D_P23, "Q3.ETR", 50, 2),
        (D_P4, "M.ETR", 400, 10), (D_P4, "T.ETR", 900, 40), (D_P4, "N.ETR", 900, 60),
    ]
    pd.DataFrame([{"Date": d, "Exchange": "ETR", "InstrumentID": i, "SecType": "E", "Rows": r, "Trades": t}
                  for d, i, r, t in instruments]).to_csv(tmp_path / "anomaly_log_instruments.csv", index=False)

    episodes_path = tmp_path / "anomaly_log_episodes.csv"
    pd.DataFrame(ep, columns=EPISODE_COLUMNS).to_csv(episodes_path, index=False)
    scores_path = tmp_path / "scores_rrcf.parquet"
    pd.DataFrame(scores, columns=["exchange", "instrument", "timestamp_ms", "z_score"]).to_parquet(scores_path)
    silence_path = tmp_path / "silence.csv"
    silence.to_csv(silence_path, index=False)
    validation_path = tmp_path / "validation.csv"
    validation.to_csv(validation_path, index=False)
    return episodes_path, scores_path, silence_path, validation_path


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("eval")
    episodes, scores, silence, validation = build_dataset(tmp)
    args = ev.parse_args([
        "--episodes", str(episodes), "--scores", str(scores),
        "--silence-log", str(silence), "--validation-log", str(validation),
        "--output", str(tmp / "out"), "--thresholds", "1,2,3,4,10", "--bootstrap", "200",
        "--mult-thresholds", "5,10",
    ])
    return ev.evaluate(args)


def test_point_anomalies_strict_and_lenient(results):
    block = results["rrcf"]["phase2"]["point"]["price_spike"]
    strict, lenient = block["strict_exact_tick"], block["lenient_within_window"]

    assert strict["episodes"] == 4
    assert strict["scorable"] == 2                      # t3 and t4 have no score on the tick itself
    assert strict["recall_scorable"] == pytest.approx(0.5)   # t1 alerts, t2 does not
    assert strict["recall_all"] == pytest.approx(0.25)

    # t4's vector was skipped by the stride, but the next vector alerts: lenient sees it
    # as an alert, yet t4 is not scorable so it does not count towards scorable recall
    assert lenient["scorable"] == 2
    assert lenient["recall_scorable"] == pytest.approx(0.5)


def test_stale_runs_only_count_effective_ones(results):
    stale = results["rrcf"]["phase2"]["stale_price"]
    assert stale["episodes_total"] == 3
    assert stale["episodes_effective"] == 2             # the changed_ticks=0 run is invisible by design
    assert stale["scorable"] == 1                       # S2.ETR never scored
    assert stale["recall_scorable"] == pytest.approx(1.0)


def test_phase1_compares_affected_with_control(results):
    p1 = results["rrcf"]["phase1"]
    assert p1["affected"]["episodes"] == 1
    assert p1["affected"]["recall_scorable"] == pytest.approx(1.0)
    assert p1["control_unaffected"]["instruments"] == 3
    assert p1["control_unaffected"]["detection_rate"] == pytest.approx(1 / 3)


def test_phase3_matches_by_last_seen_and_ignores_cold_and_index(results):
    p3 = results["phase3_feed_silence"]
    assert p3["episodes"] == 3                          # the index (SecType I) episode is dropped
    assert p3["warm_episodes"] == 2                     # Q2 has only 10 delivered ticks

    block = p3["by_multiplier"]["5.0"]
    assert block["episodes"] == 2 and block["recall_scorable"] == pytest.approx(1.0)
    assert block["latency_from_last_tick_ms"]["median"] == pytest.approx(500)

    prec = block["alert_precision_in_scope"]
    assert prec["alerts"] == 3                          # Q1, Q3 and the stray R alert; flush excluded
    assert prec["matched_to_blackouts"] == 2
    assert prec["precision"] == pytest.approx(2 / 3)
    assert block["control_unaffected"]["alert_rate"] == pytest.approx(1 / 3)   # R alerted, K1/K2 did not


def test_higher_multiplier_can_only_lose_alerts(results):
    by_mult = results["phase3_feed_silence"]["by_multiplier"]
    assert by_mult["10.0"]["alert_precision_in_scope"]["alerts"] <= by_mult["5.0"]["alert_precision_in_scope"]["alerts"]


def test_validator_recall_and_false_alerts(results):
    v = results["phase4_validator"]

    isin = v["malformed_isin"]
    assert isin["recall_scorable"] == pytest.approx(0.5)
    assert isin["alerts"] == 2 and isin["false_alerts"] == 1

    ts = v["timestamp_inversion"]
    assert ts["episodes"] == 3
    assert ts["recall_all"] == pytest.approx(1 / 3)
    assert ts["detectable_episodes"] == 2               # one rewind stays after the previous tick
    assert ts["recall_detectable"] == pytest.approx(0.5)
    assert ts["false_alerts"] == 0


def test_implausible_price_is_evaluated_by_rrcf_and_null_price_is_not(results):
    price = results["rrcf"]["phase4_implausible_price"]
    assert price["not_evaluated_null_price_episodes"] == 1
    assert price["strict_exact_tick"]["episodes"] == 1
    assert price["strict_exact_tick"]["recall_scorable"] == pytest.approx(1.0)


def test_false_alarm_rate_uses_clean_days_without_warmup(results):
    far = results["false_alarms"]["rrcf"]
    days = far["days"]
    assert days["warmup_excluded"] == [D_WARM]
    assert days["headline"] == [D_CLEAN]
    assert set(days["injected"]) == {D_P1, D_P23, D_P4}

    at2 = far["by_threshold"]["2.0"]
    assert at2["headline_alerts_per_1000_vectors"] == pytest.approx(4.0)   # 4 alerts in 1000 vectors
    assert far["by_threshold"]["4.0"]["headline_alerts_per_1000_vectors"] == pytest.approx(0.0)
    # the warm-up day is still reported, just not in the headline
    assert at2["per_day"][D_WARM]["alerts"] == 5


def test_sweep_is_monotonic_in_false_alarms(results):
    far = [row["clean_alerts_per_1000_vectors"] for row in results["sweep"]]
    assert [row["threshold"] for row in results["sweep"]] == [1.0, 2.0, 3.0, 4.0, 10.0]
    assert far == sorted(far, reverse=True)


def test_results_record_input_fingerprints(results):
    for name in ("episodes", "scores", "silence_log", "validation_log"):
        info = results["inputs"][name]
        assert len(info["sha256"]) == 64 and info["rows"] > 0
    assert results["data_checks"]["episode_instruments_present_in_scores"] > 0


def test_target_far_fixes_the_operating_threshold_from_clean_days(tmp_path):
    episodes, scores, silence, validation = build_dataset(tmp_path)
    args = ev.parse_args([
        "--episodes", str(episodes), "--scores", str(scores), "--output", str(tmp_path / "o"),
        "--thresholds", "1,2,3,4", "--target-far", "1.0", "--bootstrap", "0",
    ])
    out = ev.evaluate(args)
    # thresholds 1-3 give 4/1000 on the clean day; 4 gives 0, the first within target
    assert out["operating_threshold"] == 4.0


def test_instrument_naming_mismatch_is_reported(tmp_path):
    episodes, scores, *_ = build_dataset(tmp_path)
    df = pd.read_parquet(scores)
    df["exchange"] = "XETRA"                             # ground truth says ETR
    df.to_parquet(scores)
    args = ev.parse_args(["--episodes", str(episodes), "--scores", str(scores), "--output", str(tmp_path / "o")])
    with pytest.raises(SystemExit, match="naming"):
        ev.evaluate(args)


def test_cli_writes_results_and_sweep(tmp_path):
    episodes, scores, silence, validation = build_dataset(tmp_path)
    out = tmp_path / "out"
    rc = ev.main([
        "--episodes", str(episodes), "--scores", str(scores), "--silence-log", str(silence),
        "--validation-log", str(validation), "--output", str(out), "--thresholds", "2,3",
        "--bootstrap", "50", "--archive-inputs", "--method-name", "unit-test",
    ])
    assert rc == 0
    data = json.loads((out / "evaluation_results.json").read_text())
    assert data["method"] == "unit-test"
    assert (out / "sweep.csv").exists()
    assert (out / "inputs" / "anomaly_log_episodes.csv").exists()
    assert (out / "inputs" / "anomaly_log_instruments.csv").exists()


def test_baseline_run_without_rule_based_logs(tmp_path):
    episodes, scores, *_ = build_dataset(tmp_path)
    args = ev.parse_args(["--episodes", str(episodes), "--scores", str(scores), "--output", str(tmp_path / "o"),
                          "--bootstrap", "0"])
    out = ev.evaluate(args)
    assert "not evaluated" in out["phase3_feed_silence"]["note"]
    assert "not evaluated" in out["phase4_validator"]["malformed_isin"]["note"]
    assert out["rrcf"]["phase2"]["point"]["price_spike"]["strict_exact_tick"]["recall_scorable"] == pytest.approx(0.5)


def test_deprecated_ground_truth_flags_resolve_to_the_episode_file(tmp_path):
    """The old make target passes --ground-truth-csv; it must keep working."""
    episodes, scores, *_ = build_dataset(tmp_path)
    legacy_tick_log = tmp_path / "anomaly_log.csv"
    legacy_tick_log.write_text("Timestamp,InstrumentID\n")
    episodes.rename(tmp_path / "anomaly_log_episodes.csv")

    args = ev.parse_args([
        "--ground-truth-csv", str(legacy_tick_log),
        "--ground-truth-manifest", str(tmp_path / "injection_manifest.json"),
        "--scores", str(scores), "--output", str(tmp_path / "o"),
    ])
    assert args.episodes == str(tmp_path / "anomaly_log_episodes.csv")

    with pytest.raises(SystemExit):
        ev.parse_args(["--scores", str(scores), "--output", str(tmp_path / "o")])


def test_results_are_stratified_by_activity(results):
    # phase 2 spikes: all four on X.ETR, which traded 300 times that day
    spike = results["rrcf"]["phase2"]["point"]["price_spike"]["strict_by_trades_that_day"]
    assert spike["200+"]["episodes"] == 4
    assert "1-19" not in spike

    # stale runs: the effective one on S.ETR (5 trades) and the never-scored one on S2.ETR (0 trades)
    stale = results["rrcf"]["phase2"]["stale_price"]["by_trades_that_day"]
    assert stale["1-19"]["episodes"] == 1 and stale["1-19"]["recall_scorable"] == pytest.approx(1.0)
    assert stale["0"]["episodes"] == 1 and stale["0"]["scorable"] == 0

    # phase 3: warm blackouts Q1 (5000 messages) and Q3 (50 messages)
    tiers = results["phase3_feed_silence"]["by_multiplier"]["5.0"]["by_messages_that_day"]
    assert tiers["1k-9,999"]["episodes"] == 1 and tiers["<100"]["episodes"] == 1
    assert tiers["1k-9,999"]["recall_scorable"] == pytest.approx(1.0)

    # phase 1: one busy affected instrument, three controls of different activity
    p1 = results["rrcf"]["phase1"]["by_messages_that_day"]
    assert p1["10k+"]["affected_instruments"] == 1
    assert p1["<100"]["control_instruments"] == 1 and p1["100-999"]["control_instruments"] == 1
    assert p1["1k-9,999"]["control_instruments"] == 1
    assert p1["<100"]["control_detection_rate"] == pytest.approx(0.0) or p1["<100"]["control_detection_rate"] == pytest.approx(1.0)


def test_instruments_file_is_found_next_to_the_episodes(tmp_path):
    episodes, scores, *_ = build_dataset(tmp_path)
    args = ev.parse_args(["--episodes", str(episodes), "--scores", str(scores), "--output", str(tmp_path / "o")])
    assert args.instruments == str(tmp_path / "anomaly_log_instruments.csv")


def test_without_the_instruments_file_strata_are_simply_absent(tmp_path):
    episodes, scores, *_ = build_dataset(tmp_path)
    (tmp_path / "anomaly_log_instruments.csv").unlink()
    args = ev.parse_args(["--episodes", str(episodes), "--scores", str(scores), "--output", str(tmp_path / "o"),
                          "--bootstrap", "0"])
    assert args.instruments is None
    out = ev.evaluate(args)
    assert out["rrcf"]["phase2"]["point"]["price_spike"]["strict_by_trades_that_day"] is None
    assert out["rrcf"]["phase2"]["point"]["price_spike"]["strict_exact_tick"]["recall_scorable"] == pytest.approx(0.5)


def test_method_name_defaults_to_the_scores_file_name_and_keys_the_results(tmp_path):
    for fname, expected in [("scores_zscore.parquet", "zscore"), ("scores_isoforest.parquet", "isoforest"),
                            ("other.parquet", "other")]:
        args = ev.parse_args(["--episodes", "x_episodes.csv", "--scores", str(tmp_path / fname),
                              "--output", str(tmp_path / "o")])
        assert args.method_name == expected
    args = ev.parse_args(["--episodes", "x_episodes.csv", "--scores", str(tmp_path / "scores_zscore.parquet"),
                          "--output", str(tmp_path / "o"), "--method-name", "mine"])
    assert args.method_name == "mine"
