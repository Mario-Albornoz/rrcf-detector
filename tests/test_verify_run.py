"""Tests for scripts/verify_run.py: healthy artifacts pass, and each kind of breakage is
reported as the failure it is (the checks are only useful if they can fail)."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify_run.py"
_spec = importlib.util.spec_from_file_location("verify_run", _SCRIPT)
vr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vr)

MANIFEST = {
    "stats": {"TotalProcessed": 1_000_000, "Phase1Dropped": 1000, "Phase3Dropped": 500},
    "phases": {f"phase{i}": {"enabled": True} for i in (1, 2, 3, 4)},
}


def episodes(phases=(1, 2, 3, 4), sectype="E"):
    rows = []
    kinds = {1: "tick_rate_decline", 2: "price_spike", 3: "feed_silence", 4: "implausible_price"}
    for p in phases:
        for i in range(20):
            rows.append({"Phase": f"phase{p}", "AnomalyType": kinds[p], "Exchange": "ETR",
                         "InstrumentID": f"I{i}.ETR", "StartMs": 1_636_538_400_000 + i * 1000,
                         "EndMs": 1_636_538_400_000 + i * 1000, "ObservedMs": 1_636_538_400_000 + i * 1000,
                         "SecType": sectype})
    return pd.DataFrame(rows)


def instruments(trade_share=0.04):
    return pd.DataFrame([{"Date": "2021-11-10", "Exchange": "ETR", "InstrumentID": f"I{i}.ETR", "SecType": "E",
                          "Rows": 1000, "Trades": int(1000 * trade_share)} for i in range(20)])


def status_of(rep, check_substring):
    matches = [r for r in rep.rows if check_substring in r["check"]]
    assert matches, f"no check named like {check_substring!r}: {[r['check'] for r in rep.rows]}"
    return matches[0]["status"]


def test_healthy_ground_truth_passes():
    rep = vr.Report()
    vr.check_ground_truth(rep, MANIFEST, episodes(), instruments())
    assert rep.failed() == 0
    assert status_of(rep, "trades as a share") == "PASS"


def test_missing_phase_is_a_failure_that_names_the_cause():
    rep = vr.Report()
    vr.check_ground_truth(rep, MANIFEST, episodes(phases=(1, 2, 4)), instruments())
    assert status_of(rep, "phase3 episodes") == "FAIL"
    hint = [r for r in rep.rows if "phase3" in r["check"]][0]["hint"]
    assert "exchange_filter" in hint  # the XETRA-vs-ETR mistake this would have caught


def test_disabled_phase_is_not_expected():
    manifest = {**MANIFEST, "phases": {**MANIFEST["phases"], "phase3": {"enabled": False}}}
    rep = vr.Report()
    vr.check_ground_truth(rep, manifest, episodes(phases=(1, 2, 4)), instruments())
    assert status_of(rep, "phase3 episodes") == "SKIP"
    assert rep.failed() == 0


def test_no_trades_in_the_simulator_is_flagged():
    rep = vr.Report()
    vr.check_ground_truth(rep, MANIFEST, episodes(), instruments(trade_share=0.0))
    assert status_of(rep, "trades as a share") == "FAIL"


def test_empty_episode_file_fails():
    rep = vr.Report()
    vr.check_ground_truth(rep, MANIFEST, episodes(phases=()), None)
    assert status_of(rep, "episode file has rows") == "FAIL"


def scores(n=1000, alert_share=0.03):
    rng = np.random.default_rng(0)
    z = rng.normal(0, 1, n)
    z[: int(n * alert_share)] = 3.0
    return pd.DataFrame({"exchange": "ETR", "instrument": [f"I{i % 20}.ETR" for i in range(n)],
                         "timestamp_ms": 1_636_538_400_000 + np.arange(n) * 100, "z_score": z})


def test_healthy_scores_pass():
    rep = vr.Report()
    vr.check_scores(rep, scores(1000), vectors=10_000, episodes=episodes())
    assert rep.failed() == 0, [r for r in rep.rows if r["status"] == "FAIL"]


def test_scores_far_below_the_stride_fail():
    rep = vr.Report()
    vr.check_scores(rep, scores(100), vectors=10_000, episodes=episodes())
    assert status_of(rep, "scores vs vectors") == "FAIL"


def test_constant_scores_are_flagged():
    df = scores(1000)
    df["z_score"] = 0.0
    rep = vr.Report()
    vr.check_scores(rep, df, vectors=10_000, episodes=episodes())
    assert status_of(rep, "spread of z-scores") == "FAIL"
    assert status_of(rep, "share of scores with z >= 2") == "FAIL"


def test_nan_scores_fail():
    df = scores(1000)
    df.loc[0, "z_score"] = np.nan
    rep = vr.Report()
    vr.check_scores(rep, df, vectors=10_000, episodes=episodes())
    assert status_of(rep, "non-finite") == "FAIL"


def test_scores_for_other_instruments_fail_the_overlap_check():
    df = scores(1000)
    df["instrument"] = "OTHER.ETR"
    rep = vr.Report()
    vr.check_scores(rep, df, vectors=10_000, episodes=episodes())
    assert status_of(rep, "episode instruments present") == "FAIL"


def test_join_finds_the_injected_ticks_when_timestamps_line_up():
    ep = episodes(phases=(2,))
    sc = pd.DataFrame({"exchange": "ETR", "instrument": ep["InstrumentID"],
                       "timestamp_ms": ep["ObservedMs"], "z_score": 1.0}).iloc[:2]   # 2 of 20 scored
    rep = vr.Report()
    vr.check_join(rep, sc, ep)
    assert status_of(rep, "injected trades whose own vector") == "PASS"   # 10%, the stride


def test_join_fails_when_timestamps_do_not_line_up():
    ep = episodes(phases=(2,))
    sc = pd.DataFrame({"exchange": "ETR", "instrument": ep["InstrumentID"],
                       "timestamp_ms": ep["ObservedMs"] + 5_000, "z_score": 1.0})   # 5 s off
    rep = vr.Report()
    vr.check_join(rep, sc, ep)
    assert status_of(rep, "injected trades whose own vector") == "FAIL"


def silence(n, flush=0):
    triggers = ["resume"] * (n - flush) + ["flush"] * flush
    return pd.DataFrame({"Trigger": triggers})


def test_silence_rate_bands():
    rep = vr.Report()
    vr.check_handler_logs(rep, silence(10), None, None, vectors=10_000)   # 1 per 1000
    assert status_of(rep, "silence alerts per 1000") == "PASS"

    rep = vr.Report()
    vr.check_handler_logs(rep, silence(400), None, None, vectors=10_000)  # 40 per 1000: the old mean rule
    assert status_of(rep, "silence alerts per 1000") == "FAIL"


def test_validation_alerts_beyond_the_injected_ones_are_flagged():
    ep = pd.DataFrame({"AnomalyType": ["timestamp_inversion"] * 5})
    ok = pd.DataFrame({"AlertType": ["TIMESTAMP_INVERSION"] * 5})
    rep = vr.Report()
    vr.check_handler_logs(rep, None, ok, ep, vectors=100_000)
    assert status_of(rep, "validation alerts beyond") == "PASS"

    reordering = pd.DataFrame({"AlertType": ["TIMESTAMP_INVERSION"] * 500})   # producer reordering
    rep = vr.Report()
    vr.check_handler_logs(rep, None, reordering, ep, vectors=100_000)
    assert status_of(rep, "validation alerts beyond") in ("WARN", "FAIL")


def test_report_exit_status_reflects_failures():
    rep = vr.Report()
    rep.add("s", "a", "PASS", "x")
    assert rep.failed() == 0
    rep.add("s", "b", "FAIL", "y")
    assert rep.failed() == 1


def test_cli_runs_on_files_and_returns_nonzero_on_failure(tmp_path, capsys):
    ep = episodes(phases=(1, 2, 4))  # phase 3 missing
    ep.to_csv(tmp_path / "ep.csv", index=False)
    instruments().to_csv(tmp_path / "inst.csv", index=False)
    import json
    (tmp_path / "manifest.json").write_text(json.dumps(MANIFEST))
    rc = vr.main(["--manifest", str(tmp_path / "manifest.json"), "--episodes", str(tmp_path / "ep.csv"),
                  "--instruments", str(tmp_path / "inst.csv"), "--json", str(tmp_path / "report.json")])
    out = capsys.readouterr().out
    assert rc == 1
    assert "phase3 episodes injected" in out and "exchange_filter" in out
    assert (tmp_path / "report.json").exists()
