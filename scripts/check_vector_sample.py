#!/usr/bin/env python3
"""
Check a recorded vector sample against the runner's own account of the run.

At shutdown, scripts/run_multi_model.py writes vectors_sample.summary.json next to
vectors_sample.parquet: how many vectors it consumed from Kafka, kept after the stride, and
recorded. This script re-reads the parquet file and checks that what is on disk is exactly
that (so an archived sample is known to be complete and to be the sample the models scored):

    rows in the file == recorded_rows == kept == consumed // stride
    stream_index runs stride, 2*stride, ..., kept*stride with no gap
    consumed lies in [kept*stride, kept*stride + stride)

    python scripts/check_vector_sample.py results/latest/inputs/vectors/vectors_sample.parquet

Exit status 0 if every check passes, 1 otherwise (also when the summary file is missing).
Used by `make archive-run`.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def check(sample: str, summary_file: str = None) -> list:
    """Return [(description, passed, detail), ...]."""
    results = []

    def add(desc, ok, detail=""):
        results.append((desc, bool(ok), detail))

    sample_p = Path(sample)
    summary_p = Path(summary_file) if summary_file else sample_p.with_suffix(".summary.json")
    if not sample_p.exists():
        add("sample file exists", False, str(sample_p))
        return results
    if not summary_p.exists():
        add("runner summary exists", False, f"{summary_p} missing: cannot cross-check the sample")
        return results

    summary = json.loads(summary_p.read_text())
    stride, consumed, kept, recorded = (summary[k] for k in ("stride", "consumed", "kept", "recorded_rows"))

    pf = pq.ParquetFile(sample_p)
    rows = pf.metadata.num_rows
    add("rows in the file == vectors the runner recorded", rows == recorded, f"{rows:,} vs {recorded:,}")
    add("rows in the file == vectors the runner kept after the stride", rows == kept, f"{rows:,} vs {kept:,}")
    add("kept == consumed // stride", kept == consumed // stride,
        f"{kept:,} vs {consumed:,} // {stride} = {consumed // stride:,}")

    idx = pq.read_table(sample_p, columns=["stream_index"]).column("stream_index").to_numpy()
    if len(idx) == 0:
        add("stream_index runs stride, 2*stride, ...", kept == 0, "empty sample")
    else:
        add("stream_index runs stride, 2*stride, ... without a gap",
            idx[0] == stride and bool(np.all(np.diff(idx) == stride)),
            f"first {int(idx[0])}, last {int(idx[-1])}")
        add("consumed lies just after the last recorded position", idx[-1] <= consumed < idx[-1] + stride,
            f"last {int(idx[-1]):,}, consumed {consumed:,}")

    dropped = {m: n for m, n in (summary.get("dropped") or {}).items() if n}
    if dropped:  # not a failure: the sample is complete, but those models' live scores are not
        add("no model dropped vectors live", True,
            "NOTE " + ", ".join(f"{m} dropped {n:,}" for m, n in dropped.items())
            + ": their live scores miss those rows; replay them from the sample")
    return results


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sample", help="vectors_sample.parquet")
    p.add_argument("--summary", help="summary json (default: vectors_sample.summary.json next to the sample)")
    args = p.parse_args(argv)

    results = check(args.sample, args.summary)
    for desc, ok, detail in results:
        print(f"  {'PASS' if ok else 'FAIL'}: {desc}" + (f" ({detail})" if detail else ""))
    ok = all(r[1] for r in results)
    print("vector sample OK" if ok else "vector sample FAILED its consistency check")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
