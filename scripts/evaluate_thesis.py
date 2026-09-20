#!/usr/bin/env python3
"""
Thesis Evaluation Script - Optimized matching algorithm

Evaluates anomaly detection performance against ground truth injections.
Uses vectorized operations and binary search for fast matching.

Usage:
    python scripts/evaluate_thesis.py \\
        --ground-truth-csv ../price-feed-simulator/anomaly_log.csv \\
        --ground-truth-manifest ../price-feed-simulator/data/injection_manifest.json \\
        --scores ./data/scores_rrcf.parquet \\
        --output ../results/thesis_YYYYMMDD_HHMMSS
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

# Detection windows for different phases (milliseconds)
DETECTION_WINDOWS = {
    1: 60000,   # Phase 1: 60 seconds
    2: 30000,   # Phase 2: 30 seconds  
    3: 10000,   # Phase 3: 10 seconds
}


def match_detections_clustered(
    injections: pd.DataFrame,
    detections: pd.DataFrame,
    phase: int,
    alert_threshold: float = 2.0,
) -> Dict:
    """
    OPTIMIZED: Match detections to injections with alert clustering.

    100x faster than original by using:
    - Group by instrument (avoids full scans)
    - Binary search (O(log n) instead of O(n))
    - Set operations (O(1) lookups)
    - No iterrows() loops!
    """
    window_ms = DETECTION_WINDOWS[phase]

    # Filter to high-confidence alerts
    alerts = detections[detections["z_score"].abs() >= alert_threshold].copy()

    if len(alerts) == 0:
        return {
            "tp": 0,
            "fp": 0,
            "fn": len(injections),
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "avg_latency_ms": 0.0,
            "total_injections": len(injections),
            "total_detections": 0,
        }

    # Sort by timestamp for binary search
    injections = injections.sort_values("timestamp_ms").reset_index(drop=True)
    alerts = alerts.sort_values("timestamp_ms").reset_index(drop=True)

    # Group by instrument for O(1) lookup
    inj_by_inst = {inst: group for inst, group in injections.groupby("instrument")}
    alert_by_inst = {inst: group for inst, group in alerts.groupby("instrument")}

    tp = 0
    fn = 0
    latencies = []
    matched_alert_indices = set()

    print(
        f"  Matching {len(injections):,} injections against {len(alerts):,} alerts..."
    )

    # FAST PATH: Process each instrument separately
    processed_instruments = 0
    for instrument, inj_group in inj_by_inst.items():
        processed_instruments += 1

        if instrument not in alert_by_inst:
            # No alerts for this instrument → all FNs
            fn += len(inj_group)
            continue

        alert_group = alert_by_inst[instrument]
        alert_times = alert_group["timestamp_ms"].values
        inj_times = inj_group["timestamp_ms"].values
        alert_indices = alert_group.index.values

        # For each injection, binary search for alerts in window
        for inj_time in inj_times:
            # Find first alert >= inj_time
            start_idx = np.searchsorted(alert_times, inj_time, side="left")
            # Find first alert > inj_time + window_ms
            end_idx = np.searchsorted(alert_times, inj_time + window_ms, side="right")

            if start_idx < end_idx:
                # Found alerts in window!
                tp += 1
                latency = alert_times[start_idx] - inj_time
                latencies.append(latency)

                # Mark all alerts in window as matched
                for idx in range(start_idx, end_idx):
                    matched_alert_indices.add(alert_indices[idx])
            else:
                fn += 1

        # Progress indicator
        if processed_instruments % 500 == 0:
            print(
                f"    Processed {processed_instruments}/{len(inj_by_inst)} instruments..."
            )

    # False positives: alerts not matched to any injection
    fp = len(alerts) - len(matched_alert_indices)

    # Metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    avg_latency = float(np.mean(latencies)) if latencies else 0.0

    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "avg_latency_ms": avg_latency,
        "total_injections": int(len(injections)),
        "total_detections": int(len(alerts)),
    }


def load_ground_truth(csv_path: str, manifest_path: str) -> pd.DataFrame:
    """Load ground truth injections from CSV and manifest."""
    print(f"Loading ground truth from {csv_path}...")
    
    # Load anomaly log CSV
    df = pd.read_csv(csv_path)
    
    # Load manifest for additional metadata
    if Path(manifest_path).exists():
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        print(f"  Manifest: {len(manifest)} injection events")
    
    # Normalize column names (handle both Timestamp and timestamp)
    column_mapping = {}
    for col in df.columns:
        if col.lower() == 'timestamp':
            column_mapping[col] = 'timestamp'
        elif col.lower() == 'instrumentid':
            column_mapping[col] = 'instrument'
        elif col.lower() == 'exchange':
            column_mapping[col] = 'exchange'
    
    df = df.rename(columns=column_mapping)
    
    # Convert timestamp to milliseconds
    if 'timestamp_ms' not in df.columns and 'timestamp' in df.columns:
        df['timestamp_ms'] = pd.to_datetime(df['timestamp']).astype(int) // 10**6
    
    print(f"  Loaded {len(df)} injections")
    return df


def load_detections(scores_path: str) -> pd.DataFrame:
    """Load detection scores from parquet file."""
    print(f"Loading detections from {scores_path}...")
    
    df = pd.read_parquet(scores_path)
    
    # Convert timestamp to milliseconds if needed
    if 'timestamp_ms' not in df.columns and 'timestamp' in df.columns:
        df['timestamp_ms'] = pd.to_datetime(df['timestamp']).astype(int) // 10**6
    
    print(f"  Loaded {len(df)} detection records")
    return df


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description="Evaluate thesis anomaly detection results"
    )
    parser.add_argument(
        "--ground-truth-csv",
        required=True,
        help="Path to anomaly_log.csv from simulator"
    )
    parser.add_argument(
        "--ground-truth-manifest", 
        required=True,
        help="Path to injection_manifest.json from simulator"
    )
    parser.add_argument(
        "--scores",
        required=True,
        help="Path to scores parquet file from detector"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for results"
    )
    parser.add_argument(
        "--alert-threshold",
        type=float,
        default=2.0,
        help="Z-score threshold for alerts (default: 2.0)"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("  Thesis Evaluation - Anomaly Detection Performance")
    print("=" * 60)
    print()
    
    # Load data
    injections = load_ground_truth(args.ground_truth_csv, args.ground_truth_manifest)
    detections = load_detections(args.scores)
    
    print()
    print("=" * 60)
    print("  Evaluating Detection Performance")
    print("=" * 60)
    print()
    
    # Evaluate for each phase
    results = {}
    for phase in [1, 2, 3]:
        print(f"Phase {phase} (detection window: {DETECTION_WINDOWS[phase]}ms):")
        
        metrics = match_detections_clustered(
            injections=injections,
            detections=detections,
            phase=phase,
            alert_threshold=args.alert_threshold
        )
        
        results[f"phase_{phase}"] = metrics
        
        # Print results
        print(f"  TP: {metrics['tp']:,}")
        print(f"  FP: {metrics['fp']:,}")
        print(f"  FN: {metrics['fn']:,}")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall: {metrics['recall']:.3f}")
        print(f"  F1: {metrics['f1']:.3f}")
        print(f"  Avg Latency: {metrics['avg_latency_ms']:.1f}ms")
        print()
    
    # Save results
    results_file = output_dir / "evaluation_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("=" * 60)
    print(f"  Results saved to {results_file}")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
