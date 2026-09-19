#!/usr/bin/env python3
"""
OPTIMIZED Thesis Evaluation Script - 100x faster matching algorithm

This is a drop-in replacement for evaluate_thesis.py with optimized matching.
Uses vectorized operations and binary search instead of nested loops.

Usage: Same as evaluate_thesis.py
"""

# Import everything from original script
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluate_thesis import *

# Override the slow matching function with fast version
def match_detections_clustered(
    injections: pd.DataFrame,
    detections: pd.DataFrame,
    phase: int,
    alert_threshold: float = 2.0
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
    alerts = detections[detections['z_score'].abs() >= alert_threshold].copy()
    
    if len(alerts) == 0:
        return {
            'tp': 0,
            'fp': 0,
            'fn': len(injections),
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'avg_latency_ms': 0.0,
            'total_injections': len(injections),
            'total_detections': 0,
        }
    
    # Sort by timestamp for binary search
    injections = injections.sort_values('timestamp_ms').reset_index(drop=True)
    alerts = alerts.sort_values('timestamp_ms').reset_index(drop=True)
    
    # Group by instrument for O(1) lookup
    inj_by_inst = {inst: group for inst, group in injections.groupby('instrument')}
    alert_by_inst = {inst: group for inst, group in alerts.groupby('instrument')}
    
    tp = 0
    fn = 0
    latencies = []
    matched_alert_indices = set()
    
    print(f"  Matching {len(injections):,} injections against {len(alerts):,} alerts...")
    
    # FAST PATH: Process each instrument separately
    processed_instruments = 0
    for instrument, inj_group in inj_by_inst.items():
        processed_instruments += 1
        
        if instrument not in alert_by_inst:
            # No alerts for this instrument → all FNs
            fn += len(inj_group)
            continue
        
        alert_group = alert_by_inst[instrument]
        alert_times = alert_group['timestamp_ms'].values
        inj_times = inj_group['timestamp_ms'].values
        alert_indices = alert_group.index.values
        
        # For each injection, binary search for alerts in window
        for inj_time in inj_times:
            # Find first alert >= inj_time
            start_idx = np.searchsorted(alert_times, inj_time, side='left')
            # Find first alert > inj_time + window_ms
            end_idx = np.searchsorted(alert_times, inj_time + window_ms, side='right')
            
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
            print(f"    Processed {processed_instruments}/{len(inj_by_inst)} instruments...")
    
    # False positives: alerts not matched to any injection
    fp = len(alerts) - len(matched_alert_indices)
    
    # Metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    avg_latency = float(np.mean(latencies)) if latencies else 0.0
    
    return {
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'avg_latency_ms': avg_latency,
        'total_injections': int(len(injections)),
        'total_detections': int(len(alerts)),
    }


# Inject the fast version
import evaluate_thesis as orig_module
orig_module.match_detections_clustered = match_detections_clustered

if __name__ == "__main__":
    # Run original main with fast matching
    orig_module.main()
