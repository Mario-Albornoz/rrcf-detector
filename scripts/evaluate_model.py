#!/usr/bin/env python3
"""
Evaluate RRCF anomaly detector performance.

Loads collected data from parquet files, merges input/output, computes metrics,
and generates evaluation plots for thesis analysis.

Usage:
    python scripts/evaluate_model.py \\
        --data-dir ./data/run_001 \\
        --ground-truth ./ground_truth/anomalies.json \\
        --output-dir ./results/run_001

Ground truth format (JSON):
    [
        {
            "exchange": "binance",
            "instrument": "BTC-USDT",
            "timestamp": "2024-01-01T12:00:00",
            "type": "silence|lag|spike|gradual"
        },
        ...
    ]
"""

import argparse
import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def load_data(data_dir: str) -> tuple:
    """Load input and output parquet files."""
    input_file = os.path.join(data_dir, "input_vectors.parquet")
    output_file = os.path.join(data_dir, "output_scores.parquet")
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    if not os.path.exists(output_file):
        raise FileNotFoundError(f"Output file not found: {output_file}")
    
    input_df = pd.read_parquet(input_file)
    output_df = pd.read_parquet(output_file)
    
    print(f"Loaded {len(input_df)} input vectors")
    print(f"Loaded {len(output_df)} output scores")
    
    return input_df, output_df


def merge_data(input_df: pd.DataFrame, output_df: pd.DataFrame) -> pd.DataFrame:
    """Merge input and output by (exchange, instrument, timestamp)."""
    
    if "timestamp" in input_df.columns:
        input_df["timestamp_str"] = input_df["timestamp"]
    if "timestamp" in output_df.columns:
        output_df["timestamp_str"] = output_df["timestamp"]
    
    merge_keys = ["exchange", "instrument", "timestamp_str"]
    
    merged = pd.merge(
        input_df,
        output_df,
        on=merge_keys,
        how="inner",
        suffixes=("_input", "_output")
    )
    
    print(f"Merged to {len(merged)} rows")
    return merged


def load_ground_truth(filepath: str) -> pd.DataFrame:
    """Load ground truth anomalies from JSON file."""
    if not filepath or not os.path.exists(filepath):
        return pd.DataFrame()
    
    with open(filepath, 'r') as f:
        anomalies = json.load(f)
    
    df = pd.DataFrame(anomalies)
    df["is_anomaly"] = True
    print(f"Loaded {len(df)} ground truth anomalies")
    return df


def compute_metrics(merged_df: pd.DataFrame, ground_truth_df: pd.DataFrame, threshold_strategies: dict) -> dict:
    """Compute evaluation metrics for different threshold strategies."""
    
    if ground_truth_df.empty:
        print("⚠ No ground truth provided, skipping metric computation")
        return {}
    
    merged_with_labels = merged_df.merge(
        ground_truth_df[["exchange", "instrument", "timestamp", "is_anomaly"]],
        on=["exchange", "instrument", "timestamp_str"],
        how="left"
    )
    merged_with_labels["is_anomaly"] = merged_with_labels["is_anomaly"].fillna(False)
    
    y_true = merged_with_labels["is_anomaly"].astype(int)
    
    results = {}
    
    for strategy_name, threshold_fn in threshold_strategies.items():
        y_pred = threshold_fn(merged_with_labels).astype(int)
        
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        results[strategy_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": int(((y_pred == 1) & (y_true == 1)).sum()),
            "fp": int(((y_pred == 1) & (y_true == 0)).sum()),
            "tn": int(((y_pred == 0) & (y_true == 0)).sum()),
            "fn": int(((y_pred == 0) & (y_true == 1)).sum()),
        }
        
        print(f"\n{strategy_name}:")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall: {recall:.3f}")
        print(f"  F1: {f1:.3f}")
    
    return results


def plot_roc_curve(merged_df: pd.DataFrame, ground_truth_df: pd.DataFrame, output_dir: str):
    """Generate ROC curve plot."""
    if ground_truth_df.empty:
        print("⚠ Skipping ROC curve (no ground truth)")
        return
    
    merged_with_labels = merged_df.merge(
        ground_truth_df[["exchange", "instrument", "timestamp", "is_anomaly"]],
        on=["exchange", "instrument", "timestamp_str"],
        how="left"
    )
    merged_with_labels["is_anomaly"] = merged_with_labels["is_anomaly"].fillna(False)
    
    y_true = merged_with_labels["is_anomaly"].astype(int)
    y_scores = merged_with_labels["z_score"].abs()
    
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, linewidth=2, label=f"RRCF (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label="Random")
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curve - RRCF Anomaly Detection", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "roc_curve.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✓ Saved ROC curve to {output_path}")


def plot_score_distribution(merged_df: pd.DataFrame, output_dir: str):
    """Plot distribution of raw scores and z-scores."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].hist(merged_df["raw_score"], bins=50, alpha=0.7, edgecolor="black")
    axes[0].set_xlabel("Raw CoDisp Score", fontsize=11)
    axes[0].set_ylabel("Frequency", fontsize=11)
    axes[0].set_title("Raw Score Distribution", fontsize=12)
    axes[0].grid(alpha=0.3)
    
    axes[1].hist(merged_df["z_score"], bins=50, alpha=0.7, edgecolor="black", color="orange")
    axes[1].axvline(2, color="yellow", linestyle="--", linewidth=2, label="z=2 (medium)")
    axes[1].axvline(3, color="red", linestyle="--", linewidth=2, label="z=3 (high)")
    axes[1].set_xlabel("Z-Score", fontsize=11)
    axes[1].set_ylabel("Frequency", fontsize=11)
    axes[1].set_title("Z-Score Distribution (Calibrated)", fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "score_distribution.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✓ Saved score distribution to {output_path}")


def plot_alert_level_distribution(merged_df: pd.DataFrame, output_dir: str):
    """Plot distribution of alert levels."""
    alert_counts = merged_df["alert_level"].value_counts()
    
    plt.figure(figsize=(8, 6))
    colors = {"normal": "green", "medium": "orange", "high": "red"}
    bars = plt.bar(
        alert_counts.index,
        alert_counts.values,
        color=[colors.get(x, "gray") for x in alert_counts.index],
        edgecolor="black"
    )
    plt.xlabel("Alert Level", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.title("Alert Level Distribution", fontsize=14)
    plt.grid(axis="y", alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontsize=10
        )
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "alert_levels.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✓ Saved alert levels to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate RRCF model performance"
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Directory with input_vectors.parquet and output_scores.parquet"
    )
    parser.add_argument(
        "--ground-truth",
        help="Path to ground truth anomalies JSON file (optional)"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save evaluation results"
    )
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("RRCF Model Evaluation")
    print("=" * 60)
    print()
    
    input_df, output_df = load_data(args.data_dir)
    merged_df = merge_data(input_df, output_df)
    ground_truth_df = load_ground_truth(args.ground_truth) if args.ground_truth else pd.DataFrame()
    
    print("\n" + "=" * 60)
    print("Computing Metrics")
    print("=" * 60)
    
    threshold_strategies = {
        "z_score_2": lambda df: df["z_score"].abs() >= 2.0,
        "z_score_3": lambda df: df["z_score"].abs() >= 3.0,
        "alert_medium_or_high": lambda df: df["alert_level"].isin(["medium", "high"]),
        "alert_high_only": lambda df: df["alert_level"] == "high",
    }
    
    metrics = compute_metrics(merged_df, ground_truth_df, threshold_strategies)
    
    if metrics:
        metrics_file = os.path.join(args.output_dir, "metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\n✓ Saved metrics to {metrics_file}")
    
    print("\n" + "=" * 60)
    print("Generating Plots")
    print("=" * 60)
    print()
    
    plot_score_distribution(merged_df, args.output_dir)
    plot_alert_level_distribution(merged_df, args.output_dir)
    
    if not ground_truth_df.empty:
        plot_roc_curve(merged_df, ground_truth_df, args.output_dir)
    
    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
