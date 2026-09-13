#!/usr/bin/env python3
"""
Multi-Model Anomaly Detection Evaluation

Evaluates and compares multiple anomaly detection models from streaming parquet file.
Works with data collected by stream_collector.py for real-time evaluation.

Usage:
    # Basic comparison (no ground truth)
    python scripts/evaluate_model.py \\
        --scores-file ./data/run_001/scores.parquet \\
        --output-dir ./results/run_001
    
    # With ground truth for metrics
    python scripts/evaluate_model.py \\
        --scores-file ./data/run_001/scores.parquet \\
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

Output:
    - Model comparison plots (ROC curves, score distributions)
    - Per-model metrics (if ground truth provided)
    - Alert level distributions
    - Statistical summaries
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def load_scores(scores_file: str) -> pd.DataFrame:
    """Load scores from streaming parquet file."""
    if not os.path.exists(scores_file):
        raise FileNotFoundError(f"Scores file not found: {scores_file}")
    
    df = pd.read_parquet(scores_file)
    print(f"Loaded {len(df)} score records")
    
    # Check required columns
    required = ["exchange", "instrument", "timestamp", "model", "raw_score", "z_score"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Parse timestamp if needed
    if df["timestamp"].dtype == "object":
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    return df


def load_ground_truth(filepath: str) -> pd.DataFrame:
    """Load ground truth anomalies from JSON file."""
    if not filepath or not os.path.exists(filepath):
        return pd.DataFrame()
    
    with open(filepath, "r") as f:
        anomalies = json.load(f)
    
    df = pd.DataFrame(anomalies)
    df["is_anomaly"] = True
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    print(f"Loaded {len(df)} ground truth anomalies")
    return df


def compute_per_model_metrics(
    scores_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    threshold_strategies: Dict
) -> Dict[str, Dict]:
    """
    Compute metrics for each model separately.
    
    Returns nested dict: {model_name: {strategy_name: metrics}}
    """
    if ground_truth_df.empty:
        print("⚠ No ground truth provided, skipping metric computation")
        return {}
    
    results = {}
    models = scores_df["model"].unique()
    
    for model_name in models:
        model_scores = scores_df[scores_df["model"] == model_name].copy()
        
        # Merge with ground truth
        merged = model_scores.merge(
            ground_truth_df[["exchange", "instrument", "timestamp", "is_anomaly"]],
            on=["exchange", "instrument", "timestamp"],
            how="left",
        )
        merged["is_anomaly"] = merged["is_anomaly"].fillna(False)
        
        y_true = merged["is_anomaly"].astype(int)
        
        model_results = {}
        
        for strategy_name, threshold_fn in threshold_strategies.items():
            y_pred = threshold_fn(merged).astype(int)
            
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            model_results[strategy_name] = {
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "tp": int(((y_pred == 1) & (y_true == 1)).sum()),
                "fp": int(((y_pred == 1) & (y_true == 0)).sum()),
                "tn": int(((y_pred == 0) & (y_true == 0)).sum()),
                "fn": int(((y_pred == 0) & (y_true == 1)).sum()),
            }
        
        results[model_name] = model_results
    
    # Print summary
    print("\n" + "=" * 60)
    print("Per-Model Metrics")
    print("=" * 60)
    for model_name, strategies in results.items():
        print(f"\n{model_name.upper()}:")
        for strategy_name, metrics in strategies.items():
            print(f"  {strategy_name}:")
            print(f"    Precision: {metrics['precision']:.3f}")
            print(f"    Recall: {metrics['recall']:.3f}")
            print(f"    F1: {metrics['f1']:.3f}")
    
    return results


def plot_multi_model_roc(
    scores_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    output_dir: str
):
    """Generate ROC curves for all models on same plot."""
    if ground_truth_df.empty:
        print("⚠ Skipping ROC curves (no ground truth)")
        return
    
    plt.figure(figsize=(10, 8))
    
    models = sorted(scores_df["model"].unique())
    colors = plt.cm.Set1(np.linspace(0, 1, len(models)))
    
    for model_name, color in zip(models, colors):
        model_scores = scores_df[scores_df["model"] == model_name].copy()
        
        # Merge with ground truth
        merged = model_scores.merge(
            ground_truth_df[["exchange", "instrument", "timestamp", "is_anomaly"]],
            on=["exchange", "instrument", "timestamp"],
            how="left",
        )
        merged["is_anomaly"] = merged["is_anomaly"].fillna(False)
        
        y_true = merged["is_anomaly"].astype(int)
        y_scores = merged["z_score"].abs()
        
        if y_true.sum() == 0:
            print(f"⚠ No positive samples for {model_name}, skipping ROC")
            continue
        
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = roc_auc_score(y_true, y_scores)
        
        plt.plot(fpr, tpr, color=color, linewidth=2,
                label=f"{model_name} (AUC = {roc_auc:.3f})")
    
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curves - Multi-Model Comparison", fontsize=14)
    plt.legend(fontsize=10, loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "roc_curves_comparison.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✓ Saved ROC comparison to {output_path}")


def plot_score_distributions(scores_df: pd.DataFrame, output_dir: str):
    """Plot score distributions per model (side by side)."""
    models = sorted(scores_df["model"].unique())
    n_models = len(models)
    
    # Raw scores
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5))
    if n_models == 1:
        axes = [axes]
    
    for i, model_name in enumerate(models):
        model_data = scores_df[scores_df["model"] == model_name]
        axes[i].hist(model_data["raw_score"], bins=50, alpha=0.7, edgecolor="black")
        axes[i].set_xlabel("Raw Score", fontsize=11)
        axes[i].set_ylabel("Frequency", fontsize=11)
        axes[i].set_title(f"{model_name}", fontsize=12)
        axes[i].grid(alpha=0.3)
    
    plt.suptitle("Raw Score Distributions", fontsize=14, y=1.02)
    plt.tight_layout()
    output_path = os.path.join(output_dir, "raw_score_distributions.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved raw score distributions to {output_path}")
    
    # Z-scores
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5))
    if n_models == 1:
        axes = [axes]
    
    for i, model_name in enumerate(models):
        model_data = scores_df[scores_df["model"] == model_name]
        axes[i].hist(model_data["z_score"], bins=50, alpha=0.7, edgecolor="black", color="orange")
        axes[i].axvline(2, color="yellow", linestyle="--", linewidth=2, label="z=2")
        axes[i].axvline(3, color="red", linestyle="--", linewidth=2, label="z=3")
        axes[i].set_xlabel("Z-Score", fontsize=11)
        axes[i].set_ylabel("Frequency", fontsize=11)
        axes[i].set_title(f"{model_name}", fontsize=12)
        axes[i].legend(fontsize=9)
        axes[i].grid(alpha=0.3)
    
    plt.suptitle("Z-Score Distributions (Calibrated)", fontsize=14, y=1.02)
    plt.tight_layout()
    output_path = os.path.join(output_dir, "z_score_distributions.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved z-score distributions to {output_path}")


def plot_alert_levels(scores_df: pd.DataFrame, output_dir: str):
    """Plot alert level distributions per model."""
    if "alert_level" not in scores_df.columns:
        print("⚠ No alert_level column, skipping alert plot")
        return
    
    models = sorted(scores_df["model"].unique())
    
    # Count alert levels per model
    alert_data = []
    for model in models:
        model_scores = scores_df[scores_df["model"] == model]
        counts = model_scores["alert_level"].value_counts()
        for level in ["normal", "medium", "high"]:
            alert_data.append({
                "model": model,
                "alert_level": level,
                "count": counts.get(level, 0)
            })
    
    alert_df = pd.DataFrame(alert_data)
    
    # Stacked bar chart
    pivot_df = alert_df.pivot(index="model", columns="alert_level", values="count").fillna(0)
    pivot_df = pivot_df[["normal", "medium", "high"]]  # Order columns
    
    colors = {"normal": "green", "medium": "orange", "high": "red"}
    
    ax = pivot_df.plot(kind="bar", stacked=True, figsize=(10, 6),
                       color=[colors[col] for col in pivot_df.columns],
                       edgecolor="black", linewidth=0.5)
    
    plt.xlabel("Model", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.title("Alert Level Distribution by Model", fontsize=14)
    plt.legend(title="Alert Level", fontsize=10)
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "alert_levels_comparison.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✓ Saved alert levels to {output_path}")


def plot_summary_table(scores_df: pd.DataFrame, output_dir: str):
    """Generate summary statistics table for all models."""
    models = sorted(scores_df["model"].unique())
    
    summary_data = []
    for model in models:
        model_scores = scores_df[scores_df["model"] == model]
        summary_data.append({
            "Model": model,
            "Count": len(model_scores),
            "Raw μ": f"{model_scores['raw_score'].mean():.3f}",
            "Raw σ": f"{model_scores['raw_score'].std():.3f}",
            "Z μ": f"{model_scores['z_score'].mean():.3f}",
            "Z σ": f"{model_scores['z_score'].std():.3f}",
            "High %": f"{(model_scores['alert_level'] == 'high').sum() / len(model_scores) * 100:.2f}%"
                      if "alert_level" in model_scores.columns else "N/A",
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Create table plot
    fig, ax = plt.subplots(figsize=(12, len(models) * 0.5 + 1))
    ax.axis("tight")
    ax.axis("off")
    
    table = ax.table(cellText=summary_df.values,
                    colLabels=summary_df.columns,
                    cellLoc="center",
                    loc="center",
                    colWidths=[0.15, 0.1, 0.12, 0.12, 0.12, 0.12, 0.12])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(summary_df.columns)):
        table[(0, i)].set_facecolor("#40466e")
        table[(0, i)].set_text_props(weight="bold", color="white")
    
    # Alternate row colors
    for i in range(1, len(summary_df) + 1):
        for j in range(len(summary_df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor("#f0f0f0")
    
    plt.title("Model Summary Statistics", fontsize=14, pad=20)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "summary_table.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved summary table to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Multi-model anomaly detection evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic comparison (no ground truth)
  python scripts/evaluate_model.py \\
      --scores-file ./data/run_001/scores.parquet \\
      --output-dir ./results/run_001
  
  # With ground truth
  python scripts/evaluate_model.py \\
      --scores-file ./data/run_001/scores.parquet \\
      --ground-truth ./ground_truth/anomalies.json \\
      --output-dir ./results/run_001
"""
    )
    
    parser.add_argument(
        "--scores-file",
        required=True,
        help="Path to scores parquet file (from stream_collector.py)"
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
    print("Multi-Model Anomaly Detection Evaluation")
    print("=" * 60)
    print()
    
    # Load data
    scores_df = load_scores(args.scores_file)
    ground_truth_df = (
        load_ground_truth(args.ground_truth) if args.ground_truth else pd.DataFrame()
    )
    
    models = scores_df["model"].unique()
    print(f"\nDetected models: {', '.join(models)}")
    print()
    
    # Compute metrics if ground truth available
    if not ground_truth_df.empty:
        print("=" * 60)
        print("Computing Metrics")
        print("=" * 60)
        
        threshold_strategies = {
            "z_score_2": lambda df: df["z_score"].abs() >= 2.0,
            "z_score_3": lambda df: df["z_score"].abs() >= 3.0,
            "alert_medium_or_high": lambda df: df["alert_level"].isin(["medium", "high"])
                                              if "alert_level" in df.columns else pd.Series([False] * len(df)),
            "alert_high_only": lambda df: df["alert_level"] == "high"
                                         if "alert_level" in df.columns else pd.Series([False] * len(df)),
        }
        
        metrics = compute_per_model_metrics(scores_df, ground_truth_df, threshold_strategies)
        
        if metrics:
            metrics_file = os.path.join(args.output_dir, "metrics_per_model.json")
            with open(metrics_file, "w") as f:
                json.dump(metrics, f, indent=2)
            print(f"\n✓ Saved metrics to {metrics_file}")
    
    # Generate plots
    print("\n" + "=" * 60)
    print("Generating Plots")
    print("=" * 60)
    print()
    
    plot_summary_table(scores_df, args.output_dir)
    plot_score_distributions(scores_df, args.output_dir)
    plot_alert_levels(scores_df, args.output_dir)
    
    if not ground_truth_df.empty:
        plot_multi_model_roc(scores_df, ground_truth_df, args.output_dir)
    
    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
