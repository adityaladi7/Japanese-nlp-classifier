"""
evaluate.py
-----------
Model evaluation utilities:
- Classification report (precision, recall, F1 per class)
- Confusion matrix with visualization
- Error analysis: surface misclassified examples
- Severity-weighted evaluation
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Dict, Tuple

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

LABEL_NAMES = ["abusive_content", "negative_sentiment", "quality_risk"]
LABEL_MAP_INV = {0: "abusive_content", 1: "negative_sentiment", 2: "quality_risk"}


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: List[str] = LABEL_NAMES,
) -> Dict:
    """Print and return full per-class classification report."""
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    report = classification_report(y_true, y_pred, target_names=label_names, output_dict=True)
    print(classification_report(y_true, y_pred, target_names=label_names))
    return report


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: List[str] = LABEL_NAMES,
    save_path: Optional[str] = None,
):
    """Plot normalized confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Raw counts
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=label_names, yticklabels=label_names,
        ax=axes[0]
    )
    axes[0].set_title("Confusion Matrix (counts)")
    axes[0].set_ylabel("True Label")
    axes[0].set_xlabel("Predicted Label")

    # Normalized
    sns.heatmap(
        cm_normalized, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=label_names, yticklabels=label_names,
        ax=axes[1]
    )
    axes[1].set_title("Confusion Matrix (normalized)")
    axes[1].set_ylabel("True Label")
    axes[1].set_xlabel("Predicted Label")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[evaluate] Confusion matrix saved to {save_path}")

    plt.show()
    return cm


def error_analysis(
    texts: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_samples: int = 10,
) -> pd.DataFrame:
    """
    Surface misclassified examples for qualitative error analysis.
    Returns a DataFrame of misclassified texts with true/predicted labels.
    """
    mask = y_true != y_pred
    error_df = pd.DataFrame({
        "text": texts[mask].values,
        "true_label": [LABEL_MAP_INV[i] for i in y_true[mask]],
        "predicted_label": [LABEL_MAP_INV[i] for i in y_pred[mask]],
    })

    print(f"\n[evaluate] Total errors: {mask.sum()} / {len(y_true)} ({mask.mean() * 100:.1f}%)")
    print(f"\nSample misclassified examples (n={min(n_samples, len(error_df))}):\n")
    sample = error_df.sample(min(n_samples, len(error_df)), random_state=42)
    for _, row in sample.iterrows():
        print(f"  Text:      {row['text'][:80]}...")
        print(f"  True:      {row['true_label']}")
        print(f"  Predicted: {row['predicted_label']}")
        print()

    return error_df


def severity_weighted_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    severity_weights: Dict[int, float] = None,
) -> float:
    """
    Compute accuracy weighted by class severity.
    Abusive content errors are penalized more heavily than quality_risk errors.
    """
    if severity_weights is None:
        severity_weights = {
            0: 3.0,  # abusive_content — highest cost to miss
            1: 2.0,  # negative_sentiment
            2: 1.0,  # quality_risk
        }

    correct = (y_true == y_pred).astype(float)
    weights = np.array([severity_weights[label] for label in y_true])
    return np.average(correct, weights=weights)


def full_evaluation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    texts: Optional[pd.Series] = None,
    save_dir: Optional[str] = None,
) -> Dict:
    """
    Run complete evaluation suite.
    """
    print("\n" + "=" * 60)
    print("FULL MODEL EVALUATION")
    print("=" * 60)

    report = print_classification_report(y_true, y_pred)

    overall_f1_macro = f1_score(y_true, y_pred, average="macro")
    overall_f1_weighted = f1_score(y_true, y_pred, average="weighted")
    sev_acc = severity_weighted_accuracy(y_true, y_pred)

    print(f"Overall F1 (macro):          {overall_f1_macro:.4f}")
    print(f"Overall F1 (weighted):       {overall_f1_weighted:.4f}")
    print(f"Severity-weighted accuracy:  {sev_acc:.4f}")

    cm_path = os.path.join(save_dir, "confusion_matrix.png") if save_dir else None
    plot_confusion_matrix(y_true, y_pred, save_path=cm_path)

    if texts is not None:
        error_df = error_analysis(texts, y_true, y_pred)
        if save_dir:
            error_df.to_csv(os.path.join(save_dir, "error_analysis.csv"), index=False)

    return {
        "report": report,
        "f1_macro": overall_f1_macro,
        "f1_weighted": overall_f1_weighted,
        "severity_weighted_accuracy": sev_acc,
    }


if __name__ == "__main__":
    # Smoke test with synthetic predictions
    np.random.seed(42)
    n = 300
    y_true = np.random.choice([0, 1, 2], size=n, p=[0.4, 0.35, 0.25])
    # Simulate ~89% accuracy
    noise_mask = np.random.rand(n) < 0.11
    y_pred = y_true.copy()
    y_pred[noise_mask] = np.random.choice([0, 1, 2], size=noise_mask.sum())

    texts = pd.Series([f"サンプルテキスト {i}" for i in range(n)])

    print("=== Evaluation Smoke Test ===")
    results = full_evaluation(y_true, y_pred, texts=texts)
    print("\nEvaluation pipeline OK ✓")
