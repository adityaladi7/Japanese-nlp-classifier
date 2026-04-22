"""
train.py
--------
Model training pipeline:
- Random Forest classifier (primary)
- Stratified K-Fold cross-validation
- Hyperparameter tuning via GridSearchCV
- Model serialization
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV
from sklearn.metrics import classification_report, f1_score
from scipy.sparse import csr_matrix

# Label mapping
LABEL_MAP = {
    "abusive_content": 0,
    "negative_sentiment": 1,
    "quality_risk": 2,
}
LABEL_MAP_INV = {v: k for k, v in LABEL_MAP.items()}


def encode_labels(labels: pd.Series) -> np.ndarray:
    """Convert string labels to integer indices."""
    return labels.map(LABEL_MAP).values


def decode_labels(indices: np.ndarray) -> list:
    """Convert integer indices back to string labels."""
    return [LABEL_MAP_INV[i] for i in indices]


def build_random_forest(n_estimators: int = 200, n_jobs: int = -1, **kwargs) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",   # handles class imbalance
        n_jobs=n_jobs,
        random_state=42,
        **kwargs,
    )


def build_logistic_regression(**kwargs) -> LogisticRegression:
    """Fast baseline model — good for TF-IDF sparse matrices."""
    return LogisticRegression(
        C=1.0,
        max_iter=1000,
        class_weight="balanced",
        solver="saga",
        multi_class="multinomial",
        n_jobs=-1,
        random_state=42,
        **kwargs,
    )


def cross_validate_model(
    model,
    X: csr_matrix,
    y: np.ndarray,
    n_splits: int = 5,
    model_name: str = "Model",
) -> Dict:
    """Run stratified K-fold cross-validation and report per-fold metrics."""
    print(f"\n[train] Cross-validating {model_name} ({n_splits}-fold stratified)...")

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scoring = ["f1_macro", "f1_weighted", "accuracy"]

    results = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)

    print(f"  Accuracy:     {results['test_accuracy'].mean():.4f} ± {results['test_accuracy'].std():.4f}")
    print(f"  F1 (macro):   {results['test_f1_macro'].mean():.4f} ± {results['test_f1_macro'].std():.4f}")
    print(f"  F1 (weighted):{results['test_f1_weighted'].mean():.4f} ± {results['test_f1_weighted'].std():.4f}")

    return results


def tune_hyperparameters(
    X_train: csr_matrix,
    y_train: np.ndarray,
    n_splits: int = 3,
) -> RandomForestClassifier:
    """Grid search over key Random Forest hyperparameters."""
    print("\n[train] Running hyperparameter tuning...")

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 20, 40],
        "min_samples_split": [2, 5],
    }

    base_model = RandomForestClassifier(
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        base_model, param_grid, cv=cv,
        scoring="f1_macro", n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train, y_train)

    print(f"[train] Best params: {grid_search.best_params_}")
    print(f"[train] Best CV F1:  {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_


def train_model(
    X_train: csr_matrix,
    y_train: np.ndarray,
    tune: bool = False,
    save_path: Optional[str] = None,
) -> RandomForestClassifier:
    """
    Train the classifier. Optionally run hyperparameter tuning.
    """
    if tune:
        model = tune_hyperparameters(X_train, y_train)
    else:
        model = build_random_forest()
        print(f"\n[train] Training Random Forest on {X_train.shape[0]} samples, "
              f"{X_train.shape[1]} features...")
        model.fit(X_train, y_train)

    print("[train] Training complete ✓")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(model, f)
        print(f"[train] Model saved to {save_path}")

    return model


def load_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from scipy.sparse import csr_matrix

    print("=== Training Smoke Test ===\n")

    # Synthetic sparse data simulating TF-IDF output
    X_dense, y = make_classification(
        n_samples=500, n_features=200, n_classes=3,
        n_informative=50, random_state=42
    )
    X = csr_matrix(X_dense)

    model = train_model(X, y, tune=False)
    cv_results = cross_validate_model(model, X, y, n_splits=3, model_name="RandomForest")
    print("\nTraining pipeline OK ✓")
