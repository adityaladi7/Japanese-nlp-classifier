"""
run_pipeline.py
---------------
End-to-end pipeline runner.

Usage:
    # Train on data, evaluate, save model
    python pipeline/run_pipeline.py --input data/sample_data.csv --mode train

    # Run inference on new data
    python pipeline/run_pipeline.py --input data/new_data.csv --mode predict --model-path models/classifier.pkl

    # Full pipeline: train + evaluate + save
    python pipeline/run_pipeline.py --input data/sample_data.csv --mode full
"""

import argparse
import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from preprocess import preprocess_dataframe
from adversarial import full_adversarial_normalization, adversarial_risk_score
from features import fit_vectorizer, transform_texts, load_vectorizer
from train import train_model, encode_labels, decode_labels, load_model, cross_validate_model
from evaluate import full_evaluation

MODEL_DIR = "models"
RESULTS_DIR = "results"


def run_train(df: pd.DataFrame, tune: bool = False):
    """Full training pipeline: preprocess → features → train → evaluate → save."""

    print("\n" + "=" * 60)
    print("PIPELINE: TRAINING MODE")
    print("=" * 60)

    # 1. Preprocess
    df = preprocess_dataframe(df, text_col="text")

    # 2. Adversarial normalization
    print("\n[pipeline] Applying adversarial normalization...")
    normalized_texts = []
    adv_risk_scores = []
    for text in df["processed_text"]:
        norm_text, meta = full_adversarial_normalization(text)
        normalized_texts.append(norm_text)
        adv_risk_scores.append(adversarial_risk_score(meta))
    df["processed_text"] = normalized_texts
    df["adversarial_risk"] = adv_risk_scores
    high_risk = (df["adversarial_risk"] > 0.4).sum()
    print(f"[pipeline] High adversarial risk samples: {high_risk} ({high_risk/len(df)*100:.1f}%)")

    # 3. Train/test split
    X_train_texts, X_test_texts, y_train_raw, y_test_raw = train_test_split(
        df["processed_text"], df["label"],
        test_size=0.2, stratify=df["label"], random_state=42
    )
    print(f"\n[pipeline] Train: {len(X_train_texts)}, Test: {len(X_test_texts)}")

    # 4. Feature engineering
    os.makedirs(MODEL_DIR, exist_ok=True)
    vectorizer = fit_vectorizer(
        X_train_texts,
        save_path=os.path.join(MODEL_DIR, "vectorizer.pkl")
    )
    X_train = transform_texts(vectorizer, X_train_texts)
    X_test = transform_texts(vectorizer, X_test_texts)

    # 5. Encode labels
    y_train = encode_labels(y_train_raw.reset_index(drop=True))
    y_test = encode_labels(y_test_raw.reset_index(drop=True))

    # 6. Cross-validation
    from train import build_random_forest
    cv_model = build_random_forest()
    cross_validate_model(cv_model, X_train, y_train, n_splits=5, model_name="RandomForest")

    # 7. Train final model
    model = train_model(
        X_train, y_train,
        tune=tune,
        save_path=os.path.join(MODEL_DIR, "classifier.pkl")
    )

    # 8. Evaluate on held-out test set
    y_pred = model.predict(X_test)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = full_evaluation(
        y_test, y_pred,
        texts=X_test_texts.reset_index(drop=True),
        save_dir=RESULTS_DIR
    )

    # 9. Save results summary
    summary = {
        "train_samples": len(X_train_texts),
        "test_samples": len(X_test_texts),
        "f1_macro": round(results["f1_macro"], 4),
        "f1_weighted": round(results["f1_weighted"], 4),
        "severity_weighted_accuracy": round(results["severity_weighted_accuracy"], 4),
        "high_adversarial_risk_pct": round(high_risk / len(df) * 100, 2),
    }
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[pipeline] Results summary saved to {RESULTS_DIR}/summary.json")
    print(json.dumps(summary, indent=2))

    print("\n✅ Training pipeline complete.")
    return model, vectorizer, results


def run_predict(df: pd.DataFrame, model_path: str, vectorizer_path: str):
    """Inference pipeline: preprocess → features → predict → output JSON."""

    print("\n" + "=" * 60)
    print("PIPELINE: INFERENCE MODE")
    print("=" * 60)

    model = load_model(model_path)
    vectorizer = load_vectorizer(vectorizer_path)

    df = preprocess_dataframe(df, text_col="text")

    normalized_texts, adv_metas = [], []
    for text in df["processed_text"]:
        norm_text, meta = full_adversarial_normalization(text)
        normalized_texts.append(norm_text)
        adv_metas.append(meta)
    df["processed_text"] = normalized_texts

    X = transform_texts(vectorizer, df["processed_text"])
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    label_names = ["abusive_content", "negative_sentiment", "quality_risk"]
    severity_map = {"abusive_content": "HIGH", "negative_sentiment": "MEDIUM", "quality_risk": "LOW"}

    results = []
    for i, (pred, probs, meta) in enumerate(zip(predictions, probabilities, adv_metas)):
        label = label_names[pred]
        adv_risk = adversarial_risk_score(meta)
        severity = "CRITICAL" if (label == "abusive_content" and adv_risk > 0.4) else severity_map[label]
        results.append({
            "id": i,
            "text_preview": df["text"].iloc[i][:60] + "...",
            "predicted_class": label,
            "confidence": round(float(probs[pred]), 4),
            "severity_tier": severity,
            "escalate_to_human": severity in ("HIGH", "CRITICAL"),
            "adversarial_risk_score": round(adv_risk, 3),
            "adversarial_flags": {
                k: v for k, v in meta.items()
                if k.startswith("had_") and v
            },
        })

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "predictions.json")
    with open(out_path, "w", ensure_ascii=False) as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n[pipeline] {len(results)} predictions saved to {out_path}")

    escalations = sum(1 for r in results if r["escalate_to_human"])
    print(f"[pipeline] Flagged for human review: {escalations} ({escalations/len(results)*100:.1f}%)")

    print("\n✅ Inference pipeline complete.")
    return results


def main():
    parser = argparse.ArgumentParser(description="Japanese NLP Classifier Pipeline")
    parser.add_argument("--input", required=True, help="Path to input CSV (must have 'text' column)")
    parser.add_argument("--mode", choices=["train", "predict", "full"], default="full")
    parser.add_argument("--model-path", default="models/classifier.pkl")
    parser.add_argument("--vectorizer-path", default="models/vectorizer.pkl")
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter tuning")
    args = parser.parse_args()

    print(f"[pipeline] Loading data from {args.input}...")
    df = pd.read_csv(args.input)
    print(f"[pipeline] Loaded {len(df)} rows, columns: {list(df.columns)}")

    if args.mode in ("train", "full"):
        assert "label" in df.columns, "Training data must have a 'label' column"
        run_train(df, tune=args.tune)

    if args.mode == "predict":
        run_predict(df, args.model_path, args.vectorizer_path)


if __name__ == "__main__":
    main()
