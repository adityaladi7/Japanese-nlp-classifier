"""
features.py
-----------
Feature engineering pipeline:
- TF-IDF vectorization (unigram + bigram)
- Character-level n-gram features for adversarial robustness
- PySpark-based distributed vectorization (with sklearn fallback for local dev)
"""

import pandas as pd
import numpy as np
import pickle
import os
from typing import Tuple, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from scipy.sparse import hstack, csr_matrix

# Try PySpark import — fall back to sklearn for local development
try:
    from pyspark.ml.feature import HashingTF, IDF, Tokenizer, NGram
    from pyspark.ml import Pipeline as SparkPipeline
    from pyspark.sql import DataFrame as SparkDataFrame
    SPARK_AVAILABLE = True
except ImportError:
    SPARK_AVAILABLE = False
    print("[INFO] PySpark not available. Using sklearn TF-IDF (local mode).")


# ── Sklearn-based feature pipeline (local / dev mode) ──

def build_sklearn_vectorizer(
    word_ngram_range: Tuple[int, int] = (1, 2),
    char_ngram_range: Tuple[int, int] = (2, 4),
    max_features: int = 50000,
) -> FeatureUnion:
    """
    Build a FeatureUnion of:
    - Word-level TF-IDF (unigram + bigram)
    - Character-level TF-IDF (2–4 grams) for adversarial robustness
    """
    word_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=word_ngram_range,
        max_features=max_features,
        sublinear_tf=True,          # log(1+tf) dampens high-frequency terms
        min_df=2,                   # ignore terms appearing in < 2 docs
        max_df=0.95,                # ignore terms in > 95% of docs
        strip_accents="unicode",
        token_pattern=r"(?u)\b\w+\b",
    )

    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",         # char n-grams within word boundaries
        ngram_range=char_ngram_range,
        max_features=max_features // 2,
        sublinear_tf=True,
        min_df=2,
        max_df=0.95,
    )

    return FeatureUnion([
        ("word_tfidf", word_vectorizer),
        ("char_tfidf", char_vectorizer),
    ])


def fit_vectorizer(
    texts: pd.Series,
    save_path: Optional[str] = None,
    **kwargs,
) -> FeatureUnion:
    """Fit the feature pipeline on training texts. Optionally save to disk."""
    print(f"[features] Fitting vectorizer on {len(texts)} samples...")
    vectorizer = build_sklearn_vectorizer(**kwargs)
    vectorizer.fit(texts)
    print(f"[features] Vocabulary size: word={len(vectorizer.transformer_list[0][1].vocabulary_)}, "
          f"char={len(vectorizer.transformer_list[1][1].vocabulary_)}")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(vectorizer, f)
        print(f"[features] Vectorizer saved to {save_path}")

    return vectorizer


def transform_texts(vectorizer: FeatureUnion, texts: pd.Series) -> csr_matrix:
    """Transform texts using fitted vectorizer. Returns sparse matrix."""
    return vectorizer.transform(texts)


def load_vectorizer(path: str) -> FeatureUnion:
    """Load a previously saved vectorizer from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)


# ── PySpark-based feature pipeline (distributed / production mode) ──

def build_spark_pipeline(input_col: str = "processed_text"):
    """
    Build a Spark ML pipeline for distributed TF-IDF feature extraction.
    Requires PySpark + Hadoop setup.
    """
    if not SPARK_AVAILABLE:
        raise RuntimeError("PySpark not available. Run in cluster mode or install pyspark.")

    tokenizer = Tokenizer(inputCol=input_col, outputCol="tokens")
    bigram = NGram(n=2, inputCol="tokens", outputCol="bigrams")
    hashing_tf = HashingTF(inputCol="tokens", outputCol="raw_features", numFeatures=100000)
    idf = IDF(inputCol="raw_features", outputCol="features", minDocFreq=2)

    return SparkPipeline(stages=[tokenizer, bigram, hashing_tf, idf])


def fit_spark_pipeline(spark_df, input_col: str = "processed_text", save_path: Optional[str] = None):
    """Fit the Spark TF-IDF pipeline on a Spark DataFrame."""
    if not SPARK_AVAILABLE:
        raise RuntimeError("PySpark not available.")

    pipeline = build_spark_pipeline(input_col)
    print("[features] Fitting Spark TF-IDF pipeline...")
    model = pipeline.fit(spark_df)

    if save_path:
        model.save(save_path)
        print(f"[features] Spark pipeline model saved to {save_path}")

    return model


if __name__ == "__main__":
    # Smoke test with synthetic data
    sample_texts = pd.Series([
        "このアプリ は 最悪 です 絶対 使わない",
        "普通 レビュー 問題 ない",
        "バカ 野郎 うざい 死ね",
        "良い 商品 です また 買います",
        "クズ サービス 最低 返金 して",
        "普通 使い やすい アプリ",
    ])

    print("=== Feature Engineering Smoke Test ===\n")
    vec = fit_vectorizer(sample_texts)
    X = transform_texts(vec, sample_texts)
    print(f"Feature matrix shape: {X.shape}")
    print(f"Non-zero elements: {X.nnz}")
    print("\nFeature engineering pipeline OK ✓")
