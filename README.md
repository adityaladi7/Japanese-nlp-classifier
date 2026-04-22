# 🛡️ Japanese-Language NLP Abuse & Sentiment Classifier
### Production-Grade Multi-Class Text Classification on Distributed Infrastructure

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://python.org)
[![PySpark](https://img.shields.io/badge/PySpark-Distributed-E25A1C?logo=apachespark&logoColor=white)]()
[![Accuracy](https://img.shields.io/badge/Accuracy-89%25-brightgreen)]()
[![Records](https://img.shields.io/badge/Training_Data-22K+_records-blue)]()
[![Status](https://img.shields.io/badge/Status-Production_Pipeline-brightgreen)]()

> **Built to answer:** *Can we automatically detect abusive, harmful, and low-quality content in Japanese — at scale, with production-grade robustness?*

---

## What This Does

A full production ML pipeline that classifies Japanese-language text into abuse, negative sentiment, and content quality risk categories — trained on 22,000+ real-world records, achieving **89% accuracy** with rigorous evaluation (F1, precision, recall per class).

This is not a notebook. It's a distributed, end-to-end pipeline built on PySpark + Hadoop — designed to scale to millions of records.

---

## Pipeline Architecture

```
Raw Japanese Text (22,000+ labeled records)
        │
        ▼
[ Preprocessing ]
  - Unicode normalization (full-width → half-width)
  - MeCab tokenization (Japanese morphological analysis)
  - Stopword removal, lemmatization via spaCy/NLTK
        │
        ▼
[ Adversarial Text Handling ]  ← Production differentiator
  - Character substitution detection (ａｂｃ → abc)
  - Obfuscation patterns (leetspeak, symbol insertion)
  - Code-switching (Japanese/English mixed abuse)
        │
        ▼
[ Feature Engineering — PySpark ]
  - TF-IDF vectorization (distributed)
  - N-gram features (unigram + bigram)
  - Character-level features for adversarial robustness
        │
        ▼
[ Model Training — Distributed ]
  - Random Forest classifier (primary)
  - Cross-validation, hyperparameter tuning
  - Per-class evaluation: F1, precision, recall
        │
        ▼
[ Output: Harm Taxonomy + Severity Tiers ]
  - Class label + confidence score
  - Severity tier (Low / Medium / High / Critical)
  - Escalation flag for human review queue
```

---

## Model Performance

| Class | Precision | Recall | F1 Score |
|---|---|---|---|
| Abusive Content | 0.91 | 0.88 | 0.89 |
| Negative Sentiment | 0.87 | 0.90 | 0.88 |
| Quality Risk | 0.89 | 0.87 | 0.88 |
| **Overall** | **—** | **—** | **0.89** |

Evaluated on held-out test set (20% split, stratified).

---

## Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.10+ |
| Distributed compute | Apache PySpark, Hadoop (HDFS) |
| NLP / Tokenization | spaCy, NLTK, MeCab (Japanese) |
| Feature engineering | TF-IDF (Spark MLlib), N-grams |
| Model | Random Forest (Spark MLlib) |
| Evaluation | Scikit-learn metrics, confusion matrix |
| Output | JSON predictions + CSV export |

---

## Quickstart

```bash
# 1. Clone
git clone https://github.com/adityaladi7/japanese-nlp-classifier
cd japanese-nlp-classifier

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Install MeCab (Japanese tokenizer)
# macOS: brew install mecab mecab-ipadic
# Ubuntu: sudo apt-get install mecab libmecab-dev mecab-ipadic-utf8

# 4. Run the full pipeline (local Spark mode)
python pipeline/run_pipeline.py --input data/train.csv --mode local

# 5. Run with Hadoop (cluster mode)
spark-submit --master yarn pipeline/run_pipeline.py --input hdfs://data/train.csv
```

---

## Project Structure

```
japanese-nlp-classifier/
├── data/
│   ├── train.csv                  # Labeled training data
│   └── test.csv                   # Held-out test set
├── pipeline/
│   ├── preprocess.py              # Tokenization, normalization
│   ├── adversarial.py             # Evasion technique handling
│   ├── features.py                # TF-IDF + N-gram (Spark)
│   ├── train.py                   # Model training + CV
│   ├── evaluate.py                # F1, precision, recall, confusion matrix
│   └── run_pipeline.py            # End-to-end runner
├── taxonomy/
│   ├── harm_categories.json       # Abuse taxonomy definition
│   └── severity_tiers.json        # Escalation thresholds
├── notebooks/
│   └── exploration.ipynb          # EDA, error analysis
├── requirements.txt
└── README.md
```

---

## Harm Taxonomy

Structured output designed for production integration:

```json
{
  "text": "...",
  "predicted_class": "abusive_content",
  "confidence": 0.94,
  "severity_tier": "HIGH",
  "escalate_to_human": true,
  "taxonomy": {
    "category": "Direct Harassment",
    "subcategory": "Personal Attack",
    "flags": ["adversarial_text_detected", "code_switching"]
  }
}
```

---

## Adversarial Robustness — Key Differentiator

Real-world abuse content is designed to evade classifiers. This pipeline handles:

| Evasion Technique | Example | Handling |
|---|---|---|
| Full-width substitution | `ａｂｕｓｅ` → `abuse` | Unicode normalization |
| Symbol insertion | `a.b.u.s.e` | Regex cleaning layer |
| Code-switching | Japanese + English mixed | Multilingual tokenization |
| Homoglyph substitution | `аbuse` (Cyrillic а) | Character normalization |

---

## What I'd Build Next

- [ ] Fine-tune a Japanese BERT model (Tohoku BERT) for higher accuracy on edge cases
- [ ] Add LLM-based re-ranking for borderline cases (LangChain + OpenAI)
- [ ] Build a real-time inference API with FastAPI + Docker
- [ ] Active learning loop: model flags uncertain cases → human labels → retrains

---

*Built by [Aditya Gaur](https://linkedin.com/in/adityagaur) · [GitHub](https://github.com/adityaladi7)*
