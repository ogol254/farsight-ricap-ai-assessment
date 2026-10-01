"""C1. Python; pandas, NumPy, scikit-learn and joblib. Input: quarterly df.

Features must be point-in-time snapshots and labels fully observed before use.
The assessment's prescribed splits are retrospective: Q3 labels can mature
after Q4 begins, so this is not evidence of a deployable Q4 backtest.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import numpy as np
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier

TARGET = "label"
DROP = ["label", "taxpayer_id", "quarter_start", "additional_tax"]  # IDs/time keys and audit-derived outcome leak the label.
NUMERIC = ["filings_24m", "late_filings_24m", "payment_ratio_12m", "days_since_last_payment", "turnover_growth_yoy"]
CATEGORICAL = ["sector", "region", "business_size"]


def make_pipeline(model):
    prep = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler())]), NUMERIC),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CATEGORICAL),
    ])
    return Pipeline([("preprocess", prep), ("model", model)])


def evaluate(model, X, y):
    p = model.predict_proba(X)[:, 1]
    return {"pr_auc": round(average_precision_score(y, p), 4), "recall_at_default_threshold": round(recall_score(y, p >= .5), 4)}


def train(df: pd.DataFrame, out_dir="artifacts/uc1"):
    df = df.copy()
    required = set(NUMERIC + CATEGORICAL + ["taxpayer_id", "quarter_start", TARGET])
    if required - set(df):
        raise ValueError(f"Missing columns: {sorted(required - set(df))}")
    df["quarter_start"] = pd.to_datetime(df["quarter_start"])
    if df[["taxpayer_id", "quarter_start", TARGET]].isna().any().any():
        raise ValueError("Identifiers, quarters and labels must not be missing")
    if df.duplicated(["taxpayer_id", "quarter_start"]).any():
        raise ValueError("Expected one row per taxpayer per quarter")
    if not df[TARGET].isin([0, 1]).all():
        raise ValueError("Labels must be 0 or 1")
    if not df.quarter_start.isin(pd.date_range("2021-01-01", "2025-10-01", freq="QS")).all():
        raise ValueError("Expected quarter starts from 2021-Q1 through 2025-Q4")
    train = df[df.quarter_start < "2025-07-01"]
    valid = df[(df.quarter_start >= "2025-07-01") & (df.quarter_start < "2025-10-01")]
    test = df[(df.quarter_start >= "2025-10-01") & (df.quarter_start < "2026-01-01")]
    if any(part.empty or part[TARGET].nunique() < 2 for part in (train, valid, test)):
        raise ValueError("Each temporal split needs both classes for meaningful evaluation")
    # Explicit allow-list excludes IDs, dates, label and additional_tax (an audit outcome).
    features = NUMERIC + CATEGORICAL
    # Weights use training labels only; validation/test retain the real class balance.
    models = {
        "logistic": make_pipeline(LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
        "gradient_boosting": make_pipeline(HistGradientBoostingClassifier(max_iter=200, learning_rate=.08, max_leaf_nodes=15, class_weight="balanced", random_state=42, early_stopping=False)),
    }
    metrics = {"training_period": [str(train.quarter_start.min().date()), str(train.quarter_start.max().date())]}
    for name, model in models.items():
        model.fit(train[features], train[TARGET])
        metrics[name] = evaluate(model, valid[features], valid[TARGET])
        scores = model.predict_proba(valid[features])[:, 1]
        selected = top_k_indices(scores, valid.taxpayer_id)
        metrics[name]["validation_precision_at_500"] = float(valid[TARGET].iloc[selected].mean())
    # Select on validation only, then freeze the capacity rule; never tune on Q4.
    winner = max(models, key=lambda name: (metrics[name]["validation_precision_at_500"], metrics[name]["pr_auc"]))
    champion = models[winner]
    test_scores = champion.predict_proba(test[features])[:, 1]
    selected = top_k_indices(test_scores, test.taxpayer_id)
    metrics["precision_at_500_test"] = round(float(test[TARGET].iloc[selected].mean()), 4)
    metrics["recall_at_500_test"] = round(float(test[TARGET].iloc[selected].sum() / test[TARGET].sum()), 4)
    metrics["test_selected_count"] = len(selected)
    metrics["selected_model"] = winner
    metrics["selection_policy"] = "Rank each quarter; select min(500, population), breaking ties by taxpayer_id"
    metrics["evaluation_caveat"] = "Retrospective prescribed split; production must enforce label-availability dates"
    metrics["sklearn_version"] = sklearn.__version__
    metrics["feature_list"] = features
    metrics["model_version"] = "uc1-1.0.0"
    metrics["training_date"] = date.today().isoformat()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump(champion, Path(out_dir) / "model.joblib")
    (Path(out_dir) / "metadata.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def top_k_indices(scores, taxpayer_ids, capacity=500):
    """A probability threshold alone cannot enforce capacity when scores tie."""
    return np.lexsort((np.asarray(taxpayer_ids).astype(str), -np.asarray(scores)))[:capacity]
