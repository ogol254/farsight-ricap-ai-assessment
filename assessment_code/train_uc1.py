"""Point-in-time UC1 training pipeline. Expects a pandas DataFrame named df."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, recall_score, precision_score
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
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL),
    ])
    return Pipeline([("preprocess", prep), ("model", model)])


def evaluate(model, X, y):
    p = model.predict_proba(X)[:, 1]
    return {"pr_auc": round(average_precision_score(y, p), 4), "recall_at_default_threshold": round(recall_score(y, p >= .5), 4)}


def train(df: pd.DataFrame, out_dir="artifacts/uc1"):
    df = df.copy()
    df["quarter_start"] = pd.to_datetime(df["quarter_start"])
    train = df[df.quarter_start < "2025-07-01"]
    valid = df[(df.quarter_start >= "2025-07-01") & (df.quarter_start < "2025-10-01")]
    test = df[df.quarter_start >= "2025-10-01"]
    features = [c for c in df.columns if c not in DROP]
    models = {
        "logistic": make_pipeline(LogisticRegression(class_weight="balanced", max_iter=1000)),
        "gradient_boosting": make_pipeline(HistGradientBoostingClassifier(max_iter=200, learning_rate=.08, max_leaf_nodes=15, class_weight="balanced")),
    }
    metrics = {"training_period": [str(train.quarter_start.min().date()), str(train.quarter_start.max().date())]}
    for name, model in models.items():
        model.fit(train[features], train[TARGET])
        metrics[name] = evaluate(model, valid[features], valid[TARGET])
        if name == "gradient_boosting": champion = model
    # Select a fixed capacity threshold on validation, then report test precision@500.
    valid_scores = champion.predict_proba(valid[features])[:, 1]
    threshold = sorted(valid_scores, reverse=True)[min(499, len(valid_scores) - 1)]
    test_scores = champion.predict_proba(test[features])[:, 1]
    selected = test_scores >= threshold
    metrics["precision_at_500_test"] = round(precision_score(test[TARGET], selected, zero_division=0), 4)
    metrics["feature_list"] = features
    metrics["model_version"] = "uc1-1.0.0"
    metrics["training_date"] = date.today().isoformat()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump(champion, Path(out_dir) / "model.joblib")
    (Path(out_dir) / "metadata.json").write_text(json.dumps(metrics, indent=2))
    return metrics

