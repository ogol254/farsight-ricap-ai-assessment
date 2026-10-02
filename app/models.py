"""Train/reload small real estimators using explicitly synthetic demo data."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from assessment_code.train_uc1 import train, NUMERIC, CATEGORICAL
from app.demo_data import COMPLAINTS, ROUTES


def synthetic_quarters():
    rng = np.random.default_rng(174)
    frames = []
    for quarter in pd.date_range("2021-01-01", "2025-10-01", freq="QS"):
        n = 650
        filings = rng.integers(4, 25, n)
        late = np.array([rng.integers(0, k + 1) for k in filings])
        ratio = rng.uniform(.05, 1.3, n)
        days = rng.integers(0, 600, n)
        growth = rng.normal(0, .25, n)
        # Synthetic labels are a noisy simulation, never evidence of real evasion.
        odds = -6.0 + 3 * late / filings + 2 * (1 - np.minimum(ratio, 1)) + days / 600 - growth
        label = rng.binomial(1, 1 / (1 + np.exp(-odds)))
        frames.append(pd.DataFrame({"taxpayer_id": [f"DEMO-{i:04}" for i in range(n)], "quarter_start": quarter,
            "sector": rng.choice(["RETAIL", "SERVICES", "MANUFACTURING"], n), "region": rng.choice(["DEMO-NORTH", "DEMO-SOUTH"], n),
            "business_size": rng.choice(["MICRO", "SMALL", "MEDIUM", "LARGE"], n), "filings_24m": filings,
            "late_filings_24m": late, "payment_ratio_12m": ratio, "days_since_last_payment": days,
            "turnover_growth_yoy": growth, "label": label, "additional_tax": label * rng.uniform(100, 10000, n)}))
    return pd.concat(frames, ignore_index=True)


class Models:
    def __init__(self, directory="artifacts/connected-v1"):
        folder = Path(directory)
        folder.mkdir(parents=True, exist_ok=True)
        if not (folder / "model.joblib").exists():
            metrics = train(synthetic_quarters(), folder)
            metrics.update(model_version="uc1-synthetic-2.0", training_data="Synthetic seeded simulation; not field-validated")
            (folder / "metadata.json").write_text(json.dumps(metrics, indent=2))
        self.risk = joblib.load(folder / "model.joblib")
        self.metadata = json.loads((folder / "metadata.json").read_text())
        texts, labels = [], []
        for label, samples in COMPLAINTS.items():
            texts.extend(samples)
            labels.extend([label] * len(samples))
        self.complaints = Pipeline([("text", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True)),
                                    ("classifier", LogisticRegression(C=8, class_weight="balanced", max_iter=1000, random_state=42))])
        self.complaints.fit(texts, labels)

    def score(self, records):
        frame = pd.DataFrame(records)
        scores = self.risk.predict_proba(frame[NUMERIC + CATEGORICAL])[:, 1]
        result = []
        for row, score in zip(records, scores):
            warnings = []
            if row["payment_ratio_12m"] is not None and row["payment_ratio_12m"] > 1.3:
                warnings.append("payment_ratio_outside_demo_training_range")
            if row["days_since_last_payment"] is not None and row["days_since_last_payment"] > 600:
                warnings.append("payment_age_outside_demo_training_range")
            if abs(row["turnover_growth_yoy"] or 0) > 1:
                warnings.append("turnover_change_outside_demo_training_range")
            result.append({"taxpayer_id": row["taxpayer_id"], "risk_score": round(float(score), 4),
                           "risk_band": "HIGH" if score >= .65 else "MEDIUM" if score >= .35 else "LOW",
                           "model_version": self.metadata["model_version"], "warnings": warnings,
                           "requires_review": True})
        return result

    def classify(self, text):
        scores = self.complaints.predict_proba([text])[0]
        order = np.argsort(scores)[::-1]
        label = str(self.complaints.classes_[order[0]])
        top = float(scores[order[0]])
        uncertain = top < .55 or top - float(scores[order[1]]) < .15
        return {"category": label, "confidence": round(top, 4), "route": "human-triage" if uncertain else ROUTES[label],
                "requires_review": uncertain, "method": "TF-IDF + logistic regression", "model_version": "uc4-authored-demo-2.0",
                "alternatives": [{"category": str(self.complaints.classes_[i]), "score": round(float(scores[i]), 4)} for i in order[:3]],
                "training_data": "80 authored bilingual demo examples; confidence is not field-calibrated"}
