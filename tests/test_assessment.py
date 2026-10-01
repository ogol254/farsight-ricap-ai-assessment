"""Regression tests for assessment contracts; all fixtures are synthetic."""
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import duckdb
from fastapi.testclient import TestClient
import numpy as np
import pandas as pd

from app.main import app as demo, RiskRecord, risk_score
from assessment_code.rag_answer import answer_question
from assessment_code.serve_uc1 import create_app, verify_token, User
from assessment_code.train_uc1 import train, top_k_indices


class TrainingAndServingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        rng = np.random.default_rng(42)
        n = 600
        frames = []
        for quarter in ("2025-04-01", "2025-07-01", "2025-10-01"):
            late = rng.integers(0, 13, n)
            frames.append(pd.DataFrame({
                "taxpayer_id": [f"synthetic-{i:04d}" for i in range(n)], "quarter_start": quarter,
                "sector": [f"sector-{i % 25}" for i in range(n)], "region": [f"region-{i % 20}" for i in range(n)],
                "business_size": "SMALL", "filings_24m": 12, "late_filings_24m": late,
                "payment_ratio_12m": rng.random(n), "days_since_last_payment": rng.integers(0, 800, n),
                "turnover_growth_yoy": rng.normal(0, .2, n), "additional_tax": rng.random(n),
                "label": ((late > 10) & (rng.random(n) < .35)).astype(int),
            }))
        cls.df = pd.concat(frames, ignore_index=True)
        cls.metrics = train(cls.df, cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_capacity_and_artifact(self):
        self.assertEqual(self.metrics["test_selected_count"], 500)
        self.assertNotIn("additional_tax", self.metrics["feature_list"])
        self.assertTrue(Path(self.tmp.name, "model.joblib").is_file())
        tied = top_k_indices(np.ones(700), [str(i).zfill(4) for i in range(700)])
        self.assertEqual(len(tied), 500)
        np.testing.assert_array_equal(tied, np.arange(500))

    def test_duplicate_quarters_rejected(self):
        with self.assertRaisesRegex(ValueError, "one row"):
            train(pd.concat([self.df, self.df.iloc[:1]]), self.tmp.name)

    def test_serving_contract(self):
        service = create_app(self.tmp.name)
        record = {"taxpayer_id": "synthetic", "sector": "unseen", "filings_24m": 12,
                  "late_filings_24m": 2, "days_since_last_payment": 0}
        with patch.dict(os.environ, {"RISK_API_TOKEN": "test-only-token"}), TestClient(service) as client:
            self.assertTrue(client.get("/ready").json()["model_loaded"])
            headers = {"Authorization": "Bearer test-only-token"}
            self.assertEqual(client.post("/v1/risk-scores", json={"records": [record]}).status_code, 401)
            result = client.post("/v1/risk-scores", headers=headers, json={"records": [record]})
            self.assertEqual(result.status_code, 200)
            self.assertTrue(0 <= result.json()["results"][0]["risk_score"] <= 1)
            self.assertEqual(client.post("/v1/risk-scores", headers=headers, json={"records": [record] * 1001}).status_code, 413)
            bad = {**record, "late_filings_24m": 13}
            self.assertEqual(client.post("/v1/risk-scores", headers=headers, json={"records": [bad]}).status_code, 422)
            service.dependency_overrides[verify_token] = lambda: User(roles={"PUBLIC"})
            self.assertEqual(client.post("/v1/risk-scores", json={"records": [record]}).status_code, 403)

    def test_missing_model_is_not_ready(self):
        with tempfile.TemporaryDirectory() as empty, TestClient(create_app(empty)) as client:
            self.assertEqual(client.get("/health").status_code, 200)
            self.assertEqual(client.get("/ready").status_code, 503)


class DemoTests(unittest.TestCase):
    def test_zero_days_is_not_missing(self):
        base = dict(taxpayer_id="synthetic", filings_24m=1, late_filings_24m=0)
        self.assertLess(risk_score(RiskRecord(**base, days_since_last_payment=0)),
                        risk_score(RiskRecord(**base, days_since_last_payment=None)))

    def test_faq_selection_and_abstention(self):
        with TestClient(demo) as client:
            self.assertFalse(client.get("/ready").json()["model_loaded"])
            answer = client.post("/v1/assistant", json={"question": "What payment channels are supported?"}).json()
            self.assertEqual(answer["citations"][0]["doc_id"], "FAQ-002")
            self.assertFalse(client.post("/v1/assistant", json={"question": "Explain black holes"}).json()["grounded"])
            self.assertIsNone(client.post("/v1/complaints", json={"text": "missing payment"}).json()["confidence"])


class RagTests(unittest.TestCase):
    def chunk(self, **changes):
        return SimpleNamespace(**({"text": "Approved source", "doc_id": "FAQ-1", "section": "1",
                                  "score": .9, "lang": "en", "superseded": False,
                                  "effective_date": "2020-01-01", "audience": "PUBLIC"} | changes))

    def run_rag(self, chunks, response="Use approved channels [FAQ-1 § 1]", role="PUBLIC"):
        logs, prompts = [], []
        def generate(**kwargs):
            prompts.append(kwargs)
            return response
        out = answer_question("TIN 123456789 phone +254 712 345 678", role,
                              detect_language=lambda q: "en", embed=lambda q: [[1.0]],
                              vector_store=SimpleNamespace(search=lambda *a, **k: chunks),
                              llm=SimpleNamespace(generate=generate), logger=SimpleNamespace(info=logs.append))
        return out, logs, prompts

    def test_access_dates_threshold_and_fallback(self):
        for chunk in (self.chunk(audience="INTERNAL"), self.chunk(superseded=True),
                      self.chunk(effective_date="2999-01-01"), self.chunk(score=.1)):
            out, _, prompts = self.run_rag([chunk])
            self.assertFalse(out["grounded"])
            self.assertFalse(prompts)

    def test_citations_and_log_redaction(self):
        out, logs, prompts = self.run_rag([self.chunk()])
        self.assertTrue(out["grounded"])
        self.assertNotIn("123456789", " ".join(logs))
        self.assertNotIn("712 345 678", " ".join(logs))
        self.assertIn("English", prompts[0]["system"])
        self.assertFalse(self.run_rag([self.chunk()], response="Unsupported [FAKE § 9]")[0]["grounded"])


class FeatureQueryTests(unittest.TestCase):
    def test_old_payment_and_no_activity(self):
        # Execute the PostgreSQL-compatible query in DuckDB for boundary fixtures.
        # This supplements, rather than claims to replace, deployment tests on PostgreSQL.
        db = duckdb.connect()
        self.addCleanup(db.close)
        db.execute("CREATE TABLE taxpayers(taxpayer_id INT, tin VARCHAR, name VARCHAR, owner_gender VARCHAR, sector VARCHAR, region VARCHAR, business_size VARCHAR, status VARCHAR)")
        db.execute("INSERT INTO taxpayers VALUES (1,'000000001','Synthetic',NULL,'A','R','SMALL','ACTIVE'),(2,'000000002','Empty',NULL,'A','R','SMALL','ACTIVE')")
        db.execute("CREATE TABLE filings(taxpayer_id INT, due_date DATE, filed_date DATE, tax_due DECIMAL(12,2))")
        db.execute("INSERT INTO filings VALUES (1,'2025-06-01','2025-06-02',100),(1,'2025-07-01',NULL,100)")
        db.execute("CREATE TABLE payments(taxpayer_id INT, amount DECIMAL(12,2), status VARCHAR, paid_at TIMESTAMP)")
        db.execute("INSERT INTO payments VALUES (1,50,'SUCCESS','2023-01-01'),(1,100,'SUCCESS','2025-08-01'),(1,999,'SUCCESS','2026-01-01'),(1,999,'FAILED','2025-09-01')")
        query = Path("assessment_code/uc1_features.sql").read_text()
        rows = {r[0]: r for r in db.execute(query).fetchall()}
        self.assertEqual(rows[1][-4:-1], (2, 2, .5))
        self.assertEqual(rows[1][-1], 153)
        self.assertEqual(rows[2][-4:], (0, 0, None, None))
        db.execute("DELETE FROM payments WHERE paid_at >= '2025-01-01'")
        rows = {r[0]: r for r in db.execute(query).fetchall()}
        self.assertEqual(rows[1][-1], 1096)


if __name__ == "__main__":
    unittest.main()
