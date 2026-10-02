"""Integration tests with actual trained models, OCR and durable local storage."""
from io import BytesIO
from pathlib import Path
import tempfile
import unittest

from fastapi.testclient import TestClient
from PIL import Image, ImageDraw, ImageFont

from app.connected import create_app
from app.database import Repository, Document


def reading_image(text="001234"):
    image = Image.new("RGB", (850, 250), "white")
    draw = ImageDraw.Draw(image)
    font = next((p for p in ["/System/Library/Fonts/Supplemental/Arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"] if Path(p).exists()), None)
    draw.text((55, 50), text, fill="black", font=ImageFont.truetype(font, 110) if font else ImageFont.load_default(size=110))
    out = BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


class ConnectedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.url = f"sqlite:///{cls.tmp.name}/test.db"
        cls.app = create_app(cls.url, photo_root=f"{cls.tmp.name}/photos", neural=False)
        cls.client = TestClient(cls.app).__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)
        cls.tmp.cleanup()

    def setUp(self):
        session = self.client.post("/v1/sessions")
        self.assertEqual(session.status_code, 200)
        self.headers = {"Authorization": "Bearer " + session.json()["access_token"]}

    def test_01_readiness_and_auth(self):
        ready = self.client.get("/ready").json()
        self.assertTrue(ready["model_loaded"])
        self.assertTrue(ready["ocr_loaded"])
        self.assertEqual(ready["database"], "sqlite_local")
        self.assertEqual(self.client.get("/v1/history").status_code, 401)
        self.assertEqual(self.client.get("/v1/history", headers={"Authorization": "Bearer forged"}).status_code, 401)

    def test_02_real_risk_model_and_persistence(self):
        records = [{"taxpayer_id": "test", "filings_24m": 12, "late_filings_24m": 8, "payment_ratio_12m": .3, "days_since_last_payment": 200}]
        response = self.client.post("/v1/risk-scores", json={"records": records}, headers=self.headers)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["model_version"], "uc1-synthetic-2.0")
        hist = self.client.get("/v1/history", headers=self.headers).json()["records"]
        self.assertEqual(hist[0]["id"], response.json()["record_id"])
        # A new DB connection sees the persisted record; not an in-memory dictionary.
        other = Repository(self.url)
        with other.session() as db:
            from app.database import Record
            self.assertIsNotNone(db.get(Record, hist[0]["id"]))
        other.close()

    def test_03_database_retrieval_and_abstention(self):
        for question, lang in [("What payment channels are supported?", "en"), ("Sidee lacagta loo bixin karaa?", "so")]:
            result = self.client.post("/v1/assistant", json={"question": question}, headers=self.headers)
            self.assertEqual(result.status_code, 200)
            self.assertTrue(result.json()["grounded"], result.text)
            self.assertEqual(result.json()["language"], lang)
            self.assertTrue(result.json()["citations"])
        result = self.client.post("/v1/assistant", json={"question": "Who won the football match?"}, headers=self.headers)
        self.assertFalse(result.json()["grounded"])

    def test_04_actual_ocr_photo_and_review(self):
        response = self.client.post("/v1/meter-readings", files={"image": ("meter.png", reading_image(), "image/png")}, data={"previous_reading": "1200"}, headers=self.headers)
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result["reading_value"], 1234, result)
        self.assertGreater(result["confidence"], .85)
        photo = self.client.get(f"/v1/meter-readings/{result['record_id']}/photo", headers=self.headers)
        self.assertEqual(photo.status_code, 200)
        self.assertEqual(photo.headers["content-type"], "image/jpeg")
        review = self.client.post(f"/v1/meter-readings/{result['record_id']}/review", json={"reading_value": 1234, "reason": "Checked the display"}, headers=self.headers)
        self.assertEqual(review.status_code, 200)
        self.assertTrue(review.json()["original_ocr_preserved"])

    def test_04b_meter_fixture_reads_red_counter_and_id(self):
        fixture = Path(__file__).parent / "fixtures" / "watermeter.png"
        response = self.client.post("/v1/meter-readings", files={"image": ("watermeter.png", fixture.read_bytes(), "image/png")}, headers=self.headers)
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result["meter_id"], "31120595")
        self.assertEqual(result["reading_value"], 220)
        self.assertEqual(result["reading_text"], "220")
        self.assertNotIn("candidates", result)

    def test_05_reject_non_image_and_abstain_blank(self):
        bad = self.client.post("/v1/meter-readings", files={"image": ("bad.jpg", b"not an image", "image/jpeg")}, headers=self.headers)
        self.assertEqual(bad.status_code, 422)
        blank = self.client.post("/v1/meter-readings", files={"image": ("blank.png", reading_image(""), "image/png")}, headers=self.headers)
        self.assertEqual(blank.status_code, 200)
        self.assertIsNone(blank.json()["reading_value"])
        self.assertTrue(blank.json()["requires_review"])

    def test_06_complaint_model(self):
        result = self.client.post("/v1/complaints", json={"text": "My payment is missing"}, headers=self.headers)
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json()["category"], "payment")
        self.assertGreater(result.json()["confidence"], .5)

    def test_07_session_isolation(self):
        self.assertEqual(self.client.get("/v1/history", headers=self.headers).json()["records"], [])
        self.assertEqual(self.client.get("/v1/meter-readings/nonexistent/photo", headers=self.headers).status_code, 404)

    def test_08_internal_documents_not_exposed(self):
        db = self.app.state.repository
        db.seed_documents([{"id": "test-private", "title": "Internal", "section": "Passwords", "language": "en", "audience": "INTERNAL", "text": "Confidential review token is not public", "effective_date": "2026-01-01", "superseded": False}])
        self.app.state.assistant.refresh()
        public = self.client.get("/v1/documents").json()["documents"]
        self.assertNotIn("test-private", [d["id"] for d in public])
        answer = self.client.post("/v1/assistant", json={"question": "Confidential review token", "user_role": "OFFICER"}, headers=self.headers).json()
        self.assertNotIn("test-private", [c["doc_id"] for c in answer["citations"]])

    def test_09_database_feature_pipeline_to_model(self):
        data = self.client.get("/v1/taxpayers", headers=self.headers).json()
        self.assertEqual(len(data["records"]), 20)
        first = data["records"][0]
        self.assertEqual(first["filings_24m"], 12)
        self.assertEqual(first["payment_ratio_12m"], 1)
        self.assertEqual(first["late_filings_24m"], 0)
        result = self.client.post("/v1/audit-batches", headers=self.headers)
        self.assertEqual(result.status_code, 200, result.text)
        self.assertEqual(result.json()["selected_count"], 20)
        self.assertEqual(result.json()["source"], "database_feature_pipeline")


if __name__ == "__main__":
    unittest.main()
