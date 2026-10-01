from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

logger = logging.getLogger("ricap")
logging.basicConfig(level=logging.INFO)

ROOT = Path(__file__).parent
MODEL_VERSION = "uc1-demo-2026.10.01"
MAX_BATCH = 1000

app = FastAPI(title="RICAP AI Services", version="1.0.0", description="Synthetic-data demonstrator for the RICAP case study")


class User(BaseModel):
    role: Literal["RISK_ANALYST", "SYSTEM"]


def verify_token(authorization: Annotated[str | None, Header()] = None) -> User:
    if authorization != "Bearer demo-review-token":
        raise HTTPException(status_code=401, detail="Missing or invalid bearer token")
    return User(role="RISK_ANALYST")


class RiskRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    taxpayer_id: str = Field(min_length=1, max_length=64)
    filings_24m: int = Field(ge=0, le=100)
    late_filings_24m: int = Field(ge=0, le=100)
    payment_ratio_12m: float | None = Field(default=None, ge=0, le=5)
    days_since_last_payment: int | None = Field(default=None, ge=0, le=5000)
    turnover_growth_yoy: float | None = Field(default=None, ge=-10, le=10)
    business_size: Literal["MICRO", "SMALL", "MEDIUM", "LARGE"] = "SMALL"
    region: str = Field(default="UNKNOWN", min_length=1, max_length=64)


class RiskRequest(BaseModel):
    records: list[RiskRecord] = Field(min_length=1, max_length=MAX_BATCH)


class AssistantRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    user_role: Literal["PUBLIC", "OFFICER"] = "PUBLIC"


class ComplaintRequest(BaseModel):
    text: str = Field(min_length=3, max_length=2000)


class MeterRequest(BaseModel):
    reading_value: int | None = Field(default=None, ge=0, le=99999999)
    previous_reading: int | None = Field(default=None, ge=0, le=99999999)
    ocr_confidence: float = Field(default=0.91, ge=0, le=1)


def risk_score(r: RiskRecord) -> float:
    # Transparent demo scorer; production uses the versioned sklearn artifact.
    score = 0.08
    score += min(r.late_filings_24m / max(r.filings_24m, 1), 1.0) * 0.35
    score += (1 - (r.payment_ratio_12m if r.payment_ratio_12m is not None else 0.45)) * 0.35
    score += min((r.days_since_last_payment or 365) / 365, 1.0) * 0.15
    score += (0.08 if r.business_size == "MICRO" else 0.0)
    score += max(0.0, -(r.turnover_growth_yoy or 0)) * 0.04
    return round(max(0.0, min(1.0, score)), 4)


def band(score: float) -> str:
    return "HIGH" if score >= 0.65 else "MEDIUM" if score >= 0.35 else "LOW"


def language(text: str) -> str:
    # Lightweight demo heuristic. Production uses a validated Somali/English classifier.
    return "so" if re.search(r"\b(canshuur|bixin|shuruud|sidee|waa)\b", text.lower()) else "en"


KNOWLEDGE = [
    {"doc_id": "FAQ-001", "section": "Filing deadlines", "lang": "en", "text": "Taxpayers should file by the due date shown on their filing notice. If a filing is late, contact a Revenue Officer for assistance."},
    {"doc_id": "FAQ-002", "section": "Payment channels", "lang": "en", "text": "Supported payment channels include EVC, ZAAD, EDAHAB, bank and card. Keep the payment reference."},
    {"doc_id": "FAQ-003", "section": "Taariikhda gudbinta", "lang": "so", "text": "Canshuur-bixiyuhu waa inuu gudbiyaa foomka taariikhda ku qoran ogeysiiska. Haddii ay dib u dhacdo, la xiriir sarkaalka dakhliga."},
]


def answer_question(question: str, user_role: str) -> dict:
    lang = language(question)
    q = question.lower()
    candidates = [c for c in KNOWLEDGE if c["lang"] == lang and any(w in c["text"].lower() or w in c["section"].lower() for w in re.findall(r"[a-zA-Z]{3,}", q))]
    if not candidates:
        candidates = [c for c in KNOWLEDGE if c["lang"] == lang]
    if not candidates:
        msg = "Ma helin xog ku filan. Fadlan la xiriir sarkaalka dakhliga." if lang == "so" else "I could not find a grounded answer. Please contact a Revenue Officer."
        return {"answer": msg, "citations": [], "language": lang, "grounded": False}
    chosen = candidates[0]
    return {"answer": chosen["text"], "citations": [{"doc_id": chosen["doc_id"], "section": chosen["section"]}], "language": lang, "grounded": True}


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "ricap-ai", "version": app.version}


@app.get("/ready")
def ready() -> dict:
    return {"status": "ready", "model_loaded": True, "model_version": MODEL_VERSION}


@app.post("/v1/risk-scores")
def risk_scores(payload: RiskRequest, user: Annotated[User, Depends(verify_token)]) -> dict:
    if user.role not in {"RISK_ANALYST", "SYSTEM"}:
        raise HTTPException(status_code=403, detail="Role is not allowed")
    results = []
    for r in payload.records:
        score = risk_score(r)
        results.append({"taxpayer_id": r.taxpayer_id, "risk_score": score, "risk_band": band(score), "model_version": MODEL_VERSION})
    logger.info("scored_batch size=%s model_version=%s", len(results), MODEL_VERSION)
    return {"model_version": MODEL_VERSION, "results": results}


@app.post("/v1/assistant")
def assistant(payload: AssistantRequest) -> dict:
    return answer_question(payload.question, payload.user_role)


@app.post("/v1/complaints")
def complaints(payload: ComplaintRequest) -> dict:
    text = payload.text.lower()
    labels = {"payment": ["payment", "pay", "lacag", "bixin"], "filing": ["filing", "return", "foom"], "water": ["water", "meter", "biyo"], "technical": ["portal", "login", "app"]}
    category = max(labels, key=lambda k: sum(term in text for term in labels[k]))
    if not any(term in text for term in labels[category]): category = "general"
    return {"category": category, "confidence": 0.78 if category != "general" else 0.42, "route": "revenue-support"}


@app.post("/v1/meter-review")
def meter_review(payload: MeterRequest) -> dict:
    reasons = []
    if payload.ocr_confidence < 0.75: reasons.append("low_ocr_confidence")
    if payload.reading_value is None: reasons.append("unreadable_display")
    if payload.previous_reading is not None and payload.reading_value is not None and payload.reading_value < payload.previous_reading: reasons.append("decreasing_reading")
    return {"reading_value": payload.reading_value, "confidence": payload.ocr_confidence, "requires_review": bool(reasons), "flags": reasons}

