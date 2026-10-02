"""Connected UC1–UC4 service. Entrypoint: uvicorn app.connected:app."""
from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path
import secrets
import threading
import time
from typing import Annotated
from uuid import uuid4

from fastapi import FastAPI, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware

from app.main import RiskRecord, AssistantRequest, ComplaintRequest
from app.database import Repository
from app.demo_data import documents
from app.models import Models
from app.ocr import MeterOCR
from app.retrieval import Assistant
from app.storage import PhotoStore

logger = logging.getLogger("ricap.connected")
security = HTTPBearer(auto_error=False, description="Create a free demo session at POST /v1/sessions, then paste its access_token here.")


class ConnectedRiskRecord(RiskRecord):
    sector: str = Field(default="RETAIL", min_length=1, max_length=64)


class Batch(BaseModel):
    records: list[ConnectedRiskRecord] = Field(min_length=1, max_length=1000)


class Review(BaseModel):
    reading_value: float = Field(ge=0, le=99999999)
    reason: str = Field(min_length=3, max_length=500)


def create_app(database_url=None, model_dir="artifacts/connected-v1", photo_root="artifacts/photos", neural=None):
    secret = os.getenv("SESSION_SECRET")
    hosted = bool(os.getenv("SPACE_ID") or os.getenv("RENDER") or os.getenv("VERCEL"))
    if hosted and not secret:
        raise RuntimeError("Set SESSION_SECRET in hosting secrets before deploying")
    signer = URLSafeTimedSerializer(secret or secrets.token_urlsafe(32), salt="ricap-session-v1")
    gate = threading.BoundedSemaphore(1)
    limits = {}
    limits_lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(app):
        repository = Repository(database_url)
        app.state.repository = repository
        repository.seed_documents(documents())
        repository.seed_revenue()
        app.state.models = Models(model_dir)
        app.state.ocr = MeterOCR()
        app.state.storage = PhotoStore(photo_root)
        app.state.assistant = Assistant(repository, use_neural=neural if neural is not None else os.getenv("ENABLE_NEURAL", "0") == "1")
        if hosted and (not repository.persistent_remote or not app.state.storage.remote):
            raise RuntimeError("Hosted deployment requires persistent PostgreSQL and photo storage")
        yield
        repository.close()

    app = FastAPI(title="RICAP Connected AI Services", version="2.0.0", lifespan=lifespan,
        description="Real inference and persistent records using synthetic demo data. Start with POST /v1/sessions. No real taxpayer information, identity documents or sensitive photos.")
    app.add_middleware(CORSMiddleware, allow_origins=["https://farsight-ricap-ai-assessment.vercel.app"], allow_credentials=False,
                       allow_methods=["GET", "POST"], allow_headers=["Authorization", "Content-Type"])

    def limit(key, maximum=30):
        now = time.monotonic()
        with limits_lock:
            if len(limits) > 5000:
                for old in [k for k, (_, start) in limits.items() if now - start > 60]:
                    limits.pop(old, None)
            count, start = limits.get(key, (0, now))
            if now - start > 60:
                count, start = 0, now
            if count >= maximum:
                raise HTTPException(429, "Demo rate limit reached. Please try again in a minute.")
            limits[key] = (count + 1, start)

    def user(request: Request, auth: Annotated[HTTPAuthorizationCredentials | None, Depends(security)]):
        if not auth:
            raise HTTPException(401, "Create a demo session at /v1/sessions and supply its bearer token.", headers={"WWW-Authenticate": "Bearer"})
        try:
            claims = signer.loads(auth.credentials, max_age=86400)
        except (BadSignature, SignatureExpired):
            raise HTTPException(401, "Demo session is invalid or expired. Create a new session.")
        if claims.get("role") != "DEMO_REVIEWER" or not claims.get("sub"):
            raise HTTPException(403, "Demo role required")
        limit(claims["sub"])
        return claims["sub"]

    def expensive():
        if not gate.acquire(blocking=False):
            raise HTTPException(429, "The free CPU is busy. Please retry shortly.")
        try:
            yield
        finally:
            gate.release()

    def persist(owner, kind, result):
        try:
            if app.state.repository.count() >= 500:
                raise HTTPException(507, "Free demo record capacity reached. Ask the owner to archive records.")
            result["record_id"] = app.state.repository.save(owner, kind, result.copy())
            return result
        except HTTPException:
            raise
        except Exception:
            logger.exception("database_write_failed")
            raise HTTPException(503, "Result could not be saved. Please retry when the database is available.")

    @app.middleware("http")
    async def safety_headers(request, call_next):
        try:
            length = int(request.headers.get("content-length", "0"))
        except ValueError:
            return Response("Invalid content length", status_code=400)
        if length > 7 * 1024 * 1024:
            return Response("Upload exceeds 7 MB request limit", status_code=413)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        if request.url.path.startswith("/v1/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/health")
    def health():
        return {"status": "ok", "version": "2.0.0"}

    @app.get("/ready")
    def ready():
        try:
            app.state.repository.ping()
        except Exception:
            raise HTTPException(503, "Database unavailable")
        return {"status": "ready", "mode": "connected_demo", "database": "postgresql" if app.state.repository.persistent_remote else "sqlite_local",
                "model_loaded": True, "ocr_loaded": True, "classifier_loaded": True,
                "llm_loaded": app.state.assistant.generator is not None or app.state.assistant.small_llm is not None,
                "embeddings_loaded": app.state.assistant.encoder is not None,
                "photo_storage": "private_supabase" if app.state.storage.remote else "local_disk",
                "data_notice": "Synthetic training data and fictional guidance; no production accuracy claim"}

    @app.post("/v1/sessions")
    def session(request: Request):
        limit("sessions:" + (request.client.host if request.client else "unknown"), 12)
        return {"access_token": signer.dumps({"sub": str(uuid4()), "role": "DEMO_REVIEWER"}), "token_type": "bearer", "expires_in": 86400,
                "notice": "Public demo access only; never submit sensitive data. Save this token to access your own history."}

    @app.get("/v1/history")
    def history(owner: Annotated[str, Depends(user)]):
        return {"records": app.state.repository.history(owner)}

    @app.get("/v1/documents")
    def source_documents():
        return {"documents": app.state.repository.documents(), "notice": "Fictional demo guidance, not legal advice"}

    @app.get("/v1/model-info")
    def model_info():
        return app.state.models.metadata

    @app.get("/v1/taxpayers")
    def taxpayers(owner: Annotated[str, Depends(user)]):
        return {"records": app.state.repository.revenue_features(), "as_of": "2026-01-01", "data_source": "Fictional records aggregated from persisted taxpayers, filings and payments tables"}

    @app.post("/v1/audit-batches")
    def audit_batch(owner: Annotated[str, Depends(user)]):
        records = app.state.repository.revenue_features()
        results = app.state.models.score(records)
        results.sort(key=lambda r: (-r["risk_score"], r["taxpayer_id"]))
        return persist(owner, "audit_batch", {"results": results[:500], "as_of": "2026-01-01", "capacity": 500,
                       "selected_count": min(len(results), 500), "source": "database_feature_pipeline", "training_data": "synthetic"})

    @app.post("/v1/risk-scores")
    def risks(payload: Batch, owner: Annotated[str, Depends(user)]):
        records = [r.model_dump() for r in payload.records]
        if len({r["taxpayer_id"] for r in records}) != len(records):
            raise HTTPException(422, "Duplicate taxpayer identifiers in batch")
        results = app.state.models.score(records)
        ranked = sorted(results, key=lambda r: (-r["risk_score"], r["taxpayer_id"]))[:500]
        return persist(owner, "risk", {"results": results, "audit_selection": [r["taxpayer_id"] for r in ranked],
                     "model_version": app.state.models.metadata["model_version"], "training_data": "synthetic"})

    @app.post("/v1/assistant")
    def assistant(payload: AssistantRequest, owner: Annotated[str, Depends(user)], slot=Depends(expensive)):
        # Client-supplied OFFICER is not authority to read INTERNAL material.
        answer = app.state.assistant.answer(payload.question, "PUBLIC")
        return persist(owner, "assistant", answer)

    @app.post("/v1/complaints")
    def complaints(payload: ComplaintRequest, owner: Annotated[str, Depends(user)]):
        return persist(owner, "complaint", app.state.models.classify(payload.text))

    @app.post("/v1/meter-readings")
    async def meter(owner: Annotated[str, Depends(user)], image: UploadFile = File(...), previous_reading: float | None = Form(None, ge=0, le=99999999), slot=Depends(expensive)):
        if app.state.repository.count("meter") >= 100:
            raise HTTPException(507, "Free demo photo capacity reached. Ask the owner to archive photos.")
        raw = await image.read(6 * 1024 * 1024 + 1)
        await image.close()
        if len(raw) > 6 * 1024 * 1024:
            raise HTTPException(413, "Photo must be at most 6 MB. Crop or resize it first.")
        from starlette.concurrency import run_in_threadpool
        try:
            result, cleaned = await run_in_threadpool(app.state.ocr.read, raw, previous_reading)
        except ValueError as exc:
            raise HTTPException(422, str(exc))
        try:
            result["photo_key"] = await run_in_threadpool(app.state.storage.put, cleaned)
        except Exception:
            raise HTTPException(503, "Photo storage unavailable. No successful save was recorded.")
        return persist(owner, "meter", result)

    @app.get("/v1/meter-readings/{record_id}/photo")
    def photo(record_id: str, owner: Annotated[str, Depends(user)]):
        record = app.state.repository.get(owner, record_id)
        if not record or record.kind != "meter":
            raise HTTPException(404, "Reading not found")
        return Response(app.state.storage.get(record.payload["photo_key"]), media_type="image/jpeg")

    @app.post("/v1/meter-readings/{record_id}/review")
    def review(record_id: str, payload: Review, owner: Annotated[str, Depends(user)]):
        record = app.state.repository.get(owner, record_id)
        if not record or record.kind != "meter":
            raise HTTPException(404, "Reading not found")
        return persist(owner, "meter_review", {"reading_id": record_id, **payload.model_dump(), "source": "human_review", "original_ocr_preserved": True})

    @app.get("/", include_in_schema=False)
    def home():
        return FileResponse(Path(__file__).parent / "static" / "connected.html")

    app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")
    return app


app = create_app()
