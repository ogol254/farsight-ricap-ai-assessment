"""G1. Python; FastAPI, Pydantic, pandas, scikit-learn and joblib.

Run: uvicorn assessment_code.serve_uc1:app
Set MODEL_DIR to C1's trusted artifact directory. Load joblib only from a
controlled release source: deserialisation can execute code. The gateway's
verified identity dependency can replace the scoped-token example below.
"""
from contextlib import asynccontextmanager
import json
import logging
import os
from pathlib import Path
import secrets
from typing import Annotated

import joblib
import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field, model_validator

from app.main import RiskRecord, band
from assessment_code.train_uc1 import NUMERIC, CATEGORICAL

logger = logging.getLogger("ricap.model")


class User(BaseModel):
    roles: set[str]


def verify_token(authorization: Annotated[str | None, Header()] = None) -> User:
    expected = os.environ.get("RISK_API_TOKEN")
    if not expected:
        raise HTTPException(503, "Authentication is not configured")
    if not authorization or not secrets.compare_digest(authorization, f"Bearer {expected}"):
        raise HTTPException(401, "Invalid bearer token")
    return User(roles={"RISK_ANALYST"})


class ModelRecord(RiskRecord):
    sector: str = Field(min_length=1, max_length=64)


class Batch(BaseModel):
    records: list[ModelRecord] = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def cap_batch(cls, value):
        if isinstance(value, dict) and isinstance(value.get("records"), list) and len(value["records"]) > 1000:
            raise HTTPException(413, "At most 1,000 records are allowed")
        return value


def create_app(model_dir=None):
    @asynccontextmanager
    async def lifespan(app):
        app.state.pipeline = None
        app.state.version = None
        directory = Path(model_dir or os.environ.get("MODEL_DIR", "artifacts/uc1"))
        try:
            metadata = json.loads((directory / "metadata.json").read_text())
            if metadata["feature_list"] != NUMERIC + CATEGORICAL:
                raise ValueError("Artifact feature schema mismatch")
            pipeline = joblib.load(directory / "model.joblib")
            if not hasattr(pipeline, "predict_proba") or not metadata.get("model_version"):
                raise ValueError("Invalid artifact")
            app.state.pipeline = pipeline
            app.state.version = metadata["model_version"]
        except Exception as exc:
            # Keep liveness available; do not leak artifact contents or taxpayer data.
            logger.error("model_load_failed type=%s", type(exc).__name__)
        yield

    app = FastAPI(title="RICAP UC1 artifact-backed reference", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/ready")
    def ready():
        if app.state.pipeline is None:
            raise HTTPException(503, "Model not loaded")
        return {"status": "ready", "model_loaded": True, "model_version": app.state.version}

    @app.post("/v1/risk-scores")
    def score(payload: Batch, user: Annotated[User, Depends(verify_token)]):
        if not user.roles.intersection({"RISK_ANALYST", "SYSTEM"}):
            raise HTTPException(403, "Role is not allowed")
        if app.state.pipeline is None:
            raise HTTPException(503, "Model not loaded")
        frame = pd.DataFrame([record.model_dump() for record in payload.records])
        try:
            probabilities = app.state.pipeline.predict_proba(frame[NUMERIC + CATEGORICAL])[:, 1]
            if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
                raise ValueError("Invalid probabilities")
        except Exception as exc:
            logger.error("prediction_failed type=%s", type(exc).__name__)
            raise HTTPException(503, "Model prediction unavailable") from exc
        logger.info("scored_batch size=%s model_version=%s", len(frame), app.state.version)
        return {"model_version": app.state.version, "results": [
            {"taxpayer_id": record.taxpayer_id, "risk_score": float(p), "risk_band": band(float(p)), "model_version": app.state.version}
            for record, p in zip(payload.records, probabilities)
        ]}

    return app


app = create_app()
