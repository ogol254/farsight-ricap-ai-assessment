# RICAP AI Engineering Demonstrator

A review-friendly demonstrator for the Farsight Africa RICAP case study. It exposes typed, auditable API contracts and a lightweight browser UI for the four proposed use cases.

Live demo: https://farsight-ricap-ai-assessment.vercel.app/

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000. API documentation is available at `/docs`.

## Vercel deployment

The repository includes `vercel.json` and `api/index.py` for a Vercel Python deployment. Import the GitHub repository in Vercel and deploy with no secrets required for the synthetic demo.

The demo deliberately uses synthetic data and deterministic baseline components. It does not contain taxpayer records, credentials, or external AI API calls.

## Repository map

- [B3 architecture and diagrams](assessment_code/B3_architecture.md) — proposed government system and actual demo boundary.
- `app/` — FastAPI service and static review UI.
- `assessment_code/` — assessment-specific reference implementations: UC1 training, grounded RAG function, point-in-time SQL, Docker, Kubernetes, and CI.
- `tests/` — standard-library API smoke tests.

## Assessment references and implementation limits

The Vercel app uses deterministic risk rules, a small in-memory FAQ, keyword routing and checks on entered meter values. It has no connected database, trained model, LLM, vector store, image OCR or offline mobile client. Demo scores are not calibrated probabilities; complaint rules do not provide a confidence estimate. `/ready` reports demo availability and explicitly returns `model_loaded: false`.

The [G1 service](assessment_code/serve_uc1.py) is a separate runnable reference that loads the [C1 pipeline](assessment_code/train_uc1.py) once at startup. Run it with `MODEL_DIR=artifacts/uc1 uvicorn assessment_code.serve_uc1:app` and supply `RISK_API_TOKEN` securely. Train first with `train(df)` on the documented feature schema; no real training dataset is included. Missing artifacts produce a 503 readiness response. The [G2 Dockerfile](assessment_code/Dockerfile) packages this reference; the Kubernetes file is a deployment example requiring site-specific images, secrets and model storage.

The assessment's C2 numbers are a supplied hypothetical scenario, not measurements from this demo. C1 follows the requested retrospective split; an actual deployment must also account for when delayed audit labels become available.

## Security and deployment notes

The sample API token is for local demonstration only. Production deployment must use a secret manager, mTLS or an API gateway, structured redacted logs, real model artifacts, and an on-premise inference boundary for protected taxpayer data.
