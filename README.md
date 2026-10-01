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

- `app/` — FastAPI service and static review UI.
- `assessment_code/` — assessment-specific reference implementations: UC1 training, grounded RAG function, point-in-time SQL, Docker, Kubernetes, and CI.
- `tests/` — standard-library API smoke tests.

## Security and deployment notes

The sample API token is for local demonstration only. Production deployment must use a secret manager, mTLS or an API gateway, structured redacted logs, real model artifacts, and an on-premise inference boundary for protected taxpayer data.
