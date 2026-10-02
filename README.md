# RICAP AI Engineering Demonstrator

A connected, review-friendly demonstration for the Farsight Africa RICAP case study. All four workflows run real inference and save results. Data and guidance are explicitly fictional; this is not a validated revenue-enforcement system.

Live demo: https://farsight-ricap-ai-assessment.vercel.app/

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-connected.txt
uvicorn app.connected:app --reload
```

Open http://127.0.0.1:8000. API documentation is available at `/docs`.

## Connected deployment

Vercel serves the browser interface and proxies API requests to [Render](https://ricap-connected-demo.onrender.com). Render runs the FastAPI Docker service and CPU models. Supabase provides PostgreSQL and a private photo bucket. Infrastructure credentials live only in Render's server environment.

Hugging Face supplies a pinned open-weight model download during the image build; inference happens inside Render. There is no paid plan or external AI inference API. The free backend sleeps when idle, so the first request can take a minute or longer. See [deployment and operating limits](DEPLOYMENT.md).

| Service | Working data path |
| --- | --- |
| UC1 | PostgreSQL taxpayers, filings and payments → SQL features → fitted scikit-learn pipeline → saved risk ranking |
| UC2 | Stored English/Somali guidance → TF-IDF retrieval → local SmolLM2 → source-verification guardrail → cited answer and history |
| UC3 | Phone/file photograph → PP-OCRv4 plus meter-counter extraction → private photo and focused meter ID/reading record → separate human correction |
| UC4 | Complaint → trained eight-class text classifier → confidence, suggested route and saved record |

The browser creates its own restricted demo session. For API clients, call `POST /v1/sessions`, then supply the returned `access_token` as `Authorization: Bearer <token>`. Swagger is available at [/docs](https://farsight-ricap-ai-assessment.vercel.app/docs). Never use Supabase credentials as an application token.

## Repository map

- [B3 architecture and diagrams](assessment_code/B3_architecture.md) — proposed government system and actual demo boundary.
- `app/` — FastAPI service and static review UI.
- `assessment_code/` — assessment-specific reference implementations: UC1 training, grounded RAG function, point-in-time SQL, Docker, Kubernetes, and CI.
- [Connected integration tests](tests/test_connected.py) and [live verification](tests/verify_live.py).

## Assessment references and implementation limits

The connected demo replaces the earlier rule-only app. UC1 is trained on synthetic quarters; UC4 uses 80 authored examples. Their scores are not field-calibrated. UC2 uses lightweight text retrieval, not a dedicated vector database, and the tiny language model is restricted to verified source extracts. UC3 performs server-side OCR, not offline Android inference or visual tamper detection. Cropping and human review remain necessary for ambiguous photographs. WhatsApp, Ministry source systems, production monitoring and a model registry are proposed architecture components, not deployed integrations.

The [G1 service](assessment_code/serve_uc1.py) is a separate runnable reference that loads the [C1 pipeline](assessment_code/train_uc1.py) once at startup. Run it with `MODEL_DIR=artifacts/uc1 uvicorn assessment_code.serve_uc1:app` and supply `RISK_API_TOKEN` securely. Train first with `train(df)` on the documented feature schema; no real training dataset is included. Missing artifacts produce a 503 readiness response. The [G2 Dockerfile](assessment_code/Dockerfile) packages this reference; the Kubernetes file is a deployment example requiring site-specific images, secrets and model storage.

The assessment's C2 numbers are a supplied hypothetical scenario, not measurements from this demo. C1 follows the requested retrospective split; an actual deployment must also account for when delayed audit labels become available.

### UC3 OCR test image

Use [`tests/fixtures/watermeter.png`](tests/fixtures/watermeter.png) as the reference image when testing the meter OCR. The meter-specific counter pass was calibrated against this example to distinguish the red `220` m³ counter from the meter/customer ID `31120595` and nearby printed numbers. A successful OCR test should show:

```text
Meter ID: 31120595
Reading: 220 m³
```

The connected UI and saved UC3 record intentionally expose only the meter ID and reading. The original photo remains private for human review when the model is uncertain.

## Security and deployment notes

Signed demo tokens expire after 24 hours and isolate each session's records and photos. Application tables have RLS enabled with no anonymous Data API access. No confidential information should be submitted. Production deployment requires real officer identity, least-privilege service roles, evaluated models, operational monitoring and the required on-premise processing boundary. The old `app.main` and `api/index.py` are retained as historical reference code, not the connected deployment entrypoint.
