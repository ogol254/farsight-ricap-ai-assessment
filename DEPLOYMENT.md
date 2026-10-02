# Connected demo deployment (work in progress)

The connected implementation is `app.connected:app`. The existing `app.main:app`
and Vercel entrypoint remain unchanged until the replacement passes deployment tests.

## Local run

```sh
pip install -r requirements-connected.txt
uvicorn app.connected:app --host 127.0.0.1 --port 8001
```

Local development uses a persistent SQLite file and photo directory under `artifacts/`.
Hosted deployments must use PostgreSQL and private Supabase storage; they fail at
startup instead of silently falling back to an ephemeral filesystem.

## Free hosted configuration

Use **Dockerfile.free**, one worker, Render Free, and port 10000. Do not enable
billing upgrades. The container builds a real UC1 model from synthetic time-series
data and downloads a pinned quantised open-weight model, not an inference API.
Free hosting can sleep, cold-start slowly and refuse requests at quota limits.
Simultaneous expensive inference is limited to one request to contain memory use.

Set these **server-side environment secrets**, never in Git or browser JavaScript:

- `DATABASE_URL`: Supabase session-pooler PostgreSQL URL, percent-encoded password,
  `sslmode=require`. The server role needs schema migration privileges.
- `SUPABASE_URL`: the project's HTTPS URL.
- `SUPABASE_SERVICE_KEY`: server-only storage credential.
- `SESSION_SECRET`: a randomly generated value of at least 32 bytes.

Create a private bucket named `ricap-photos`, with a 6 MB upload limit and JPEG-only
stored content. Uploads are decoded and re-encoded to remove EXIF/GPS metadata.
Disable automatic table exposure and enable RLS. Application tables must not be
readable by anonymous Data API clients. The API applies session ownership checks.

## What is real and what is demo data

- **UC1:** persisted fictional taxpayers/filings/payments → SQL aggregation →
  fitted scikit-learn model → ranked audit batch → saved results. Training uses
  reproducible synthetic quarters and the C1 time-split pipeline.
- **UC2:** database-backed English/Somali demo documents → vectorised text
  retrieval → local open-weight model → evidence validation → citations and history.
  On the smallest free deployment, retrieval uses character/word TF-IDF rather
  than the larger proposed multilingual neural encoder. A tiny model is not
  trusted to invent or translate legal statements: unsupported generation is
  replaced by the actual retrieved source text. Unrelated queries abstain.
- **UC3:** camera/file upload → pretrained PP-OCR image inference → confidence,
  ambiguity and previous-reading checks → private photo + record → separate human
  correction. It is **server-side**, not offline Android inference. It does not
  claim visual tamper detection. Multiple numeric regions require cropping/review.
- **UC4:** trained eight-class character TF-IDF/logistic model → confidence and
  alternatives → routing record. No external department is messaged automatically.

These are real software paths using **fictional data**, not measurements of real
taxpayer risk, Somali-language production quality or field meter accuracy. The
assessment's C2 numbers are not the demo's metrics.

## Authentication and limits

`POST /v1/sessions` issues a signed, 24-hour demo token. Use it in Swagger's
**Authorize** control or the `Authorization: Bearer ...` header. This public demo
identity is not production officer authentication. It never grants internal-doc access.
Results and photos are scoped to the session. Preserve the token to retain access;
creating a new session produces a separate history. No secrets are in the frontend.

The demo rejects files over 6 MB, unsupported image types and malformed input.
It limits request rates, concurrent image/LLM inference, stored result count (500)
and photos (100). These are defensive demo limits, not production capacity claims.
Do not submit sensitive or real taxpayer data. Rotate demo infrastructure credentials
after the assessment and explicitly archive/delete demo records when no longer needed.

## Verification

```sh
pip install -r requirements-dev.txt -r requirements-connected.txt
python -m unittest tests.test_assessment tests.test_connected -v
```

Tests run real model inference, OCR on generated digit/blank images, durable writes,
session isolation, invalid-image rejection, bilingual retrieval and abstention.
Local test-image success does not prove accuracy on arbitrary photographed meters.
Cloud PostgreSQL, private object storage, cold-starts, free-tier memory and actual
phone camera behaviour must also pass before the live deployment is called complete.
