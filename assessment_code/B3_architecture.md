# B3 — RICAP architecture

This is the proposed government deployment for the case study. It is **not** a diagram of services already connected to the public Vercel demo. Officers approve audits and penalties; model outputs are recommendations.

## Proposed production system

```mermaid
flowchart TB
    subgraph Channels[User channels]
        Portal[Revenue platform and officer portal]
        Phone[Officer Android app: encrypted offline queue]
        WA[WhatsApp: public guidance only]
    end

    subgraph Gov[Government data centre: protected data and inference]
        subgraph Sources[Sources and ingestion]
            PG[Revenue PostgreSQL] --> Replica[Read replica or throttled CDC]
            ERP[Water ERP] --> SFTP[Nightly SFTP CSV: checksum and deduplication]
            Docs[Approved document repository] --> OCR[OCR and legal version review]
            Photos[Meter photos] --> Upload[Authenticated upload and sync]
            Complaints[SMS and web complaints] --> Intake[Validated complaint intake]
        end
        Replica & SFTP & OCR & Upload & Intake --> Raw[Encrypted immutable raw object storage]
        Raw --> Validate[Airflow: validate, deduplicate and reconcile]
        Validate -->|Failed checks| Quarantine[Quarantine and operator alert]
        Validate --> Curated[Curated PostgreSQL and versioned Parquet]
        Curated --> Features[Feature store: shared point-in-time SQL and snapshots]
        Curated --> Index[Approved chunks: audience, language, dates and version]
        Index --> Vector[PostgreSQL pgvector and keyword index]

        subgraph ML[Training and release]
            Features --> Train[CPU training: time splits and mature labels]
            Curated --> GPUTrain[Scheduled NLP and vision training]
            Train & GPUTrain --> Tracking[MLflow experiments and dataset versions]
            Tracking --> Registry[Model registry and signed artifacts]
            Registry --> Gates[Quality, fairness and latency tests; human approval]
        end

        subgraph Serving[On-premise Kubernetes]
            Gateway[API gateway: authenticated roles and rate limits]
            Gates --> UC1[UC1: quarterly CPU batch scoring]
            Features --> UC1
            UC1 --> Audit[Rank at most 500; officer review and explanations]
            Gateway --> UC2[UC2: retrieval, citations and safe handover]
            Vector --> UC2
            UC2 --> LLM[Local quantised 8B LLM: vLLM on L4]
            Gates --> LLM
            Gateway --> UC4[UC4: real-time complaint classification]
            Gates --> UC4
            UC4 --> Queue[Department queues and uncertain-case review]
            Gateway --> MeterReview[UC3: sync, plausibility checks and human review]
            Gates --> Edge[Signed compact INT8 meter model]
        end
        Audit & Queue & MeterReview --> Workflow[Revenue workflow: officer decisions and feedback]
        Workflow --> Curated
        Serving & Validate & ML --> Obs[Prometheus and Grafana; redacted audit logs]
        Obs --> Archive[Append-only encrypted seven-year audit archive]
        Security[Cross-cutting: RBAC, TLS, key management, network policies and backups]
    end

    Portal --> Gateway
    Phone -->|Online sync| Gateway
    Edge -->|Approved model update| Phone
    Phone -->|On-device detection and OCR while offline| Phone
    WA -->|Public questions; reject personal-record requests| Gateway
    Workflow --> Portal

    subgraph DR[AWS disaster recovery: Ministry-approved scope]
        Backup[Encrypted replicas in S3; restricted KMS keys]
        Recovery[Versioned images, manifests and recovery runbook]
    end
    Raw & Curated & Registry & Archive -.->|Approved encrypted replication only| Backup
    Gates -.->|Approved release artifacts| Recovery
```

Read the main path as: collect data safely, check it, build consistent features, test models, and give officers recommendations. The feedback path captures corrections and completed audit outcomes for later evaluation. Failed data checks stop publication of the affected batch.

The feature store starts as governed SQL and versioned snapshots, not a separate platform to operate. UC1 runs quarterly on CPUs. UC4 starts with a CPU text classifier. UC3 runs a compact detector and digit reader on the phone, stores results securely while offline, and synchronises when a connection is available. UC2 retrieves authorised, current sources before asking the local language model for a cited answer.

WhatsApp is an external channel, so it must be limited to public guidance. Account-specific conversations move to the authenticated portal. Protected taxpayer records never go to public AI APIs. DR replication needs Ministry approval; failover must respect the same approved data-processing boundary and be tested through restore exercises.

## How the two L4 GPUs shape the design

Each L4 has 24 GB; the two cards do not automatically act as one 48 GB memory pool. I would start with a quantised 8B language model on one card and reserve the other for embeddings, reranking and scheduled experiments. UC1 and the initial UC4 classifier run on CPUs; UC3 runs on phones. I would cap context and answer length, batch inference, and cache only authorised public answers. Fifty concurrent users and a five-second response target require a measured load test. If that target fails, I would reduce generation work or agree more capacity before release, rather than promise untested performance.

## What the public demonstration actually connects

```mermaid
flowchart LR
    Git[Public GitHub repository] --> Vercel[Vercel Python deployment]
    Browser[Browser review interface] -->|HTTPS| API[FastAPI through api/index.py]
    Vercel --> API
    API --> Risk[Deterministic risk rules]
    API --> FAQ[Small in-memory bilingual FAQ]
    API --> Triage[Keyword complaint routing]
    API --> Meter[Entered-reading checks]
```

The public demo has no database, external LLM, vector database, image OCR, offline mobile app or model registry. It is a synthetic-data interface demonstration. The separate G1 reference service loads the C1 model artifact, but that reference service is not the Vercel application.

## Review links

- [Live demonstration](https://farsight-ricap-ai-assessment.vercel.app/)
- [Public demo API](../app/main.py)
- [C1 training pipeline](train_uc1.py)
- [D2 grounded-answer reference](rag_answer.py)
- [F1 PostgreSQL feature query](uc1_features.sql)
- [G1 artifact-backed serving reference](serve_uc1.py)
- [G2 container](Dockerfile) and [Kubernetes reference](k8s.yaml)

Model candidates must pass Somali-language and workload-specific evaluation. Sources: [Qwen3-8B model card](https://huggingface.co/Qwen/Qwen3-8B) and [BGE-M3 model card](https://huggingface.co/BAAI/bge-m3).
