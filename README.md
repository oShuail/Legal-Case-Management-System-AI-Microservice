# Legal Case Management System – AI Microservice

[![CI](https://github.com/oShuail/Legal-Case-Management-System-AI-Microservice/actions/workflows/ci.yml/badge.svg)](https://github.com/oShuail/Legal-Case-Management-System-AI-Microservice/actions/workflows/ci.yml)

**Status:** ✅ Production Ready | **Backend Integration:** Ready  
**Master API:** https://orca-app-uayze.ondigitalocean.app/

This repository contains the **AI microservice** for the _Legal Case Management System (LCMS)_ project.

The service provides AI-powered semantic matching between legal cases and regulations using multilingual embeddings (Arabic/English).

---

## 🎯 What It Does

1. **Analyzes legal cases** - Receives case text (title + description)
2. **Matches against regulations** - Compares against all available laws/regulations
3. **Ranks by relevance** - Returns matches with confidence scores (0.0-1.0)
4. **Enables quick linking** - Backend stores results as case-regulation relationships
5. **Explains why a match happened** - Returns line-level evidence and confidence breakdown

**Use Case:** When a lawyer creates a labor dispute case, they click "Generate AI Suggestions" and the system finds all relevant labor laws automatically.

---

## 🚀 Quick Start

### 1. Start the Service
```bash
cd Legal-Case-Management-System-AI-Microservice
pip install -r requirements.txt
python -m uvicorn ai_service.app.main:app --host 0.0.0.0 --port 8000
```

### 2. View API Documentation
Open: **http://localhost:8000/docs** (Swagger UI)

### 3. Test the Integration Endpoint
See [QUICKSTART.md](QUICKSTART.md) for sample requests

---

## 📚 Documentation

| Document | Purpose |
|----------|---------|
| **[QUICKSTART.md](QUICKSTART.md)** | Get running in 5 minutes ⭐ |
| **[INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md)** | Complete backend integration (500+ lines) ⭐ |
| **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** | What was implemented and why |
| **[VERIFICATION_CHECKLIST.md](VERIFICATION_CHECKLIST.md)** | Quality assurance checklist |
| **[context/API_SPECIFICATION.md](context/API_SPECIFICATION.md)** | Backend API contract |
| **[context/AI_API_SPECIFICATION.md](context/AI_API_SPECIFICATION.md)** | AI service endpoints |
| **[context/LCMS_Comprehensive_Context.md](context/LCMS_Comprehensive_Context.md)** | System architecture |

---

## 📋 API Endpoints

### ⭐ NEW: POST `/similarity/find-related`

Find regulations related to a case. **For backend integration.**

**Request:**
```json
{
  "case_text": "Labor dispute regarding wrongful termination",
  "case_fragments": [
    {
      "fragment_id": "case:description",
      "text": "Employer terminated employee without notice and withheld dues",
      "source": "case"
    }
  ],
  "regulations": [
    {
      "id": 1,
      "title": "Saudi Labor Law",
      "category": "labor",
      "regulation_version_id": 44,
      "content_text": "Article 77 regarding termination...",
      "candidate_chunks": [
        {
          "chunk_id": 9001,
          "chunk_index": 2,
          "line_start": 120,
          "line_end": 146,
          "article_ref": "Article 77",
          "text": "Termination compensation and notification requirements..."
        }
      ]
    }
  ],
  "top_k": 10,
  "threshold": 0.3,
  "strict_mode": true
}
```

**Response:**
```json
{
  "related_regulations": [
    {
      "regulation_id": 1,
      "matched_regulation_version_id": 44,
      "title": "Saudi Labor Law",
      "category": "labor",
      "similarity_score": 0.92,
      "evidence": [
        {
          "fragment_id": "case:description",
          "source": "case",
          "score": 0.91
        }
      ],
      "line_matches": [
        {
          "case_fragment_id": "case:description",
          "case_snippet": "terminated employee without notice",
          "regulation_chunk_id": 9001,
          "regulation_snippet": "Termination compensation and notification requirements",
          "line_start": 120,
          "line_end": 146,
          "article_ref": "Article 77",
          "pair_score": 0.89,
          "contribution": 0.42
        }
      ],
      "score_breakdown": {
        "semantic_max": 0.91,
        "support_coverage": 0.86,
        "lexical_overlap": 0.57,
        "category_prior": 1.0,
        "final_score": 0.92
      },
      "warnings": []
    }
  ],
  "query_length": 48,
  "candidates_count": 1
}
```

### Existing Endpoints

- **POST `/embed/`** - Generate embeddings for text
- **POST `/similarity/`** - Rank documents by similarity (text-based)
- **POST `/regulations/extract`** - Fetch + extract regulation text (parser + OCR fallback)
- **POST `/documents/extract`** - Extract attachment text (shared parser/OCR pipeline)
- **POST `/documents/case-insights`** - Generate case-focused summary + related snippets
- **GET `/`** - Root endpoint
- **GET `/health/`** - Liveness check (service is up)
- **GET `/health/embeddings`** - Embedding-provider readiness (drives the
  frontend "AI assistant is warming up" banner; see section below)

---

## 🧯 Embedding Provider Fallback & Lazy Loading

The embedding service is configured with a **provider order** so a single
hosted-endpoint outage doesn't take AI features down. The order is read from
`HF_EMBED_PROVIDER_ORDER` (default: `["endpoint", "serverless", "bge"]`):

| # | Provider                | What it is                                                                                       | Why                              |
| - | ----------------------- | ------------------------------------------------------------------------------------------------ | -------------------------------- |
| 1 | `endpoint`              | Dedicated HF Inference Endpoint serving the fine-tuned `devSaleh/BGE-M3-Saudi-Legal-Cases-regulations` model. | Primary. Fast, scalable, paid.   |
| 2 | `serverless`            | HF serverless inference API for the same model.                                                  | Secondary. Cheaper, rate-limited. |
| 3 | `bge`                   | Local `sentence-transformers` BGE-M3 loaded in-process.                                          | Last-resort fallback.            |

### Lazy load (memory-saving)

The local `bge` provider is **lazy-loaded**: it is _not_ initialised at startup
even when listed in the provider order. The model + PyTorch + tokenizers
weigh **6–8 GB of RAM**, and the hosted endpoint covers > 99% of requests,
so paying that memory cost upfront is wasteful. The first time fallback is
actually needed, `_init_local_bge_model()` is called inside the request loop
(see [`embeddings.py`](ai_service/app/core/embeddings.py)) — it is idempotent
so subsequent calls are no-ops. The cold load takes ~50 s on CPU. Switching
back to eager loading just means restoring the previous unconditional call in
`EmbeddingService.__init__`.

### Health endpoint (`GET /health/embeddings`)

Returns the provider state so the frontend can surface a friendly banner
while the fallback is loading or actively serving. Sample response:

```json
{
  "configured_provider": "hf",
  "hf_provider_order": ["endpoint", "serverless", "bge"],
  "local_model_state": "not_loaded",
  "last_provider": "endpoint",
  "fallback_active": false,
  "warming_up": false
}
```

| Field                 | Meaning                                                                            |
| --------------------- | ---------------------------------------------------------------------------------- |
| `configured_provider` | The top-level provider mode (`hf`, `bge`, `fake`).                                 |
| `hf_provider_order`   | The active fallback chain.                                                         |
| `local_model_state`   | `not_loaded` \| `loading` \| `ready` — drives the "warming up" message.             |
| `last_provider`       | Which provider served the most recent successful embed call.                       |
| `fallback_active`     | `true` when `last_provider == "bge"` while the configured provider is `hf`.        |
| `warming_up`          | `true` while the local model is in the middle of its one-time cold load.           |

### End-to-end UX flow (when the hosted endpoint goes down)

```
┌──────────────────┐  GET /api/ai/health   ┌──────────────────┐  GET /health/embeddings  ┌─────────────────┐
│  Frontend        │ ───────────────────▶ │  Node backend    │ ────────────────────────▶ │  AI microservice │
│  (Next.js)       │                       │  (Fastify)       │                           │  (FastAPI)       │
│                  │ ◀─────────────────── │  5s cache,       │ ◀──────────────────────── │  reads           │
│  <AIStatusBanner>│  {ready, warmingUp,  │  3s timeout      │  {warming_up,             │  EmbeddingService│
│  polls every     │   fallbackActive,    │                  │   fallback_active, …}     │  state           │
│  5s while warming│   message}           │                  │                           │                  │
│  else 60s        │                       │                  │                           │                  │
└──────────────────┘                       └──────────────────┘                           └─────────────────┘
```

What the user sees:

1. **Normal day** — banner hidden, AI features are served by the fine-tuned
   model on the hosted endpoint.
2. **Hosted endpoint outage, first request** — fallback chain advances to
   `bge`, the local model starts loading. Within ~5 s the frontend banner
   shows: _"AI assistant is warming up — your request will be ready shortly."_
   The user's request returns once the cold load completes (~50 s) and the
   embedding call runs.
3. **Subsequent requests during the outage** — local model is in RAM, so
   they are fast. Banner switches to amber: _"AI assistant is running on the
   backup service. Some requests may be slightly slower than usual."_
4. **Hosted endpoint recovers** — next successful request flips
   `last_provider` back to `endpoint`, the banner clears within ~60 s.

### Files involved

| Layer            | File                                                                                                    |
| ---------------- | ------------------------------------------------------------------------------------------------------- |
| AI microservice  | [`ai_service/app/core/embeddings.py`](ai_service/app/core/embeddings.py) — state tracking + lazy init   |
| AI microservice  | [`ai_service/app/api/routes/health.py`](ai_service/app/api/routes/health.py) — `/health/embeddings`     |
| Node backend     | `Legal-Case-Management-System/src/services/ai-client.service.ts` — `getEmbeddingsHealth()`              |
| Node backend     | `Legal-Case-Management-System/src/routes/ai/index.ts` — `GET /api/ai/health` (cached 5 s)               |
| Frontend         | `Legal_Case_Management_Website/src/lib/hooks/use-ai-health.ts` — adaptive polling                       |
| Frontend         | `Legal_Case_Management_Website/src/components/features/ai/ai-status-banner.tsx` — banner UI             |
| Frontend         | `Legal_Case_Management_Website/src/app/(dashboard)/layout.tsx` — mounts the banner globally             |

### Tuning

- **To skip the fallback entirely** (saves memory but accepts a hard failure
  on hosted-endpoint outages): set
  `HF_EMBED_PROVIDER_ORDER=["endpoint","serverless"]`.
- **To eager-load the fallback** (pre-pay the ~7 GB so the first fallback
  request is instant): restore the unconditional `_init_local_bge_model()`
  call in `EmbeddingService.__init__`. There's a comment in that file
  marking the spot.

### Worker deployment modes (Node backend, not this service)

Unrelated to embeddings but relevant to deployment: the Node backend's
background work was split into three buckets — interactive (sync HTTP),
scheduled cron, and real async extraction. You can deploy in two modes:

| Mode               | Pods | Run command                                  | When to use                                            |
| ------------------ | ---- | -------------------------------------------- | ------------------------------------------------------ |
| **Combined**       | 1    | `npm run worker` (or `dist/workers/combined.worker.js`) | Default. Small/medium load. Cheapest.                |
| **Split**          | 2    | `npm run worker:scheduler` + `npm run worker:extraction` | When extraction volume justifies horizontal scaling. |

The split mode is safe to flip on at any time: `runPendingExtractions`
already uses `SELECT ... FOR UPDATE SKIP LOCKED` so multiple extraction
pods grab disjoint row sets without races. The scheduler stays at one pod
because it holds a Postgres advisory lock for subscription runs.

---

## 🏗️ Tech Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| Framework | FastAPI | 0.115.6 |
| Server | Uvicorn | latest |
| Python | Python | 3.12 |
| Model | BAAI/bge-m3 | multilingual |
| Embeddings | sentence-transformers | 3.3.1 |
| Config | Pydantic v2 | latest |
| Testing | pytest | 8.3.5 |
| Logging | loguru | latest |
| Containerization | Docker | latest |

---

## 📁 Project Structure

```
ai_service/
├── app/
│   ├── main.py                      # FastAPI app
│   ├── config.py                    # Settings (environment variables)
│   ├── api/
│   │   ├── routes/
│   │   │   ├── health.py
│   │   │   ├── embeddings.py
│   │   │   ├── similarity.py
│   │   │   └── find_related.py      # ⭐ NEW endpoint for backend
│   │   └── schemas/
│   │       ├── requests.py
│   │       └── responses.py
│   ├── core/
│   │   ├── embeddings.py
│   │   ├── models.py
│   │   └── similarity.py
│   ├── tests/
│   │   ├── test_api.py
│   │   ├── test_similarity_core.py
│   │   ├── test_integration.py      # ⭐ NEW
│   │   └── test_integration_manual.py # ⭐ NEW
│   └── utils/
│       └── logger.py
│
├── scripts/                         # BGE fine-tuning pipeline
│   ├── _shared/
│   │   ├── arabic_utils.py          # Arabic normalization + number word conversion
│   │   ├── db_client.py             # Direct PostgreSQL queries
│   │   └── paths.py                 # Shared path constants
│   ├── scrape_moj_judgments.py      # Step 1: Scrape MOJ judicial decisions API
│   ├── extract_citations.py         # Step 2: Extract regulation citations from text
│   ├── qa_sample.py                 # Step 2b: Sample citations for manual QA review
│   ├── build_training_triplets.py   # Step 3: Build (query, positive, negative) triplets
│   └── fine_tune_bge.py             # Step 4: Fine-tune BGE-M3 on Saudi legal data
│
├── data/                            # Pipeline outputs (gitignored)
│   ├── raw/                         # judgments.jsonl
│   ├── citations/                   # citations.jsonl, citations_stats.json
│   ├── triplets/                    # train.jsonl, val.jsonl, stats.json
│   └── qa/                          # qa_sample.csv
│
├── models/
│   ├── models--BAAI--bge-m3/        # Pre-downloaded base model
│   ├── bge-m3-saudi-legal-v1/       # Fine-tuned model output (gitignored)
│   └── evaluation_report.json       # Baseline vs fine-tuned metrics
│
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── pytest.ini
└── README.md
```

---

## 🧠 BGE-M3 Fine-Tuning Pipeline

Fine-tune `BAAI/bge-m3` on real Saudi judicial decisions to improve case-to-regulation retrieval quality. The pipeline scrapes the MOJ (Ministry of Justice) public API, extracts regulation citations from judgment text, builds training triplets, and trains the model.

**Hardware requirement:** NVIDIA GPU with ≥ 8GB VRAM (tested on RTX 4070 Ti Super, 16GB).

### Prerequisites

Add the following to your `.env` file (same PostgreSQL DB the backend uses):

```env
DATABASE_URL=postgresql://user:pass@host:5432/dbname
```

Install the pipeline dependencies (already in `requirements.txt`):

```bash
pip install psycopg2-binary accelerate tqdm requests
```

### Running the Pipeline

```bash
cd Legal-Case-Management-System-AI-Microservice

# 1) Register raw data sources once.
# Existing normalized JSONL sources:
python -m ai_service.scripts.run_finetune_pipeline source-add-jsonl \
  --source-id moj-initial \
  --path ai_service/data/raw/judgments.jsonl \
  --description "Initial MOJ scrape batch"

python -m ai_service.scripts.run_finetune_pipeline source-add-jsonl \
  --source-id moj-followup \
  --path ai_service/data/raw/judgments2.jsonl \
  --description "Follow-up MOJ scrape batch"

# Future MOJ scrape sources can also be registered directly:
python -m ai_service.scripts.run_finetune_pipeline source-add-moj \
  --source-id moj-commercial-2026q2 \
  --start-page 1 --max-pages 0 --delay 1.5 --page-gap 5

# 2) Inspect the active source registry.
python -m ai_service.scripts.run_finetune_pipeline source-list

# 3) Run one unified pipeline over all enabled sources.
python -m ai_service.scripts.run_finetune_pipeline run \
  --run-id fine-tuning-v3 \
  --all-enabled-sources \
  --skip-qa \
  --base-model BAAI/bge-m3 \
  --embedding-model BAAI/bge-m3 \
  --batch-size 4 \
  --max-seq-length 384 \
  --eval-batch-size 2 \
  --skip-epoch-eval \
  --checkpoint-save-steps 250 \
  --checkpoint-save-total-limit 2
```

Any future source that can export the normalized raw judgment schema can be added through `source-add-jsonl`, including database exports and external partner datasets. MOJ scraping remains available through `source-add-moj`.

The unified orchestrator now owns the normal workflow:

- It ingests one or more registered sources.
- It writes source-specific raw snapshots under `ai_service/data/pipeline_runs/<run_id>/sources/`.
- It merges and deduplicates all source records into one canonical raw dataset for that run.
- It extracts citations, optionally writes a QA sample, builds triplets, and fine-tunes a model using run-scoped artifact paths.
- It stores a run manifest at `ai_service/data/pipeline_runs/<run_id>/run_manifest.json`.
- It stores the model at `ai_service/models/pipeline_runs/<run_id>/model`.
- It updates aliases at `ai_service/data/pipeline/latest_run.json` and `ai_service/models/latest_model.json`.

This keeps new sources additive instead of creating parallel `judgments2`, `citations2`, `triplets2`, or manual merge-only flows.

### Importing Existing Legacy Runs

Current one-off runs can be registered in the unified manifest system without moving files:

```bash
python -m ai_service.scripts.run_finetune_pipeline import-legacy \
  --run-id legacy-v1 \
  --raw ai_service/data/raw/judgments.jsonl \
  --citations ai_service/data/citations/citations.jsonl \
  --triplets-dir ai_service/data/triplets \
  --model-dir ai_service/models/bge-m3-saudi-legal-v1 \
  --evaluation-report ai_service/models/evaluation_report.json
```

### Advanced / Stage-Level Scripts

The lower-level scripts still exist for debugging or one-off execution, but the recommended flow is the orchestrator above:

```bash
python -m ai_service.scripts.scrape_moj_judgments
python -m ai_service.scripts.extract_citations
python -m ai_service.scripts.qa_sample
python -m ai_service.scripts.build_training_triplets
python -m ai_service.scripts.fine_tune_bge
```

### CLI Options

| Script | Key flags |
|--------|-----------|
| `run_finetune_pipeline source-add-jsonl` | `--source-id`, `--path`, `--description` |
| `run_finetune_pipeline source-add-moj` | `--source-id`, `--start-page`, `--max-pages`, `--delay`, `--page-gap` |
| `run_finetune_pipeline run` | `--run-id`, `--all-enabled-sources`, `--stages`, `--skip-qa`, `--base-model`, `--embedding-model`, `--checkpoint-save-steps` |
| `run_finetune_pipeline import-legacy` | `--run-id`, `--raw`, `--citations`, `--triplets-dir`, `--model-dir` |
| `scrape_moj_judgments` | `--max-pages N`, `--delay 0.5`, `--court-type 1`, `--start-page N` |
| `extract_citations` | `--input`, `--output`, `--regulations-cache` |
| `qa_sample` | `--n 200`, `--seed 42` |
| `build_training_triplets` | `--hard-neg-k 10`, `--max-hard-neg 3`, `--val-ratio 0.2`, `--max-query-chars 1500`, `--embedding-model`, `--device` |
| `fine_tune_bge` | `--batch-size 8`, `--epochs 3`, `--lr 2e-5`, `--max-seq-length 512`, `--evaluation-report`, `--no-fp16` |

### Verifying Results

| Step | Check |
|------|-------|
| After `run_finetune_pipeline run` | `ai_service/data/pipeline_runs/<run_id>/run_manifest.json` should show completed stages |
| After merge stage | `ai_service/data/pipeline_runs/<run_id>/merged/judgments.jsonl` should contain the deduplicated union of active sources |
| After citations stage | `ai_service/data/pipeline_runs/<run_id>/citations/citations_stats.json` should show a healthy citation rate |
| After triplets stage | `ai_service/data/pipeline_runs/<run_id>/triplets/stats.json` should show the train/val counts for that run |
| After training stage | `ai_service/models/pipeline_runs/<run_id>/evaluation_report.json` should show the fine-tuned metrics |

### Deploying the Fine-Tuned Model

No code changes needed. Update your `.env`:

```env
EMBEDDING_MODEL_NAME=./ai_service/models/pipeline_runs/<run_id>/model
EMBEDDING_DEVICE=cuda
```

The existing `EmbeddingService` (`app/core/embeddings.py`) accepts both HuggingFace model IDs and local paths — the fine-tuned model loads automatically on the next service restart. After restarting, re-embed all regulation chunks via the backend's reindex endpoint to use the updated embeddings.

---

## 🔧 Configuration

### Environment Variables

```env
# App info
APP_NAME=AI Microservice
APP_VERSION=0.1.0
ENV=development
DEBUG=true
LOG_LEVEL=INFO

# Server
HOST=0.0.0.0
PORT=8000

# Embeddings
EMBEDDINGS_PROVIDER=bge
EMBEDDING_MODEL_NAME=BAAI/bge-m3
EMBEDDING_DEVICE=cpu

# CORS
CORS_ORIGINS=["http://localhost:3000","http://localhost:5173"]

# Backend API (new)
BACKEND_API_URL=https://orca-app-uayze.ondigitalocean.app
BACKEND_API_KEY=

# Regulation extraction / OCR
OCR_PRIMARY_PROVIDER=alapi
OCR_SECONDARY_PROVIDER=none
ALAPI_BASE_URL=https://alapi.deep.sa
ALAPI_API_KEY=
ALAPI_OCR_PATH=/ocr
SOURCE_WHITELIST_DOMAINS=laws.boe.gov.sa,laws.moj.gov.sa,boe.gov.sa,moj.gov.sa
EXTRACTION_TIMEOUT_SECONDS=30
EXTRACTION_MAX_BYTES=15000000
EXTRACTION_MAX_CHARS=120000
OCR_MIN_TEXT_CHARS=400
OCR_STRICT_MODE=false

# Document case insights
INSIGHTS_DEFAULT_TOP_K=5
INSIGHTS_MAX_SOURCE_CHARS=15000
INSIGHTS_SUMMARY_SENTENCES=4
REG_INSIGHTS_MAX_SOURCE_CHARS=40000
REG_IMPACT_MAX_SOURCE_CHARS=40000

# Optional LLM provider for regulation summary/impact generation
LLM_PROVIDER=heuristic
LLM_BASE_URL=
LLM_API_KEY=
LLM_MODEL=
LLM_TIMEOUT_SECONDS=30
```

---

## 🧪 Testing

```bash
# Run all tests
pytest

# Test integration endpoint
python app/tests/test_integration_manual.py

# Run async tests
python app/tests/test_integration.py
```

---

## 🚀 Deployment

### Local Development
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Docker
```bash
docker build -t ai-service .
docker run -p 8000:8000 ai-service
```

### Docker Compose
```bash
docker-compose up -d
```

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| Model Size | ~560MB |
| Memory | ~2GB |
| First Request | ~5-10s (model load) |
| Typical Request | 200-500ms |
| Max Regulations | 50+ |

---

## 🔒 Security

- ✅ Input validation (type & length checking)
- ✅ CORS properly configured
- ✅ Error messages safe (no leaks)
- ✅ No SQL injection risks
- ✅ Rate limiting compatible

---

## 🎯 Backend Integration

The backend should implement:

1. **AI Client Service** - HTTP wrapper for `/similarity/find-related`
2. **Route Handler** - `POST /api/ai-links/:caseId/generate`
3. **Database Storage** - Save results to `ai_links` table
4. **Error Handling** - Graceful degradation if AI service unavailable

**See [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md) for complete TypeScript examples.**

---

## ✅ What's New (v1.2.0)

- ✨ **New Endpoint:** `POST /similarity/find-related` for backend integration
- ✨ **New Endpoint:** `POST /documents/case-insights` for case-focused attachment insights
- ✨ **New Endpoint:** `POST /documents/extract` for OCR-aware attachment extraction
- ✨ **Explainability:** line-level matches, evidence fragments, and confidence score breakdown in `/similarity/find-related`
- 📝 **Comprehensive Docs:** 1500+ lines of integration guides
- 🧪 **Test Scripts:** Integration test suite
- 🔒 **Better Error Handling:** Descriptive messages & logging
- ⚙️ **Configuration:** Environment-based setup for backend API

See [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) for details.

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Port already in use | Use different port: `--port 8001` |
| Model not found | Auto-downloads on first run (~560MB) |
| No matches | Lower threshold, add regulation content |
| Slow performance | First request loads model (normal) |
| `ModuleNotFoundError: No module named 'app'` | Start from repo root with `python -m uvicorn ai_service.app.main:app ...` |
| `ModuleNotFoundError: No module named 'bs4'` | Install dependencies with `pip install -r requirements.txt` |

---

## 📞 Support

1. **Quick Start?** → [QUICKSTART.md](QUICKSTART.md)
2. **Backend Integration?** → [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md)
3. **API Questions?** → http://localhost:8000/docs
4. **Issues?** → Check logs & [VERIFICATION_CHECKLIST.md](VERIFICATION_CHECKLIST.md)

---

## 📜 License

Copyright © 2026 Legal Case Management System. All rights reserved.

---

**Status:** ✅ Production Ready  
**Last Updated:** January 31, 2026  
**Backend Integration:** Ready (see [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md))

**Ready to integrate?** Start with [QUICKSTART.md](QUICKSTART.md)!
