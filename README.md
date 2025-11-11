# AI Microservice (FastAPI)

Minimal FastAPI service that will expose:
- `GET /health/` for status
- `POST /embed/` for text embeddings (soon)

## Quickstart
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000