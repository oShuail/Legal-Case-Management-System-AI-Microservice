from fastapi import APIRouter, Depends
from app.config import settings
from app.api.deps import get_embedding_service
from app.core.embeddings import EmbeddingService

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/")
def health():
    return {
        "status": "healthy",
        "service": settings.app_name,
        "version": settings.app_version,
    }


@router.get("/embeddings")
def embeddings_health(
    embedder: EmbeddingService = Depends(get_embedding_service),
):
    """
    Returns the embedding-provider state so the UI can surface a friendly
    banner whenever the local fallback model is loading or actively serving
    requests instead of the fast hosted endpoint.

    Response shape:
      {
        "configured_provider": "hf",
        "hf_provider_order": ["endpoint", "serverless", "bge"],
        "local_model_state": "not_loaded" | "loading" | "ready",
        "last_provider": "endpoint" | "serverless" | "bge" | null,
        "fallback_active": bool,
        "warming_up": bool
      }
    """
    return embedder.get_health()
