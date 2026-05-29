import random

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.errors.handlers import register_exception_handlers
from app.api.routes import (
    health,
    embeddings,
    similarity,
    find_related,
    find_related_multi_source,
    regulation_extract,
    regulation_insights,
    document_extract,
    document_insights,
    assistant,
    admin_insights,
)

random.seed(42)

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Canonical error envelope handlers — produces the same shape as the Node backend.
register_exception_handlers(app)


@app.get("/")
async def root():
    return {
        "app": settings.app_name,
        "version": settings.app_version,
        "env": settings.env,
    }


app.include_router(health.router, tags=["health"])
app.include_router(embeddings.router, tags=["embeddings"])
app.include_router(similarity.router, tags=["similarity"])
app.include_router(find_related.router, tags=["similarity"])
app.include_router(find_related_multi_source.router, tags=["similarity"])
app.include_router(regulation_extract.router, tags=["regulation-extraction"])
app.include_router(regulation_insights.router, tags=["regulation-insights"])
app.include_router(document_extract.router, tags=["document-extraction"])
app.include_router(document_insights.router, tags=["document-insights"])
app.include_router(assistant.router, tags=["assistant"])
app.include_router(admin_insights.router, tags=["admin-insights"])
