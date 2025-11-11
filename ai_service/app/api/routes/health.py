from fastapi import APIRouter
from app.config import settings

router = APIRouter(prefix="/health", tags=["health"])

@router.get("/")
def health():
    return {
        "status": "healthy",
        "service": settings.app_name,
        "version": settings.app_version,
    }
