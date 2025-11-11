from pydantic import BaseModel, Field
from typing import List

class EmbedRequest(BaseModel):
    """Request body for /embed endpoint."""
    texts: List[str] = Field(..., min_length=1, description="List of input strings to embed")
    normalize: bool = True

class SimilarityRequest(BaseModel):
    """Request body for /similarity endpoint (semantic search)."""
    queries: List[str] = Field(..., min_length=1)
    corpus: List[str] = Field(..., min_length=1)
    top_k: int = 5
