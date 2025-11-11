from pydantic import BaseModel
from typing import List

class EmbedResponse(BaseModel):
    embeddings: List[List[float]]
    dimension: int
    count: int

class SimilarityItem(BaseModel):
    doc: str
    score: float

class SimilarityResponse(BaseModel):
    results: List[List[SimilarityItem]]