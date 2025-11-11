# AI Microservice Implementation Plan - FastAPI

## Legal Case Management System - Semantic Similarity Service

> **Framework**: FastAPI with Python 3.11+  
> **ML Library**: sentence-transformers, transformers (Hugging Face)  
> **Primary Model**: BAAI/bge-m3 (multilingual, Arabic support)  
> **Similarity Metric**: Cosine similarity  
> **Optional**: BGE-reranker-v2-m3 for precision improvement  
> **Deployment**: Docker, Uvicorn

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Tech Stack](#tech-stack)
3. [Project Structure](#project-structure)
4. [Phase 1: Foundation Setup](#phase-1-foundation-setup)
5. [Phase 2: Model Loading & Management](#phase-2-model-loading--management)
6. [Phase 3: Embedding Generation](#phase-3-embedding-generation)
7. [Phase 4: Similarity Calculation](#phase-4-similarity-calculation)
8. [Phase 5: API Endpoints](#phase-5-api-endpoints)
9. [Phase 6: Optional Reranking](#phase-6-optional-reranking)
10. [Phase 7: Caching & Optimization](#phase-7-caching--optimization)
11. [Phase 8: Testing](#phase-8-testing)
12. [Phase 9: Deployment](#phase-9-deployment)

---

## Project Overview

### Core Functionality

The AI microservice is responsible for:

- **Embedding Generation**: Convert Arabic/English legal texts into dense vector representations
- **Semantic Similarity**: Find the most relevant regulations for a given case using cosine similarity
- **Optional Reranking**: Improve top-K precision using cross-encoder models
- **Model Management**: Load, cache, and serve AI models efficiently

### Key Features

- ✅ Arabic and multilingual text support (BGE-M3)
- ✅ Fast embedding generation with batching
- ✅ Cosine similarity calculation for semantic search
- ✅ Optional cross-encoder reranking
- ✅ Model caching and optimization
- ✅ Auto-generated OpenAPI documentation
- ✅ Health checks and monitoring
- ✅ Stateless, horizontally scalable design

### Design Principles

- **Stateless**: No database dependency, pure computation
- **Fast**: Batch processing, model caching, async operations
- **Simple**: Clear API contracts for easy integration
- **Robust**: Error handling, input validation, logging
- **Scalable**: Can run multiple instances behind a load balancer

---

## Tech Stack

### Core Dependencies

**`requirements.txt`**:

```txt
# FastAPI and Server
fastapi==0.115.6
uvicorn[standard]==0.34.0
pydantic==2.10.6
pydantic-settings==2.7.1

# ML and NLP
sentence-transformers==3.3.1
transformers==4.48.0
torch==2.5.1
numpy==2.2.1
scikit-learn==1.6.1

# Utilities
python-dotenv==1.0.1
python-multipart==0.0.20
aiofiles==24.1.0

# Logging and Monitoring
loguru==0.7.3

# Testing
pytest==8.3.5
pytest-asyncio==0.25.2
httpx==0.28.1
```

**`requirements-dev.txt`**:

```txt
black==24.10.0
ruff==0.9.1
mypy==1.14.1
pre-commit==4.0.1
```

---

## Project Structure

```
ai_service/
├── app/
│   ├── main.py                     # FastAPI application
│   ├── config.py                   # Configuration and settings
│   ├── api/
│   │   ├── __init__.py
│   │   ├── deps.py                 # Dependency injection
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── health.py           # Health check endpoints
│   │   │   ├── embeddings.py       # Embedding endpoints
│   │   │   ├── similarity.py       # Similarity endpoints
│   │   │   └── rerank.py           # Reranking endpoints (optional)
│   │   └── schemas/
│   │       ├── __init__.py
│   │       ├── requests.py         # Request models
│   │       └── responses.py        # Response models
│   ├── core/
│   │   ├── __init__.py
│   │   ├── models.py               # Model loader and manager
│   │   ├── embeddings.py           # Embedding generation logic
│   │   ├── similarity.py           # Similarity calculation
│   │   └── reranker.py             # Reranking logic (optional)
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── text_processing.py     # Text preprocessing
│   │   ├── arabic_utils.py        # Arabic-specific utilities
│   │   └── logger.py              # Logging configuration
│   └── tests/
│       ├── __init__.py
│       ├── test_embeddings.py
│       ├── test_similarity.py
│       └── test_api.py
├── models/                         # Downloaded model cache (gitignored)
├── .env.example
├── .env
├── .gitignore
├── requirements.txt
├── requirements-dev.txt
├── Dockerfile
├── docker-compose.yml
├── pytest.ini
└── README.md
```

---

## Phase 1: Foundation Setup

### Step 1.1: Initialize Python Project

**Goal**: Set up the Python environment and project structure.

```bash
# Create project directory
mkdir ai_service && cd ai_service

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate
# Or (Linux/Mac)
source venv/bin/activate

# Create directory structure
mkdir -p app/{api/{routes,schemas},core,utils,tests}
touch app/__init__.py app/main.py app/config.py
```

### Step 1.2: Environment Configuration

**File**: `.env.example`

```env
# Application
APP_NAME="Legal AI Service"
APP_VERSION="1.0.0"
ENV=development
DEBUG=true
LOG_LEVEL=INFO

# Server
HOST=0.0.0.0
PORT=8000
WORKERS=1

# Model Configuration
MODEL_NAME=BAAI/bge-m3
MODEL_CACHE_DIR=./models
EMBEDDING_DIMENSION=1024
MAX_SEQ_LENGTH=8192

# Optional Reranker
USE_RERANKER=false
RERANKER_MODEL=BAAI/bge-reranker-v2-m3

# Performance
BATCH_SIZE=32
MAX_WORKERS=4

# CORS
CORS_ORIGINS=http://localhost:3000,http://localhost:5173
```

**File**: `app/config.py`

```python
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List


class Settings(BaseSettings):
    # Application
    app_name: str = Field(default="Legal AI Service", env="APP_NAME")
    app_version: str = Field(default="1.0.0", env="APP_VERSION")
    env: str = Field(default="development", env="ENV")
    debug: bool = Field(default=True, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")

    # Server
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    workers: int = Field(default=1, env="WORKERS")

    # Model Configuration
    model_name: str = Field(default="BAAI/bge-m3", env="MODEL_NAME")
    model_cache_dir: str = Field(default="./models", env="MODEL_CACHE_DIR")
    embedding_dimension: int = Field(default=1024, env="EMBEDDING_DIMENSION")
    max_seq_length: int = Field(default=8192, env="MAX_SEQ_LENGTH")

    # Optional Reranker
    use_reranker: bool = Field(default=False, env="USE_RERANKER")
    reranker_model: str = Field(default="BAAI/bge-reranker-v2-m3", env="RERANKER_MODEL")

    # Performance
    batch_size: int = Field(default=32, env="BATCH_SIZE")
    max_workers: int = Field(default=4, env="MAX_WORKERS")

    # CORS
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:5173"],
        env="CORS_ORIGINS"
    )

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
```

### Step 1.3: Logger Setup

**File**: `app/utils/logger.py`

```python
import sys
from loguru import logger
from app.config import settings

# Remove default handler
logger.remove()

# Add custom handler
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level=settings.log_level,
    colorize=True,
)

# Add file handler for production
if settings.env == "production":
    logger.add(
        "logs/ai_service_{time}.log",
        rotation="500 MB",
        retention="10 days",
        level="INFO",
    )

__all__ = ["logger"]
```

### Step 1.4: Dependencies Installation

```bash
# Install all dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Verify installation
python -c "import fastapi; import torch; import sentence_transformers; print('All dependencies installed!')"
```

---

## Phase 2: Model Loading & Management

### Step 2.1: Model Manager

**File**: `app/core/models.py`

```python
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import Optional
from app.config import settings
from app.utils.logger import logger


class ModelManager:
    """Singleton class to manage AI models."""

    _instance: Optional['ModelManager'] = None
    _embedding_model: Optional[SentenceTransformer] = None
    _reranker_model: Optional[AutoModelForSequenceClassification] = None
    _reranker_tokenizer: Optional[AutoTokenizer] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load_embedding_model(self) -> SentenceTransformer:
        """Load the embedding model (BGE-M3 or alternative)."""
        if self._embedding_model is None:
            logger.info(f"Loading embedding model: {settings.model_name}")

            self._embedding_model = SentenceTransformer(
                settings.model_name,
                cache_folder=settings.model_cache_dir,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )

            # Set max sequence length
            self._embedding_model.max_seq_length = settings.max_seq_length

            logger.info(f"Embedding model loaded successfully on {self._embedding_model.device}")
            logger.info(f"Model dimension: {self._embedding_model.get_sentence_embedding_dimension()}")

        return self._embedding_model

    def load_reranker_model(self):
        """Load the reranker model (optional)."""
        if not settings.use_reranker:
            return None, None

        if self._reranker_model is None:
            logger.info(f"Loading reranker model: {settings.reranker_model}")

            self._reranker_tokenizer = AutoTokenizer.from_pretrained(
                settings.reranker_model,
                cache_dir=settings.model_cache_dir
            )

            self._reranker_model = AutoModelForSequenceClassification.from_pretrained(
                settings.reranker_model,
                cache_dir=settings.model_cache_dir
            )

            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self._reranker_model.to(device)

            logger.info(f"Reranker model loaded successfully on {device}")

        return self._reranker_model, self._reranker_tokenizer

    @property
    def embedding_model(self) -> SentenceTransformer:
        """Get or load the embedding model."""
        if self._embedding_model is None:
            return self.load_embedding_model()
        return self._embedding_model

    @property
    def reranker(self):
        """Get or load the reranker model."""
        if self._reranker_model is None and settings.use_reranker:
            return self.load_reranker_model()
        return self._reranker_model, self._reranker_tokenizer

    def get_model_info(self) -> dict:
        """Get information about loaded models."""
        return {
            "embedding_model": {
                "name": settings.model_name,
                "dimension": self.embedding_model.get_sentence_embedding_dimension(),
                "max_seq_length": self.embedding_model.max_seq_length,
                "device": str(self.embedding_model.device),
            },
            "reranker_enabled": settings.use_reranker,
            "reranker_model": settings.reranker_model if settings.use_reranker else None,
        }


# Global instance
model_manager = ModelManager()
```

---

## Phase 3: Text Processing Utilities

### Step 3.1: Arabic Text Preprocessing

**File**: `app/utils/arabic_utils.py`

```python
import re
from typing import List


class ArabicTextProcessor:
    """Utilities for processing Arabic legal text."""

    # Arabic diacritics pattern
    DIACRITICS_PATTERN = re.compile(r'[\u064B-\u0652\u0670\u0640]')

    # Arabic punctuation
    ARABIC_PUNCTUATION = '،؛؟'

    def __init__(self):
        pass

    def remove_diacritics(self, text: str) -> str:
        """Remove Arabic diacritics (tashkeel) for better matching."""
        return self.DIACRITICS_PATTERN.sub('', text)

    def normalize_arabic(self, text: str) -> str:
        """Normalize Arabic characters."""
        # Normalize Alef variants
        text = text.replace('إ', 'ا')
        text = text.replace('أ', 'ا')
        text = text.replace('آ', 'ا')
        text = text.replace('ٱ', 'ا')

        # Normalize Taa Marbuta
        text = text.replace('ة', 'ه')

        # Normalize Yaa variants
        text = text.replace('ى', 'ي')

        return text

    def clean_text(self, text: str) -> str:
        """Clean and normalize Arabic legal text."""
        # Remove diacritics
        text = self.remove_diacritics(text)

        # Normalize Arabic letters
        text = self.normalize_arabic(text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Remove URLs
        text = re.sub(r'http\S+|www\.\S+', '', text)

        return text

    def extract_legal_entities(self, text: str) -> dict:
        """Extract legal entities like case numbers, article numbers."""
        entities = {
            'case_numbers': [],
            'article_numbers': [],
            'law_references': []
        }

        # Saudi case number pattern: "القضية رقم 12345/1445"
        case_pattern = r'القضية\s+رقم\s+(\d+/\d+)'
        entities['case_numbers'] = re.findall(case_pattern, text)

        # Article pattern: "المادة 15"
        article_pattern = r'المادة\s+(\d+)'
        entities['article_numbers'] = re.findall(article_pattern, text)

        # Law reference pattern: "نظام العمل", "نظام الإجراءات الجزائية"
        law_pattern = r'نظام\s+([^\n.،؛]{5,50})'
        entities['law_references'] = re.findall(law_pattern, text)

        return entities


# Global instance
arabic_processor = ArabicTextProcessor()
```

### Step 3.2: General Text Processing

**File**: `app/utils/text_processing.py`

```python
from typing import List, Union
from app.utils.arabic_utils import arabic_processor
from app.utils.logger import logger


def preprocess_text(text: str, is_arabic: bool = True) -> str:
    """Preprocess text before embedding."""
    if not text or not text.strip():
        return ""

    # Apply Arabic-specific preprocessing if needed
    if is_arabic:
        text = arabic_processor.clean_text(text)

    # General cleaning
    text = text.strip()

    return text


def preprocess_batch(texts: List[str], is_arabic: bool = True) -> List[str]:
    """Preprocess a batch of texts."""
    return [preprocess_text(text, is_arabic) for text in texts]


def truncate_text(text: str, max_length: int = 8192) -> str:
    """Truncate text to maximum length (in characters)."""
    if len(text) > max_length:
        logger.warning(f"Text truncated from {len(text)} to {max_length} characters")
        return text[:max_length]
    return text


def combine_case_text(title: str, description: str = "") -> str:
    """Combine case title and description for embedding."""
    parts = [title]
    if description:
        parts.append(description)
    return "\n\n".join(parts)
```

---

## Phase 4: Embedding Generation

### Step 4.1: Embedding Service

**File**: `app/core/embeddings.py`

```python
from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer
from app.utils.logger import logger
from app.utils.text_processing import preprocess_batch


class EmbeddingService:
    """Service for generating embeddings from text."""

    def __init__(self, model: SentenceTransformer):
        self.model = model
        self.dimension = model.get_sentence_embedding_dimension()

    def encode_single(
        self,
        text: str,
        normalize: bool = True,
        preprocess: bool = True
    ) -> np.ndarray:
        """Generate embedding for a single text."""
        if preprocess:
            text = preprocess_batch([text])[0]

        if not text:
            logger.warning("Empty text provided for embedding")
            return np.zeros(self.dimension)

        embedding = self.model.encode(
            text,
            normalize_embeddings=normalize,
            show_progress_bar=False,
            convert_to_numpy=True
        )

        return embedding

    def encode_batch(
        self,
        texts: List[str],
        normalize: bool = True,
        preprocess: bool = True,
        batch_size: int = 32
    ) -> np.ndarray:
        """Generate embeddings for multiple texts."""
        if preprocess:
            texts = preprocess_batch(texts)

        # Filter empty texts
        valid_texts = [t if t else " " for t in texts]

        embeddings = self.model.encode(
            valid_texts,
            normalize_embeddings=normalize,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True
        )

        logger.info(f"Generated embeddings for {len(texts)} texts")

        return embeddings

    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self.dimension
```

---

## Phase 5: Similarity Calculation

### Step 5.1: Similarity Service

**File**: `app/core/similarity.py`

```python
from typing import List, Dict, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from app.utils.logger import logger


class SimilarityService:
    """Service for calculating semantic similarity."""

    @staticmethod
    def cosine_similarity_single(
        query_embedding: np.ndarray,
        candidate_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Calculate cosine similarity between one query and multiple candidates.

        Args:
            query_embedding: Shape (dimension,)
            candidate_embeddings: Shape (n_candidates, dimension)

        Returns:
            similarities: Shape (n_candidates,)
        """
        # Reshape query to (1, dimension) if needed
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        similarities = cosine_similarity(query_embedding, candidate_embeddings)[0]
        return similarities

    @staticmethod
    def find_top_k(
        query_embedding: np.ndarray,
        candidate_embeddings: List[Dict[str, any]],
        top_k: int = 10,
        threshold: float = 0.0
    ) -> List[Dict]:
        """
        Find top-K most similar candidates.

        Args:
            query_embedding: Query vector
            candidate_embeddings: List of dicts with 'id', 'embedding', and metadata
            top_k: Number of results to return
            threshold: Minimum similarity score

        Returns:
            List of top-K matches with scores
        """
        if not candidate_embeddings:
            return []

        # Extract embeddings and metadata
        embeddings_matrix = np.array([
            item['embedding'] for item in candidate_embeddings
        ])

        # Calculate similarities
        similarities = SimilarityService.cosine_similarity_single(
            query_embedding,
            embeddings_matrix
        )

        # Create results with scores
        results = []
        for idx, similarity in enumerate(similarities):
            if similarity >= threshold:
                item = candidate_embeddings[idx].copy()
                item['similarity_score'] = float(similarity)
                # Remove embedding from response (too large)
                item.pop('embedding', None)
                results.append(item)

        # Sort by score descending
        results.sort(key=lambda x: x['similarity_score'], reverse=True)

        # Return top-K
        return results[:top_k]

    @staticmethod
    def batch_cosine_similarity(
        embeddings_a: np.ndarray,
        embeddings_b: np.ndarray
    ) -> np.ndarray:
        """
        Calculate pairwise cosine similarities.

        Args:
            embeddings_a: Shape (n, dimension)
            embeddings_b: Shape (m, dimension)

        Returns:
            similarities: Shape (n, m)
        """
        return cosine_similarity(embeddings_a, embeddings_b)
```

---

## Phase 6: API Schemas

### Step 6.1: Request Schemas

**File**: `app/api/schemas/requests.py`

```python
from pydantic import BaseModel, Field
from typing import List, Optional


class EmbedRequest(BaseModel):
    """Request schema for embedding generation."""
    texts: List[str] = Field(
        ...,
        description="List of texts to embed",
        min_length=1,
        max_length=100
    )
    normalize: bool = Field(
        default=True,
        description="Whether to normalize embeddings"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "texts": [
                    "قضية تجارية تتعلق بعقد بيع",
                    "نزاع عمالي حول فصل تعسفي"
                ],
                "normalize": True
            }
        }


class RegulationCandidate(BaseModel):
    """Single regulation candidate with embedding."""
    id: int = Field(..., description="Regulation ID")
    title: str = Field(..., description="Regulation title")
    embedding: List[float] = Field(..., description="Pre-computed embedding vector")
    category: Optional[str] = Field(None, description="Regulation category")


class FindRelatedRequest(BaseModel):
    """Request schema for finding related regulations."""
    case_text: str = Field(
        ...,
        description="Case text (title + description)",
        min_length=10
    )
    regulation_candidates: List[RegulationCandidate] = Field(
        ...,
        description="List of regulation candidates with embeddings",
        min_length=1
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of top results to return"
    )
    threshold: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score threshold"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "case_text": "قضية تجارية حول نزاع في عقد توريد بضائع",
                "regulation_candidates": [
                    {
                        "id": 1,
                        "title": "نظام التجارة السعودي",
                        "embedding": [0.1, 0.2, 0.3],  # ... 1024 dimensions
                        "category": "commercial_law"
                    }
                ],
                "top_k": 5,
                "threshold": 0.3
            }
        }


class RerankRequest(BaseModel):
    """Request schema for reranking."""
    query: str = Field(..., description="Query text")
    candidates: List[Dict] = Field(
        ...,
        description="Candidate documents with text",
        min_length=1
    )
    top_k: int = Field(default=5, ge=1, le=50)

    class Config:
        json_schema_extra = {
            "example": {
                "query": "قضية عمالية عن فصل تعسفي",
                "candidates": [
                    {"id": 1, "text": "نظام العمل - الفصل التعسفي"},
                    {"id": 2, "text": "نظام الإجراءات العمالية"}
                ],
                "top_k": 5
            }
        }
```

### Step 6.2: Response Schemas

**File**: `app/api/schemas/responses.py`

```python
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class EmbedResponse(BaseModel):
    """Response schema for embedding generation."""
    embeddings: List[List[float]] = Field(
        ...,
        description="List of embedding vectors"
    )
    dimension: int = Field(..., description="Embedding dimension")
    count: int = Field(..., description="Number of embeddings generated")


class SimilarityMatch(BaseModel):
    """Single similarity match result."""
    regulation_id: int = Field(..., alias="id")
    title: str
    similarity_score: float = Field(..., ge=0.0, le=1.0)
    category: Optional[str] = None

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "regulation_id": 1,
                "title": "نظام العمل السعودي",
                "similarity_score": 0.8542,
                "category": "labor_law"
            }
        }


class FindRelatedResponse(BaseModel):
    """Response schema for finding related regulations."""
    related_regulations: List[SimilarityMatch] = Field(
        ...,
        description="Top-K most similar regulations"
    )
    query_length: int = Field(..., description="Length of query text")
    candidates_count: int = Field(..., description="Total candidates evaluated")

    class Config:
        json_schema_extra = {
            "example": {
                "related_regulations": [
                    {
                        "regulation_id": 1,
                        "title": "نظام العمل",
                        "similarity_score": 0.89,
                        "category": "labor_law"
                    }
                ],
                "query_length": 150,
                "candidates_count": 50
            }
        }


class RerankResponse(BaseModel):
    """Response schema for reranking."""
    reranked_results: List[Dict[str, Any]]
    top_k: int


class ModelInfoResponse(BaseModel):
    """Model information response."""
    embedding_model: Dict[str, Any]
    reranker_enabled: bool
    reranker_model: Optional[str]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    model_loaded: bool
    device: str
    version: str
```

---

## Phase 7: Core API Endpoints

### Step 7.1: Health Check Endpoint

**File**: `app/api/routes/health.py`

```python
from fastapi import APIRouter, Depends
from app.api.schemas.responses import HealthResponse, ModelInfoResponse
from app.core.models import model_manager
from app.config import settings
import torch

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/", response_model=HealthResponse)
async def health_check():
    """Basic health check endpoint."""
    model_loaded = model_manager._embedding_model is not None
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    return HealthResponse(
        status="healthy",
        model_loaded=model_loaded,
        device=device,
        version=settings.app_version
    )


@router.get("/model-info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get information about loaded models."""
    return model_manager.get_model_info()
```

### Step 7.2: Embedding Endpoint

**File**: `app/api/routes/embeddings.py`

```python
from fastapi import APIRouter, HTTPException, Depends
from app.api.schemas.requests import EmbedRequest
from app.api.schemas.responses import EmbedResponse
from app.core.models import model_manager
from app.core.embeddings import EmbeddingService
from app.utils.logger import logger

router = APIRouter(prefix="/embed", tags=["embeddings"])


@router.post("/", response_model=EmbedResponse)
async def generate_embeddings(request: EmbedRequest):
    """
    Generate embeddings for a list of texts.

    This endpoint uses BGE-M3 (or configured model) to convert Arabic/English
    legal texts into dense vector representations for semantic similarity.
    """
    try:
        # Get model
        model = model_manager.embedding_model

        # Create embedding service
        embedding_service = EmbeddingService(model)

        # Generate embeddings
        embeddings = embedding_service.encode_batch(
            texts=request.texts,
            normalize=request.normalize,
            preprocess=True
        )

        # Convert to list for JSON serialization
        embeddings_list = embeddings.tolist()

        logger.info(f"Generated {len(embeddings_list)} embeddings")

        return EmbedResponse(
            embeddings=embeddings_list,
            dimension=embedding_service.dimension,
            count=len(embeddings_list)
        )

    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")
```

### Step 7.3: Similarity Endpoint

**File**: `app/api/routes/similarity.py`

```python
from fastapi import APIRouter, HTTPException
from app.api.schemas.requests import FindRelatedRequest
from app.api.schemas.responses import FindRelatedResponse, SimilarityMatch
from app.core.models import model_manager
from app.core.embeddings import EmbeddingService
from app.core.similarity import SimilarityService
from app.utils.logger import logger
from app.utils.text_processing import preprocess_text
import numpy as np

router = APIRouter(prefix="/similarity", tags=["similarity"])


@router.post("/find-related", response_model=FindRelatedResponse)
async def find_related_regulations(request: FindRelatedRequest):
    """
    Find most similar regulations to a case text using cosine similarity.

    This is the core endpoint for AI-powered case-regulation linking.
    It compares the case text embedding with pre-computed regulation embeddings
    and returns the top-K most semantically similar regulations.
    """
    try:
        # Get model and services
        model = model_manager.embedding_model
        embedding_service = EmbeddingService(model)

        # Preprocess case text
        case_text = preprocess_text(request.case_text)

        # Generate embedding for case
        logger.info("Generating embedding for case text")
        case_embedding = embedding_service.encode_single(
            text=case_text,
            normalize=True,
            preprocess=False  # Already preprocessed
        )

        # Prepare candidates with embeddings
        candidates = []
        for reg in request.regulation_candidates:
            candidates.append({
                'id': reg.id,
                'title': reg.title,
                'embedding': np.array(reg.embedding),
                'category': reg.category
            })

        # Find top-K similar regulations
        logger.info(f"Calculating similarity with {len(candidates)} candidates")
        results = SimilarityService.find_top_k(
            query_embedding=case_embedding,
            candidate_embeddings=candidates,
            top_k=request.top_k,
            threshold=request.threshold
        )

        # Convert to response schema
        matches = [
            SimilarityMatch(
                regulation_id=r['id'],
                title=r['title'],
                similarity_score=r['similarity_score'],
                category=r.get('category')
            )
            for r in results
        ]

        logger.info(f"Found {len(matches)} related regulations (threshold: {request.threshold})")

        return FindRelatedResponse(
            related_regulations=matches,
            query_length=len(case_text),
            candidates_count=len(candidates)
        )

    except Exception as e:
        logger.error(f"Error finding related regulations: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Similarity calculation failed: {str(e)}"
        )


@router.post("/batch-similarity")
async def batch_similarity(
    queries: List[str],
    documents: List[str]
):
    """
    Calculate pairwise similarity between queries and documents.

    Useful for bulk operations or analysis.
    """
    try:
        model = model_manager.embedding_model
        embedding_service = EmbeddingService(model)

        # Generate embeddings
        query_embeddings = embedding_service.encode_batch(queries)
        doc_embeddings = embedding_service.encode_batch(documents)

        # Calculate similarities
        similarities = SimilarityService.batch_cosine_similarity(
            query_embeddings,
            doc_embeddings
        )

        return {
            "similarities": similarities.tolist(),
            "shape": list(similarities.shape)
        }

    except Exception as e:
        logger.error(f"Error in batch similarity: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
```

### Step 7.4: Optional Reranking Endpoint

**File**: `app/api/routes/rerank.py`

```python
from fastapi import APIRouter, HTTPException
from app.api.schemas.requests import RerankRequest
from app.api.schemas.responses import RerankResponse
from app.core.models import model_manager
from app.core.reranker import RerankerService
from app.utils.logger import logger
from app.config import settings

router = APIRouter(prefix="/rerank", tags=["reranking"])


@router.post("/", response_model=RerankResponse)
async def rerank_candidates(request: RerankRequest):
    """
    Rerank candidates using cross-encoder for higher precision.

    Use this after initial retrieval to improve top-K accuracy.
    Cross-encoders score query-document pairs directly for better relevance.
    """
    if not settings.use_reranker:
        raise HTTPException(
            status_code=501,
            detail="Reranker is not enabled. Set USE_RERANKER=true in config."
        )

    try:
        # Get reranker
        reranker_model, reranker_tokenizer = model_manager.reranker

        if reranker_model is None:
            raise HTTPException(status_code=500, detail="Reranker model not loaded")

        # Create reranker service
        reranker_service = RerankerService(reranker_model, reranker_tokenizer)

        # Rerank
        results = reranker_service.rerank(
            query=request.query,
            candidates=request.candidates,
            top_k=request.top_k
        )

        logger.info(f"Reranked {len(request.candidates)} candidates, returning top {request.top_k}")

        return RerankResponse(
            reranked_results=results,
            top_k=request.top_k
        )

    except Exception as e:
        logger.error(f"Error reranking: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
```

### Step 7.5: Reranker Service (Optional)

**File**: `app/core/reranker.py`

```python
from typing import List, Dict, Any
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from app.utils.logger import logger


class RerankerService:
    """Service for reranking using cross-encoder models."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Rerank candidates using cross-encoder.

        Args:
            query: Query text
            candidates: List of dicts with 'id', 'text', and optional metadata
            top_k: Number of top results to return

        Returns:
            Reranked list of candidates with scores
        """
        if not candidates:
            return []

        # Prepare pairs for scoring
        pairs = [(query, candidate.get('text', candidate.get('title', '')))
                 for candidate in candidates]

        # Tokenize
        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        ).to(self.device)

        # Get scores
        with torch.no_grad():
            scores = self.model(**inputs).logits.squeeze(-1)

        # Combine with candidates
        results = []
        for idx, candidate in enumerate(candidates):
            result = candidate.copy()
            result['rerank_score'] = float(scores[idx].cpu().numpy())
            results.append(result)

        # Sort by score
        results.sort(key=lambda x: x['rerank_score'], reverse=True)

        return results[:top_k]
```

---

## Phase 8: Main Application

### Step 8.1: FastAPI App

**File**: `app/main.py`

```python
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import time

from app.config import settings
from app.core.models import model_manager
from app.api.routes import health, embeddings, similarity, rerank
from app.utils.logger import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan events for the FastAPI app.
    Load models on startup, cleanup on shutdown.
    """
    # Startup
    logger.info("Starting AI Service...")
    logger.info(f"Environment: {settings.env}")
    logger.info(f"Model: {settings.model_name}")

    # Load embedding model
    start_time = time.time()
    model_manager.load_embedding_model()
    load_time = time.time() - start_time
    logger.info(f"Embedding model loaded in {load_time:.2f}s")

    # Load reranker if enabled
    if settings.use_reranker:
        start_time = time.time()
        model_manager.load_reranker_model()
        load_time = time.time() - start_time
        logger.info(f"Reranker model loaded in {load_time:.2f}s")

    logger.info("AI Service ready!")

    yield

    # Shutdown
    logger.info("Shutting down AI Service...")


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    description="AI microservice for semantic similarity in legal case management",
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests."""
    start_time = time.time()

    response = await call_next(request)

    process_time = time.time() - start_time
    logger.info(
        f"{request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Time: {process_time:.3f}s"
    )

    return response


# Exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Global error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)}
    )


# Include routers
app.include_router(health.router)
app.include_router(embeddings.router)
app.include_router(similarity.router)

if settings.use_reranker:
    app.include_router(rerank.router)


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with service information."""
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "docs": "/docs",
        "model": settings.model_name
    }
```

---

## Phase 9: Complete Implementation Examples

### Step 9.1: Complete Embedding Service

**File**: `app/core/embeddings.py` (complete version)

```python
from typing import List, Union, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from app.utils.logger import logger
from app.utils.text_processing import preprocess_batch, truncate_text
from app.config import settings


class EmbeddingService:
    """Service for generating embeddings from text."""

    def __init__(self, model: SentenceTransformer):
        self.model = model
        self.dimension = model.get_sentence_embedding_dimension()
        self.max_length = model.max_seq_length

    def encode_single(
        self,
        text: str,
        normalize: bool = True,
        preprocess: bool = True
    ) -> np.ndarray:
        """Generate embedding for a single text."""
        if preprocess:
            text = preprocess_batch([text])[0]

        # Truncate if needed
        text = truncate_text(text, self.max_length)

        if not text:
            logger.warning("Empty text provided for embedding")
            return np.zeros(self.dimension, dtype=np.float32)

        try:
            embedding = self.model.encode(
                text,
                normalize_embeddings=normalize,
                show_progress_bar=False,
                convert_to_numpy=True,
                batch_size=1
            )

            return embedding.astype(np.float32)

        except Exception as e:
            logger.error(f"Error encoding text: {str(e)}")
            raise

    def encode_batch(
        self,
        texts: List[str],
        normalize: bool = True,
        preprocess: bool = True,
        batch_size: Optional[int] = None
    ) -> np.ndarray:
        """Generate embeddings for multiple texts."""
        if not texts:
            return np.array([])

        if preprocess:
            texts = preprocess_batch(texts)

        # Truncate texts
        texts = [truncate_text(t, self.max_length) for t in texts]

        # Replace empty texts with space
        valid_texts = [t if t.strip() else " " for t in texts]

        try:
            embeddings = self.model.encode(
                valid_texts,
                normalize_embeddings=normalize,
                batch_size=batch_size or settings.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True
            )

            logger.info(
                f"Generated embeddings for {len(texts)} texts "
                f"(shape: {embeddings.shape})"
            )

            return embeddings.astype(np.float32)

        except Exception as e:
            logger.error(f"Error encoding batch: {str(e)}")
            raise

    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self.dimension

    def get_model_info(self) -> dict:
        """Get model information."""
        return {
            "model_name": self.model.get_config_dict().get('_name_or_path', 'unknown'),
            "dimension": self.dimension,
            "max_length": self.max_length,
            "device": str(self.model.device)
        }
```

### Step 9.2: Complete Similarity Service

**File**: `app/core/similarity.py` (complete version)

```python
from typing import List, Dict, Optional, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from app.utils.logger import logger


class SimilarityService:
    """Service for calculating semantic similarity."""

    @staticmethod
    def cosine_similarity_single(
        query_embedding: np.ndarray,
        candidate_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Calculate cosine similarity between one query and multiple candidates.

        Formula: cos(θ) = (A · B) / (||A|| ||B||)

        Args:
            query_embedding: Shape (dimension,) or (1, dimension)
            candidate_embeddings: Shape (n_candidates, dimension)

        Returns:
            similarities: Shape (n_candidates,) with values in [-1, 1]
        """
        # Reshape query to (1, dimension) if needed
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Calculate cosine similarity
        similarities = cosine_similarity(query_embedding, candidate_embeddings)[0]

        return similarities

    @staticmethod
    def find_top_k(
        query_embedding: np.ndarray,
        candidate_embeddings: List[Dict[str, any]],
        top_k: int = 10,
        threshold: float = 0.0
    ) -> List[Dict]:
        """
        Find top-K most similar candidates using cosine similarity.

        Args:
            query_embedding: Query vector (normalized)
            candidate_embeddings: List of dicts with 'id', 'embedding', and metadata
            top_k: Number of results to return
            threshold: Minimum similarity score (0.0 to 1.0)

        Returns:
            List of top-K matches with scores, sorted descending
        """
        if not candidate_embeddings:
            logger.warning("No candidates provided for similarity search")
            return []

        # Extract embeddings and metadata
        try:
            embeddings_matrix = np.array([
                item['embedding'] for item in candidate_embeddings
            ], dtype=np.float32)
        except (KeyError, ValueError) as e:
            logger.error(f"Error extracting embeddings: {str(e)}")
            raise ValueError("Invalid candidate format: missing or invalid 'embedding' field")

        # Calculate similarities
        similarities = SimilarityService.cosine_similarity_single(
            query_embedding,
            embeddings_matrix
        )

        # Create results with scores
        results = []
        for idx, similarity in enumerate(similarities):
            if similarity >= threshold:
                item = candidate_embeddings[idx].copy()
                item['similarity_score'] = float(similarity)
                # Remove embedding from response (too large for API response)
                item.pop('embedding', None)
                results.append(item)

        # Sort by score descending
        results.sort(key=lambda x: x['similarity_score'], reverse=True)

        # Log statistics
        logger.info(
            f"Similarity search: {len(results)} matches above threshold {threshold:.2f} "
            f"(from {len(candidate_embeddings)} candidates)"
        )

        if results:
            logger.info(
                f"Top score: {results[0]['similarity_score']:.4f}, "
                f"Lowest score in top-K: {results[min(top_k-1, len(results)-1)]['similarity_score']:.4f}"
            )

        # Return top-K
        return results[:top_k]

    @staticmethod
    def batch_cosine_similarity(
        embeddings_a: np.ndarray,
        embeddings_b: np.ndarray
    ) -> np.ndarray:
        """
        Calculate pairwise cosine similarities between two sets of embeddings.

        Args:
            embeddings_a: Shape (n, dimension)
            embeddings_b: Shape (m, dimension)

        Returns:
            similarities: Shape (n, m) with values in [-1, 1]
        """
        return cosine_similarity(embeddings_a, embeddings_b)

    @staticmethod
    def filter_by_threshold(
        results: List[Dict],
        threshold: float
    ) -> List[Dict]:
        """Filter results by minimum similarity threshold."""
        return [r for r in results if r.get('similarity_score', 0) >= threshold]
```

---

## Phase 10: Docker & Deployment

### Step 10.1: Dockerfile

**File**: `Dockerfile`

```dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY ./app ./app
COPY .env.example .env

# Create models directory
RUN mkdir -p /app/models

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
```

### Step 10.2: Docker Compose (for development)

**File**: `docker-compose.yml`

```yaml
version: "3.8"

services:
  ai-service:
    build: .
    container_name: legal-ai-service
    ports:
      - "8000:8000"
    environment:
      - ENV=development
      - DEBUG=true
      - MODEL_NAME=BAAI/bge-m3
      - MODEL_CACHE_DIR=/app/models
      - USE_RERANKER=false
    volumes:
      - ./app:/app/app
      - ./models:/app/models # Persist downloaded models
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: "2"
          memory: 4G
        reservations:
          cpus: "1"
          memory: 2G
```

### Step 10.3: Startup Script

**File**: `run.sh` (Linux/Mac) or `run.ps1` (Windows)

**`run.sh`**:

```bash
#!/bin/bash

# Activate virtual environment
source venv/bin/activate

# Run with uvicorn
uvicorn app.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --reload \
    --log-level info
```

**`run.ps1`**:

```powershell
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Run with uvicorn
uvicorn app.main:app `
    --host 0.0.0.0 `
    --port 8000 `
    --reload `
    --log-level info
```

---

## Phase 11: Testing

### Step 11.1: Test Configuration

**File**: `pytest.ini`

```ini
[pytest]
testpaths = app/tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
asyncio_mode = auto
```

### Step 11.2: Test Embeddings

**File**: `app/tests/test_embeddings.py`

```python
import pytest
import numpy as np
from app.core.models import model_manager
from app.core.embeddings import EmbeddingService


@pytest.fixture
def embedding_service():
    """Fixture for embedding service."""
    model = model_manager.embedding_model
    return EmbeddingService(model)


def test_embed_single_text(embedding_service):
    """Test single text embedding."""
    text = "قضية تجارية حول نزاع في العقد"

    embedding = embedding_service.encode_single(text)

    assert isinstance(embedding, np.ndarray)
    assert embedding.shape[0] == embedding_service.dimension
    assert not np.isnan(embedding).any()


def test_embed_batch(embedding_service):
    """Test batch embedding generation."""
    texts = [
        "قضية عمالية عن فصل تعسفي",
        "نزاع تجاري بشأن عقد توريد",
        "قضية جنائية - سرقة"
    ]

    embeddings = embedding_service.encode_batch(texts)

    assert embeddings.shape == (3, embedding_service.dimension)
    assert not np.isnan(embeddings).any()


def test_empty_text(embedding_service):
    """Test handling of empty text."""
    embedding = embedding_service.encode_single("")

    assert isinstance(embedding, np.ndarray)
    assert embedding.shape[0] == embedding_service.dimension
```

### Step 11.3: Test Similarity

**File**: `app/tests/test_similarity.py`

```python
import pytest
import numpy as np
from app.core.similarity import SimilarityService


def test_cosine_similarity():
    """Test cosine similarity calculation."""
    query = np.array([1.0, 0.0, 0.0])
    candidates = np.array([
        [1.0, 0.0, 0.0],  # Same direction - similarity = 1.0
        [0.0, 1.0, 0.0],  # Perpendicular - similarity = 0.0
        [-1.0, 0.0, 0.0]  # Opposite - similarity = -1.0
    ])

    similarities = SimilarityService.cosine_similarity_single(query, candidates)

    assert similarities.shape == (3,)
    assert np.isclose(similarities[0], 1.0)
    assert np.isclose(similarities[1], 0.0)
    assert np.isclose(similarities[2], -1.0)


def test_find_top_k():
    """Test finding top-K similar items."""
    query = np.array([1.0, 0.0, 0.0])

    candidates = [
        {'id': 1, 'title': 'Reg 1', 'embedding': [0.9, 0.1, 0.0]},
        {'id': 2, 'title': 'Reg 2', 'embedding': [0.5, 0.5, 0.0]},
        {'id': 3, 'title': 'Reg 3', 'embedding': [0.1, 0.9, 0.0]},
    ]

    results = SimilarityService.find_top_k(
        query_embedding=query,
        candidate_embeddings=candidates,
        top_k=2
    )

    assert len(results) == 2
    assert results[0]['id'] == 1  # Most similar
    assert results[0]['similarity_score'] > results[1]['similarity_score']
    assert 'embedding' not in results[0]  # Should be removed


def test_threshold_filtering():
    """Test similarity threshold filtering."""
    query = np.array([1.0, 0.0, 0.0])

    candidates = [
        {'id': 1, 'title': 'High sim', 'embedding': [0.95, 0.05, 0.0]},
        {'id': 2, 'title': 'Low sim', 'embedding': [0.1, 0.9, 0.0]},
    ]

    results = SimilarityService.find_top_k(
        query_embedding=query,
        candidate_embeddings=candidates,
        top_k=10,
        threshold=0.5
    )

    assert len(results) == 1  # Only one above threshold
    assert results[0]['id'] == 1
```

### Step 11.4: Test API Endpoints

**File**: `app/tests/test_api.py`

```python
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_check():
    """Test health check endpoint."""
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_loaded" in data


def test_root():
    """Test root endpoint."""
    response = client.get("/")

    assert response.status_code == 200
    data = response.json()
    assert "service" in data
    assert "version" in data


def test_embed_endpoint():
    """Test embedding generation endpoint."""
    response = client.post("/embed/", json={
        "texts": ["قضية تجارية", "نظام العمل"],
        "normalize": True
    })

    assert response.status_code == 200
    data = response.json()
    assert "embeddings" in data
    assert len(data["embeddings"]) == 2
    assert data["count"] == 2


def test_find_related_endpoint():
    """Test find related regulations endpoint."""
    # First, generate embeddings for regulations
    embed_response = client.post("/embed/", json={
        "texts": ["نظام العمل السعودي", "نظام التجارة"],
        "normalize": True
    })
    embeddings = embed_response.json()["embeddings"]

    # Then find related
    response = client.post("/similarity/find-related", json={
        "case_text": "قضية عمالية عن فصل تعسفي",
        "regulation_candidates": [
            {
                "id": 1,
                "title": "نظام العمل السعودي",
                "embedding": embeddings[0],
                "category": "labor_law"
            },
            {
                "id": 2,
                "title": "نظام التجارة",
                "embedding": embeddings[1],
                "category": "commercial_law"
            }
        ],
        "top_k": 2,
        "threshold": 0.0
    })

    assert response.status_code == 200
    data = response.json()
    assert "related_regulations" in data
    assert len(data["related_regulations"]) <= 2


def test_invalid_request():
    """Test validation error handling."""
    response = client.post("/embed/", json={
        "texts": [],  # Empty list - should fail
        "normalize": True
    })

    assert response.status_code == 422  # Validation error
```

---

## Phase 12: Advanced Features

### Step 12.1: Caching Service (Optional)

**File**: `app/utils/cache.py`

```python
from typing import Optional, Dict, Any
from functools import lru_cache
import hashlib
import json
import numpy as np
from app.utils.logger import logger


class EmbeddingCache:
    """Simple in-memory cache for embeddings."""

    def __init__(self, max_size: int = 1000):
        self.cache: Dict[str, np.ndarray] = {}
        self.max_size = max_size

    def _generate_key(self, text: str) -> str:
        """Generate cache key from text."""
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def get(self, text: str) -> Optional[np.ndarray]:
        """Get cached embedding."""
        key = self._generate_key(text)
        return self.cache.get(key)

    def set(self, text: str, embedding: np.ndarray):
        """Cache an embedding."""
        if len(self.cache) >= self.max_size:
            # Simple FIFO eviction
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            logger.debug(f"Cache full, evicted oldest entry")

        key = self._generate_key(text)
        self.cache[key] = embedding

    def clear(self):
        """Clear the cache."""
        self.cache.clear()
        logger.info("Cache cleared")

    def size(self) -> int:
        """Get current cache size."""
        return len(self.cache)


# Global cache instance
embedding_cache = EmbeddingCache(max_size=1000)
```

### Step 12.2: Batch Processing Utilities

**File**: `app/utils/batch_processing.py`

```python
from typing import List, TypeVar, Callable
import asyncio
from concurrent.futures import ThreadPoolExecutor
from app.config import settings
from app.utils.logger import logger

T = TypeVar('T')
R = TypeVar('R')


def chunk_list(lst: List[T], chunk_size: int) -> List[List[T]]:
    """Split a list into chunks."""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


async def process_in_batches(
    items: List[T],
    process_fn: Callable[[List[T]], R],
    batch_size: int = 32
) -> List[R]:
    """Process items in batches asynchronously."""
    chunks = chunk_list(items, batch_size)

    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=settings.max_workers) as executor:
        futures = [
            loop.run_in_executor(executor, process_fn, chunk)
            for chunk in chunks
        ]
        results = await asyncio.gather(*futures)

    return results
```

---

## Phase 13: Alternative Models Support

### Step 13.1: Model Configuration Templates

**File**: `app/config/model_configs.py`

```python
from typing import Dict, Any


# Model configurations for different use cases
MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    # Arabic + Multilingual (Recommended for Saudi legal texts)
    "bge-m3": {
        "name": "BAAI/bge-m3",
        "dimension": 1024,
        "max_length": 8192,
        "languages": ["ar", "en", "100+"],
        "description": "Multilingual model with strong Arabic support"
    },

    # Alternative: Multilingual E5
    "multilingual-e5-large": {
        "name": "intfloat/multilingual-e5-large",
        "dimension": 1024,
        "max_length": 512,
        "languages": ["ar", "en", "100+"],
        "description": "Strong multilingual model including Arabic"
    },

    # Lightweight option
    "all-minilm-l6-v2": {
        "name": "sentence-transformers/all-MiniLM-L6-v2",
        "dimension": 384,
        "max_length": 256,
        "languages": ["en"],
        "description": "Fast baseline model (English only)"
    },

    # Arabic-specific (if available)
    "arabic-sbert": {
        "name": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        "dimension": 768,
        "max_length": 128,
        "languages": ["ar", "en", "50+"],
        "description": "Multilingual model with Arabic support"
    }
}


# Reranker configurations
RERANKER_CONFIGS: Dict[str, Dict[str, Any]] = {
    "bge-reranker-v2-m3": {
        "name": "BAAI/bge-reranker-v2-m3",
        "description": "BGE reranker for multilingual texts"
    },

    "ms-marco-minilm": {
        "name": "cross-encoder/ms-marco-MiniLM-L6-v2",
        "description": "Fast reranker for English"
    }
}


def get_model_config(model_key: str) -> Dict[str, Any]:
    """Get configuration for a specific model."""
    return MODEL_CONFIGS.get(model_key, MODEL_CONFIGS["bge-m3"])
```

---

## Phase 14: Advanced Endpoints

### Step 14.1: Bulk Operations

**File**: `app/api/routes/bulk.py`

```python
from fastapi import APIRouter, HTTPException
from typing import List, Dict
from pydantic import BaseModel
from app.core.models import model_manager
from app.core.embeddings import EmbeddingService
from app.utils.logger import logger

router = APIRouter(prefix="/bulk", tags=["bulk-operations"])


class BulkEmbedRequest(BaseModel):
    """Bulk embedding request for regulations."""
    regulations: List[Dict[str, any]]


class BulkEmbedResponse(BaseModel):
    """Bulk embedding response."""
    results: List[Dict[str, any]]
    success_count: int
    error_count: int


@router.post("/embed-regulations", response_model=BulkEmbedResponse)
async def bulk_embed_regulations(request: BulkEmbedRequest):
    """
    Generate embeddings for multiple regulations in bulk.

    Useful for initial data ingestion or batch updates.
    """
    try:
        model = model_manager.embedding_model
        embedding_service = EmbeddingService(model)

        # Extract texts
        texts = [
            f"{reg.get('title', '')}\n\n{reg.get('content', '')}"
            for reg in request.regulations
        ]

        # Generate embeddings
        logger.info(f"Generating embeddings for {len(texts)} regulations")
        embeddings = embedding_service.encode_batch(texts)

        # Prepare results
        results = []
        success_count = 0
        error_count = 0

        for idx, reg in enumerate(request.regulations):
            try:
                results.append({
                    "id": reg.get("id"),
                    "title": reg.get("title"),
                    "embedding": embeddings[idx].tolist(),
                    "dimension": embedding_service.dimension
                })
                success_count += 1
            except Exception as e:
                logger.error(f"Error processing regulation {reg.get('id')}: {str(e)}")
                error_count += 1

        return BulkEmbedResponse(
            results=results,
            success_count=success_count,
            error_count=error_count
        )

    except Exception as e:
        logger.error(f"Bulk embedding error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
```

---

## Phase 15: Monitoring & Metrics

### Step 15.1: Simple Metrics Endpoint

**File**: `app/api/routes/metrics.py`

```python
from fastify import APIRouter
from pydantic import BaseModel
from typing import Dict
import psutil
import torch
from app.utils.logger import logger

router = APIRouter(prefix="/metrics", tags=["monitoring"])


class MetricsResponse(BaseModel):
    """System metrics response."""
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    gpu_available: bool
    gpu_memory_used_mb: Optional[float] = None


@router.get("/system", response_model=MetricsResponse)
async def get_system_metrics():
    """Get system resource usage metrics."""
    memory = psutil.virtual_memory()

    metrics = MetricsResponse(
        cpu_percent=psutil.cpu_percent(interval=1),
        memory_percent=memory.percent,
        memory_used_mb=memory.used / (1024 ** 2),
        gpu_available=torch.cuda.is_available()
    )

    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / (1024 ** 2)
        metrics.gpu_memory_used_mb = gpu_memory

    return metrics
```

---

## Implementation Checklist

### Week 1: Foundation & Setup

- [ ] Set up Python virtual environment
- [ ] Install dependencies (FastAPI, sentence-transformers, torch)
- [ ] Create project structure
- [ ] Configure environment variables
- [ ] Set up logging
- [ ] Create Docker configuration

### Week 2: Model Loading

- [ ] Implement model manager (singleton pattern)
- [ ] Load BGE-M3 embedding model
- [ ] Test model loading and inference
- [ ] Add model caching
- [ ] Verify GPU/CPU detection

### Week 3: Text Processing & Embeddings

- [ ] Implement Arabic text preprocessing
- [ ] Create embedding service
- [ ] Add batch processing support
- [ ] Test with Arabic and English texts
- [ ] Optimize for performance

### Week 4: Similarity & Core Logic

- [ ] Implement cosine similarity calculation
- [ ] Create find top-K function
- [ ] Add threshold filtering
- [ ] Test similarity accuracy
- [ ] Add edge case handling

### Week 5: API Development

- [ ] Create request/response schemas
- [ ] Implement health check endpoint
- [ ] Implement /embed endpoint
- [ ] Implement /find-related endpoint
- [ ] Add input validation
- [ ] Generate OpenAPI docs

### Week 6: Optional Features & Testing

- [ ] Add reranker model (optional)
- [ ] Implement reranking endpoint
- [ ] Write unit tests
- [ ] Write integration tests
- [ ] Test API with Postman/httpx
- [ ] Performance benchmarking

### Week 7: Deployment & Integration

- [ ] Build Docker image
- [ ] Test containerized deployment
- [ ] Integration testing with Fastify backend
- [ ] Document API usage
- [ ] Create deployment guide
- [ ] Performance tuning

---

## API Usage Examples

### Example 1: Generate Embeddings

**Request**:

```bash
curl -X POST "http://localhost:8000/embed/" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["قضية تجارية حول عقد توريد", "نظام التجارة السعودي"],
    "normalize": true
  }'
```

**Response**:

```json
{
  "embeddings": [
    [0.023, -0.145, 0.892, ...],  // 1024 dimensions
    [0.156, -0.089, 0.234, ...]
  ],
  "dimension": 1024,
  "count": 2
}
```

### Example 2: Find Related Regulations

**Request**:

```bash
curl -X POST "http://localhost:8000/similarity/find-related" \
  -H "Content-Type: application/json" \
  -d '{
    "case_text": "قضية عمالية تتعلق بفصل تعسفي لموظف",
    "regulation_candidates": [
      {
        "id": 1,
        "title": "نظام العمل - الفصل التعسفي",
        "embedding": [0.1, 0.2, ...],
        "category": "labor_law"
      },
      {
        "id": 2,
        "title": "نظام التجارة",
        "embedding": [0.3, 0.4, ...],
        "category": "commercial_law"
      }
    ],
    "top_k": 5,
    "threshold": 0.3
  }'
```

**Response**:

```json
{
  "related_regulations": [
    {
      "regulation_id": 1,
      "title": "نظام العمل - الفصل التعسفي",
      "similarity_score": 0.8654,
      "category": "labor_law"
    }
  ],
  "query_length": 45,
  "candidates_count": 2
}
```

---

## Performance Optimization Tips

### Model Optimization

- ✅ Use GPU if available (CUDA)
- ✅ Enable model quantization for faster inference
- ✅ Use smaller batch sizes on CPU, larger on GPU
- ✅ Cache frequently accessed embeddings
- ✅ Normalize embeddings once at encoding time

### API Optimization

- ✅ Use async/await for I/O operations
- ✅ Enable response compression
- ✅ Add request rate limiting
- ✅ Use connection pooling
- ✅ Implement proper error handling

### Memory Management

- ✅ Monitor model memory usage
- ✅ Clear cache periodically
- ✅ Use float32 instead of float64
- ✅ Batch processing for large datasets
- ✅ Implement proper cleanup on shutdown

---

## Deployment Options

### Option 1: Docker (Recommended)

```bash
# Build image
docker build -t legal-ai-service:latest .

# Run container
docker run -d \
  --name ai-service \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -e MODEL_NAME=BAAI/bge-m3 \
  legal-ai-service:latest
```

### Option 2: Direct Python

```bash
# Activate environment
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Run with uvicorn
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
```

### Option 3: Production with Gunicorn

```bash
gunicorn app.main:app \
  --workers 2 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 120 \
  --access-logfile - \
  --error-logfile -
```

---

## Model Selection Guide

### For Your Project (Saudi Legal System)

**Recommended**: **BAAI/bge-m3**

- ✅ Excellent Arabic support
- ✅ Multilingual (100+ languages)
- ✅ Long context (8192 tokens)
- ✅ High quality embeddings
- ⚠️ Larger model size (~2GB)
- ⚠️ Slower than lightweight models

**Alternative 1**: **intfloat/multilingual-e5-large**

- ✅ Good Arabic support
- ✅ Strong performance
- ✅ Well-documented
- ⚠️ Shorter context (512 tokens)

**Alternative 2**: **paraphrase-multilingual-mpnet-base-v2**

- ✅ Supports Arabic
- ✅ Smaller size (~1GB)
- ✅ Faster inference
- ⚠️ Lower accuracy than BGE-M3

### Quick Comparison

| Model                         | Dimension | Max Length | Arabic Quality | Speed     | Size  |
| ----------------------------- | --------- | ---------- | -------------- | --------- | ----- |
| BGE-M3                        | 1024      | 8192       | Excellent      | Medium    | 2.3GB |
| multilingual-e5-large         | 1024      | 512        | Very Good      | Fast      | 2.2GB |
| paraphrase-multilingual-mpnet | 768       | 128        | Good           | Very Fast | 1.1GB |
| all-MiniLM-L6-v2              | 384       | 256        | Poor (EN only) | Fastest   | 90MB  |

---

## Integration with Backend

### Backend Communication Flow

1. **User creates/updates a case** → Fastify backend
2. **Backend calls AI service** → `POST /similarity/find-related`
3. **AI service returns top-K regulations** → Similarity scores
4. **Backend stores links** → `case_regulation_links` table
5. **Frontend displays** → AI suggestions with scores

### Example Integration Code (Backend Side)

```typescript
// In Fastify backend: src/services/ai-client.service.ts
async findRelatedRegulations(caseText: string, topK: number = 10) {
  // Get all regulations with embeddings from database
  const regulations = await this.db.query.regulations.findMany({
    where: eq(regulations.status, 'active'),
    columns: {
      id: true,
      title: true,
      embedding: true,  // Pre-stored embedding
      category: true
    }
  });

  // Call AI service
  const response = await fetch(`${AI_SERVICE_URL}/similarity/find-related`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      case_text: caseText,
      regulation_candidates: regulations.map(r => ({
        id: r.id,
        title: r.title,
        embedding: r.embedding,  // Already computed
        category: r.category
      })),
      top_k: topK,
      threshold: 0.3
    })
  });

  return await response.json();
}
```

---

## Useful Commands

```bash
# Development
python -m uvicorn app.main:app --reload --port 8000

# Run tests
pytest app/tests/ -v
pytest app/tests/ --cov=app --cov-report=html

# Code quality
black app/                    # Format code
ruff check app/              # Lint
mypy app/                    # Type check

# Docker
docker build -t ai-service .
docker run -p 8000:8000 ai-service

# Model download (pre-download for faster startup)
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"
```

---

## Environment Setup Instructions

### Step-by-Step Setup

1. **Create Python environment**:

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

2. **Install dependencies**:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

3. **Create `.env` file**:

```bash
cp .env.example .env
# Edit .env with your configuration
```

4. **Download models** (first time):

```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3', cache_folder='./models')"
```

5. **Run the service**:

```bash
uvicorn app.main:app --reload --port 8000
```

6. **Test the API**:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/docs  # Open in browser
```

---

## Production Considerations

### Performance

- Use GPU for faster inference (4-10x speedup)
- Enable model quantization (INT8) for smaller memory footprint
- Use multiple workers for high throughput
- Implement request queueing for burst traffic
- Cache frequent embeddings

### Security

- Add API key authentication if needed
- Rate limit by IP address
- Validate input sizes (prevent DoS)
- Sanitize all inputs
- Use HTTPS in production

### Monitoring

- Log all requests and errors
- Track inference latency
- Monitor memory usage
- Set up health checks
- Alert on model failures

### Scalability

- Stateless design allows horizontal scaling
- Use load balancer for multiple instances
- Consider model serving platforms (TorchServe, Triton)
- Implement caching layer (Redis) for frequent queries

---

## Troubleshooting Guide

### Common Issues

**1. Model download fails**

- Check internet connection
- Verify Hugging Face is accessible
- Use manual download with `huggingface-cli`

**2. Out of memory**

- Reduce batch size
- Use smaller model
- Enable model quantization
- Add more RAM/VRAM

**3. Slow inference**

- Check if using GPU (`torch.cuda.is_available()`)
- Reduce max sequence length
- Use batch processing
- Consider model distillation

**4. Arabic text not working well**

- Verify model supports Arabic (BGE-M3 does)
- Check text preprocessing
- Normalize Arabic characters
- Remove diacritics

---

## Next Steps

After completing the AI microservice:

1. **Integrate with Fastify Backend** - Connect via HTTP client
2. **Test End-to-End Flow** - Case creation → AI linking → Display results
3. **Optimize Performance** - Benchmark and tune
4. **Deploy** - Docker containerization and deployment

---

## Notes

- BGE-M3 supports 100+ languages including Arabic
- Cosine similarity ranges from -1 to 1 (normalized embeddings: 0 to 1)
- Model loading takes 10-30 seconds on first startup
- GPU highly recommended for production (10x faster)
- Embedding dimension: BGE-M3 = 1024, store in PostgreSQL with pgvector
- Consider reranker only if top-5 precision is critical
- All endpoints auto-documented at `/docs` (Swagger UI)

---

**End of AI Microservice Implementation Plan**
