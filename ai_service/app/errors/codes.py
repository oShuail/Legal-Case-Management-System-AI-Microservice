"""
Canonical error codes for the AI microservice.

The Node backend's src/errors/codes.ts mirrors the `AI_*` codes here so it
can parse upstream errors and translate them into `EXTERNAL_AI_*`.
"""

from __future__ import annotations

from enum import Enum


class AICode(str, Enum):
    INPUT_INVALID = "AI_INPUT_INVALID"
    MODEL_LOADING = "AI_MODEL_LOADING"
    EMBEDDING_FAILED = "AI_EMBEDDING_FAILED"
    LLM_FAILED = "AI_LLM_FAILED"
    LLM_TIMEOUT = "AI_LLM_TIMEOUT"
    INTERNAL = "AI_INTERNAL"


STATUS_MAP: dict[AICode, int] = {
    AICode.INPUT_INVALID: 400,
    AICode.MODEL_LOADING: 503,
    AICode.EMBEDDING_FAILED: 500,
    AICode.LLM_FAILED: 502,
    AICode.LLM_TIMEOUT: 504,
    AICode.INTERNAL: 500,
}


DEFAULT_MESSAGES: dict[AICode, str] = {
    AICode.INPUT_INVALID: "Invalid AI input",
    AICode.MODEL_LOADING: "AI model is still loading, please retry shortly",
    AICode.EMBEDDING_FAILED: "Failed to compute embeddings",
    AICode.LLM_FAILED: "Language model verification failed",
    AICode.LLM_TIMEOUT: "Language model timed out",
    AICode.INTERNAL: "Internal AI service error",
}
