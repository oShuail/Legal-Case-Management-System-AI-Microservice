"""
Centralized FastAPI exception handlers for the AI microservice.

All handlers emit the canonical envelope:
  { "error": { "code", "message", "details?", "traceId", "timestamp" } }

The traceId is taken from the inbound `X-Request-Id` header so that the
Node backend (which originates most calls) can correlate logs end-to-end.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.utils.logger import logger

from .codes import AICode, STATUS_MAP
from .exceptions import AIError


def _trace_id(request: Request) -> str:
    incoming = request.headers.get("X-Request-Id")
    if incoming:
        return incoming
    return f"ai-{uuid.uuid4().hex[:12]}"


def _envelope(
    *,
    code: str,
    message: str,
    status: int,
    request: Request,
    details: Any = None,
) -> JSONResponse:
    body = {
        "error": {
            "code": code,
            "message": message,
            **({"details": jsonable_encoder(details)} if details is not None else {}),
            "traceId": _trace_id(request),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    }
    return JSONResponse(status_code=status, content=body)


async def ai_error_handler(request: Request, exc: AIError) -> JSONResponse:
    logger.warning(f"AIError code={exc.code.value} status={exc.status} msg={exc.message}")
    return _envelope(
        code=exc.code.value,
        message=exc.message,
        status=exc.status,
        request=request,
        details=exc.details,
    )


async def validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    return _envelope(
        code=AICode.INPUT_INVALID.value,
        message="Invalid input",
        status=400,
        request=request,
        details={"issues": exc.errors()},
    )


async def http_exception_handler(
    request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    # Translate any legacy `HTTPException(...)` raised in routes into our envelope.
    # Map to the appropriate AICode so the status matches the code's canonical value.
    if exc.status_code == 400:
        code = AICode.INPUT_INVALID
    elif exc.status_code == 503:
        code = AICode.MODEL_LOADING
    elif exc.status_code == 504:
        code = AICode.LLM_TIMEOUT
    else:
        # Normalize unknown statuses to AI_INTERNAL (500) to respect
        # the single-status-per-code rule.
        code = AICode.INTERNAL
    return _envelope(
        code=code.value,
        message=str(exc.detail) if exc.detail else "Error",
        status=STATUS_MAP[code],
        request=request,
    )


async def unhandled_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    logger.exception(f"Unhandled exception in AI service: {exc}")
    return _envelope(
        code=AICode.INTERNAL.value,
        message="Internal AI service error",
        status=500,
        request=request,
    )


def register_exception_handlers(app: Any) -> None:
    """Wire all handlers into a FastAPI app. Call once during startup."""
    app.add_exception_handler(AIError, ai_error_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)
