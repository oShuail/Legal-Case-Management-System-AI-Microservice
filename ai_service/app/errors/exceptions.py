"""
Custom exception hierarchy for the AI microservice.

Raising `AIError(AICode.X)` from any router or core module produces the
canonical envelope when caught by the registered exception handler.
"""

from __future__ import annotations

from typing import Any

from .codes import AICode, DEFAULT_MESSAGES, STATUS_MAP


class AIError(Exception):
    """Base exception that produces a canonical error envelope."""

    def __init__(
        self,
        code: AICode,
        message: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.code: AICode = code
        self.status: int = STATUS_MAP[code]
        self.message: str = message or DEFAULT_MESSAGES[code]
        self.details: dict[str, Any] | None = details
        super().__init__(self.message)
