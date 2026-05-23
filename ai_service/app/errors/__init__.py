"""
Error handling subsystem for the Silah AI microservice.

Mirrors the canonical envelope used by the Node backend so that
clients (backend, frontend) can rely on a single error contract.
"""

from .codes import AICode, STATUS_MAP, DEFAULT_MESSAGES
from .exceptions import AIError

__all__ = ["AICode", "AIError", "STATUS_MAP", "DEFAULT_MESSAGES"]
