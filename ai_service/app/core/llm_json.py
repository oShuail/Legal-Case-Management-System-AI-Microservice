"""
Shared LLM JSON helper.

Provides a single, config-gated way to ask the configured LLM provider for a
strict JSON object. Used by every "deterministic baseline + optional LLM
enrichment" route (regulation insights, amendment impact, admin AI insights).

Design rules:
- Deterministic output is always authoritative. The LLM only *enriches* the
  narrative; callers must keep their heuristic fallback.
- Never raises. On any failure it returns ``(None, reason)`` so the caller can
  append ``reason`` to its ``warnings`` and fall back to deterministic output.
- Gated on ``settings.llm_provider``. When the provider is heuristic/disabled or
  not configured, returns ``(None, "llm_disabled" | "llm_not_configured")`` —
  these two reasons are *expected* and should NOT be surfaced as warnings.
"""

from __future__ import annotations

import json
import re
from typing import Any

import httpx

from app.config import settings
from app.utils.logger import logger

# Reasons that mean "LLM intentionally off / not set up" — callers should not
# surface these as warnings (they are the normal state with LLM_PROVIDER off).
LLM_DISABLED_REASONS = {"llm_disabled", "llm_not_configured"}


def extract_json_object(raw_text: str) -> dict[str, Any] | None:
    """Best-effort parse of a JSON object out of an LLM completion.

    Handles ```json fenced blocks and trailing prose by falling back to the
    outermost ``{...}`` span.
    """
    if not raw_text:
        return None

    text = raw_text.strip()
    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```", "", text)
    text = re.sub(r"```$", "", text)
    text = text.strip()

    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            value = json.loads(text[start : end + 1])
            if isinstance(value, dict):
                return value
        except Exception:
            return None

    return None


async def try_llm_json(
    *,
    system_prompt: str,
    user_payload: dict[str, Any],
    timeout_seconds: float,
) -> tuple[dict[str, Any] | None, str | None]:
    """Ask the configured LLM for a JSON object.

    Returns ``(parsed_dict, None)`` on success, or ``(None, reason)`` otherwise.
    Reasons in ``LLM_DISABLED_REASONS`` are expected; other reasons are
    transient errors worth surfacing as a warning.
    """
    if settings.llm_provider.lower() in {"", "heuristic", "none", "disabled"}:
        return None, "llm_disabled"
    if not settings.llm_base_url or not settings.llm_model:
        return None, "llm_not_configured"

    url = settings.llm_base_url.rstrip("/") + "/chat/completions"
    headers: dict[str, str] = {
        "Content-Type": "application/json",
    }
    if settings.llm_api_key:
        headers["Authorization"] = f"Bearer {settings.llm_api_key}"

    payload = {
        "model": settings.llm_model,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
    }

    try:
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            response = await client.post(url, headers=headers, json=payload)
        if response.status_code >= 400:
            return None, f"llm_http_{response.status_code}"

        body = response.json()
        content = (
            body.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        parsed = extract_json_object(content)
        if not parsed:
            return None, "llm_invalid_json"

        return parsed, None
    except Exception as exc:
        logger.warning(
            "LLM structured generation failed",
            extra={"error": str(exc)},
        )
        return None, "llm_request_failed"
