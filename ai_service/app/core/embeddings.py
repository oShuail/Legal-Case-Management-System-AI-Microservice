from __future__ import annotations

import hashlib
import math
from collections import OrderedDict
from typing import Any, List, Sequence

import httpx

from app.config import settings
from app.utils.logger import logger

try:
    # Required only for the real BGE model path.
    from sentence_transformers import SentenceTransformer  # type: ignore
except ImportError:
    SentenceTransformer = None  # type: ignore


class EmbeddingService:
    """
    Embedding wrapper with multiple providers.

    - provider="fake":
        Fast deterministic vectors for local/dev and tests.

    - provider="bge":
        Local sentence-transformers model (CPU/GPU).

    - provider="hf":
        Hugging Face hosted inference with provider fallback order.
    """

    def __init__(
        self,
        provider: str | None = None,
        model_name: str | None = None,
        device: str | None = None,
    ) -> None:
        self.provider = (provider or settings.embeddings_provider).lower()
        self.model = None
        self._cache: OrderedDict[str, List[float]] = OrderedDict()
        self._cache_size = max(64, int(settings.hf_embed_cache_size))
        self._hf_provider_order = self._resolve_hf_provider_order()

        # ---- Provider-health state ----
        # Tracked so the frontend can show a friendly "AI warming up" banner
        # whenever the local BGE fallback has to be loaded or is currently
        # serving requests (instead of the fast hosted endpoint).
        #
        # local_model_state: "not_loaded" | "loading" | "ready"
        # last_provider:     name of the provider that served the most recent
        #                    successful embed call ("endpoint" | "serverless"
        #                    | "bge" | "fake" | None)
        # fallback_active:   True when last_provider == "bge" AND provider
        #                    config is "hf" (i.e. we're on the backup model).
        self._local_model_state: str = "not_loaded"
        self._last_provider: str | None = None
        self._fallback_active: bool = False

        # Remember constructor args so a lazy fallback load can use the same.
        self._init_model_name = model_name
        self._init_device = device

        if self.provider == "bge":
            # Primary provider is local BGE — load it now, it'll be hit on
            # every request.
            self._init_local_bge_model(model_name=model_name, device=device)

        # When provider == "hf", DO NOT eagerly load the local BGE fallback.
        # The local BGE-M3 model weighs ~6-8 GB of RAM with sentence-
        # transformers + torch, and the remote HF endpoint covers 99%+ of
        # requests. The fallback path at line ~229 already calls
        # `_init_local_bge_model()` lazily on first invocation (and that
        # method is idempotent), so we keep the same correctness guarantee
        # without paying the startup memory cost.

    def _init_local_bge_model(
        self,
        model_name: str | None = None,
        device: str | None = None,
    ) -> None:
        if self.model is not None:
            self._local_model_state = "ready"
            return
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers is required for provider='bge'. "
                "Install it with `pip install sentence-transformers`."
            )
        name = model_name or settings.embedding_model_name
        dev = device or settings.embedding_device
        # Flip state to "loading" BEFORE the heavy load so a concurrent health
        # poll sees the warming-up state and can surface it to the user.
        self._local_model_state = "loading"
        logger.info("Loading local BGE model lazily name={} device={}", name, dev)
        self.model = SentenceTransformer(name, device=dev)
        self._local_model_state = "ready"

    def get_health(self) -> dict:
        """
        Snapshot of embedding-provider health for /health/embeddings.

        Designed for the UI: returns plain user-friendly flags rather than
        internal jargon so the frontend can show a banner like
          "AI assistant is warming up — your request will be ready shortly."
        when fallback is loading, and
          "AI assistant is running on the backup model."
        when fallback is actively serving.
        """
        return {
            "configured_provider": self.provider,
            "hf_provider_order": self._hf_provider_order,
            "local_model_state": self._local_model_state,
            "last_provider": self._last_provider,
            "fallback_active": self._fallback_active,
            # `warming_up` is the single flag the UI typically watches.
            "warming_up": self._local_model_state == "loading",
        }

    def _resolve_hf_provider_order(self) -> list[str]:
        allowed = {"serverless", "endpoint", "bge"}
        configured = [
            item for item in settings.hf_embed_provider_order if item in allowed
        ]
        if not configured:
            return ["serverless", "endpoint", "bge"]
        return configured

    @staticmethod
    def _normalize_vector(vec: Sequence[float]) -> List[float]:
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [float(x / norm) for x in vec]

    def _embed_fake(self, texts: Sequence[str]) -> List[List[float]]:
        dimensions = 1024
        vectors: List[List[float]] = []

        for text in texts:
            lowered = text.lower()
            vector = [0.0] * dimensions
            vector[0] = 1.0 if "contract" in lowered else 0.0
            vector[1] = 1.0 if "labor" in lowered else 0.0
            vector[2] = 1.0 if "case" in lowered else 0.0
            vector[3] = min(len(text) / 100.0, 10.0)

            for token in lowered.split():
                if not token:
                    continue
                digest = hashlib.sha1(token.encode("utf-8")).digest()
                hashed = int.from_bytes(digest[:4], byteorder="little", signed=False)
                idx = 4 + (hashed % (dimensions - 4))
                vector[idx] += 1.0

            vectors.append(vector)

        return vectors

    def _cache_get(self, text: str) -> List[float] | None:
        if text not in self._cache:
            return None
        value = self._cache.pop(text)
        self._cache[text] = value
        return value

    def _cache_put(self, text: str, vector: List[float]) -> None:
        if text in self._cache:
            self._cache.pop(text)
        self._cache[text] = vector
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

    def _resolve_hf_serverless_url(self) -> str:
        model_name = (
            settings.hf_serverless_model_name.strip()
            or settings.embedding_model_name.strip()
        )
        if not model_name:
            raise ValueError("HF serverless model name is not configured")
        return f"{settings.hf_serverless_api_base.rstrip('/')}/{model_name}"

    def _request_hf_embeddings(
        self,
        *,
        provider: str,
        texts: Sequence[str],
        normalize: bool,
    ) -> List[List[float]]:
        if provider == "serverless":
            url = self._resolve_hf_serverless_url()
            token = settings.hf_serverless_api_token.strip()
            if not token:
                raise RuntimeError("HF_SERVERLESS_API_TOKEN is not configured")
        elif provider == "endpoint":
            url = settings.hf_endpoint_url.strip()
            token = settings.hf_endpoint_api_token.strip()
            if not url:
                raise RuntimeError("HF_ENDPOINT_URL is not configured")
            if not token:
                raise RuntimeError("HF_ENDPOINT_API_TOKEN is not configured")
        else:
            raise RuntimeError(f"Unsupported HF provider '{provider}'")

        timeout = max(5.0, float(settings.hf_embed_request_timeout_seconds))
        attempts = max(1, int(settings.hf_embed_retry_attempts) + 1)
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "inputs": list(texts),
            "options": {"wait_for_model": True},
            "normalize": normalize,
        }

        with httpx.Client(timeout=timeout) as client:
            for attempt in range(1, attempts + 1):
                try:
                    response = client.post(url, headers=headers, json=payload)
                    if response.status_code in {408, 409, 425, 429, 500, 502, 503, 504}:
                        if attempt < attempts:
                            logger.warning(
                                "HF embed retry provider={} status={} attempt={}/{}",
                                provider,
                                response.status_code,
                                attempt,
                                attempts,
                            )
                            continue
                    response.raise_for_status()
                    data = response.json()
                    if not isinstance(data, list) or not data:
                        raise RuntimeError(
                            f"HF embedding response is not a non-empty list for provider '{provider}'"
                        )
                    if not all(isinstance(item, list) for item in data):
                        raise RuntimeError(
                            f"HF embedding response contains invalid vector entries for provider '{provider}'"
                        )
                    return [[float(value) for value in vector] for vector in data]
                except Exception as exc:
                    if attempt < attempts:
                        logger.warning(
                            "HF embed retry exception provider={} attempt={}/{} error={}",
                            provider,
                            attempt,
                            attempts,
                            str(exc),
                        )
                        continue
                    raise

        raise RuntimeError(f"HF embedding request failed for provider '{provider}'")

    def _embed_hf(self, texts: Sequence[str], normalize: bool) -> List[List[float]]:
        if not texts:
            return []

        results: List[List[float] | None] = [None] * len(texts)
        missing_indices: list[int] = []
        missing_texts: list[str] = []

        for idx, text in enumerate(texts):
            cached = self._cache_get(text)
            if cached is not None:
                results[idx] = cached
            else:
                missing_indices.append(idx)
                missing_texts.append(text)

        if missing_texts:
            max_batch = max(1, int(settings.hf_embed_max_batch_size))
            computed_vectors: list[List[float]] = []
            selected_provider: str | None = None
            last_error: Exception | None = None

            for provider_name in self._hf_provider_order:
                try:
                    provider_vectors: list[List[float]] = []
                    if provider_name == "bge":
                        # Lazy-load on first fallback. _init_local_bge_model
                        # is idempotent so subsequent calls are no-ops.
                        self._init_local_bge_model(
                            model_name=self._init_model_name,
                            device=self._init_device,
                        )
                        for i in range(0, len(missing_texts), max_batch):
                            batch = missing_texts[i : i + max_batch]
                            raw_vectors = self.model.encode(  # type: ignore[union-attr]
                                list(batch),
                                convert_to_numpy=True,
                            )
                            provider_vectors.extend([v.tolist() for v in raw_vectors])
                    else:
                        for i in range(0, len(missing_texts), max_batch):
                            batch = missing_texts[i : i + max_batch]
                            provider_vectors.extend(
                                self._request_hf_embeddings(
                                    provider=provider_name,
                                    texts=batch,
                                    normalize=normalize,
                                )
                            )

                    if len(provider_vectors) != len(missing_texts):
                        raise RuntimeError(
                            f"Provider '{provider_name}' returned {len(provider_vectors)} vectors for {len(missing_texts)} texts"
                        )

                    computed_vectors = provider_vectors
                    selected_provider = provider_name
                    break
                except Exception as exc:
                    last_error = exc
                    logger.warning(
                        "HF embed provider failed provider={} error={}",
                        provider_name,
                        str(exc),
                    )

            if selected_provider is None:
                raise RuntimeError("All embedding providers failed") from last_error

            # Update health state so /health/embeddings can surface to the UI.
            self._last_provider = selected_provider
            self._fallback_active = selected_provider == "bge"

            logger.info(
                "HF embed provider selected provider={} batch_text_count={}",
                selected_provider,
                len(missing_texts),
            )

            for list_index, vector in enumerate(computed_vectors):
                text = missing_texts[list_index]
                output_vector = (
                    self._normalize_vector(vector)
                    if normalize and selected_provider == "bge"
                    else vector
                )
                self._cache_put(text, output_vector)
                results[missing_indices[list_index]] = output_vector

        final_vectors = [item for item in results if item is not None]
        if len(final_vectors) != len(texts):
            raise RuntimeError(
                f"Embedding provider returned incomplete results ({len(final_vectors)}/{len(texts)})"
            )

        return [list(vector) for vector in final_vectors]

    def embed_documents(
        self,
        texts: Sequence[str],
        normalize: bool | None = None,
    ) -> List[List[float]]:
        if self.provider == "bge":
            raw_vectors = self.model.encode(  # type: ignore[union-attr]
                list(texts),
                convert_to_numpy=True,
            )
            vectors: List[List[float]] = [v.tolist() for v in raw_vectors]
        elif self.provider == "hf":
            should_normalize = True if normalize is None else normalize
            vectors = self._embed_hf(texts, normalize=should_normalize)
        else:
            vectors = self._embed_fake(texts)

        if normalize is None:
            normalize = self.provider in {"bge", "hf"}

        if normalize and self.provider not in {"hf"}:
            vectors = [self._normalize_vector(v) for v in vectors]

        return vectors

    def embed_query(
        self,
        text: str,
        normalize: bool | None = None,
    ) -> List[float]:
        return self.embed_documents([text], normalize=normalize)[0]
