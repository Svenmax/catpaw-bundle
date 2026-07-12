"""Dynamic CatPaw model catalog with a short in-memory cache."""

import http.client
import json
import ssl
import time
from typing import Any, Dict, List

from .crypto import decrypt_response_body
from .token_manager import TokenManager


class ModelCatalog:
    """Fetch models available to the current CatPaw account, cached like IDE UI."""

    def __init__(self, token_manager: TokenManager, host: str, ttl: float = 60.0):
        self.token_manager = token_manager
        self.host = host
        self.ttl = ttl
        self._cache: List[Dict[str, Any]] = []
        self._cached_at = 0.0
        self._ssl_context = ssl.create_default_context()
        self._ssl_context.check_hostname = False
        self._ssl_context.verify_mode = ssl.CERT_NONE

    def get_models(self) -> List[Dict[str, Any]]:
        if self._cache and time.time() - self._cached_at < self.ttl:
            return self._cache

        token = self.token_manager.get_token()
        if not token:
            raise RuntimeError("CatPaw token not found")
        headers = self.token_manager.build_headers(token)
        connection = http.client.HTTPSConnection(self.host, context=self._ssl_context)
        connection.request("GET", "/api/chat/get-user-available-models", headers=headers)
        response = connection.getresponse()
        response_headers = dict(response.getheaders())
        body = response.read().decode("utf-8", errors="replace")
        status = response.status
        connection.close()

        if status != 200:
            raise RuntimeError(f"CatPaw model catalog returned HTTP {status}")
        if response_headers.get("encrypted-key"):
            body = decrypt_response_body(body.strip('"'), response_headers["encrypted-key"])
        payload = json.loads(body)
        if payload.get("code") not in (0, 200):
            raise RuntimeError(payload.get("msg") or "CatPaw model catalog failed")

        models = payload.get("data") or []
        if not isinstance(models, list):
            raise RuntimeError("CatPaw model catalog data is not a list")
        self._cache = models
        self._cached_at = time.time()
        return models

    def find_model(self, model_name: str) -> Dict[str, Any]:
        """Find current-account model metadata by case-insensitive display name."""
        normalized = (model_name or "").lower()
        for model in self.get_models():
            if str(model.get("modelTypeName", "")).lower() == normalized:
                return model
        return {}


def as_openai_models(models: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Map CatPaw model metadata to OpenAI's compact /v1/models shape."""
    result = []
    seen = set()
    for item in models:
        model_id = item.get("modelTypeName")
        if not model_id or model_id.lower() in seen:
            continue
        seen.add(model_id.lower())
        result.append({
            "id": model_id,
            "object": "model",
            "created": 1700000000,
            "owned_by": "catpaw",
            "metadata": {
                "model_type": item.get("modelType"),
                "support_image": bool(item.get("supportImage")),
                "support_thinking": bool(item.get("supportThinking")),
                "support_agent": bool(item.get("supportAgent")),
                "description": item.get("modelUsageDesc") or "",
            },
        })
    return result
