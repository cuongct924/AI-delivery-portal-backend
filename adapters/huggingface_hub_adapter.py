"""Real IHuggingFaceHubAdapter — reads a model's metadata straight off the
public HuggingFace Hub HTTP API, no SDK dependency (same "just httpx"
choice as auth/thunder.py's own Thunder JWKS fetch).

Two calls per lookup:
  1. GET /api/models/{model_id} — existence, gated flag, param count
     (`safetensors.total`), license.
  2. GET /{model_id}/resolve/main/config.json — architecture fields
     (num_hidden_layers/hidden_size/num_attention_heads/
     num_key_value_heads/max_position_embeddings) the VRAM estimator
     (llm_serving/gpu_sizing.py) needs. Skipped when (1) already showed
     the model is gated and no token is configured — would 401 anyway.
"""

from typing import cast

import httpx
from core.config import settings

from adapters.interfaces import HuggingFaceModelInfo, IHuggingFaceHubAdapter

_HUB_API_BASE = "https://huggingface.co"
_TIMEOUT_SECONDS = 10.0


class HuggingFaceHubAdapter(IHuggingFaceHubAdapter):
    def get_model_info(self, model_id: str) -> HuggingFaceModelInfo:
        headers = (
            {"Authorization": f"Bearer {settings.huggingface_hub_token}"}
            if settings.huggingface_hub_token
            else {}
        )
        with httpx.Client(timeout=_TIMEOUT_SECONDS, headers=headers) as client:
            response = client.get(f"{_HUB_API_BASE}/api/models/{model_id}")
            if response.status_code == 404:
                return self._not_found(model_id)
            if response.status_code in (401, 403):
                # No token, or token lacks access to this gated repo — still a
                # real model, just can't read its detail.
                return self._gated_without_access(model_id)
            response.raise_for_status()
            metadata = response.json()

            is_gated = bool(metadata.get("gated"))
            param_count = metadata.get("safetensors", {}).get("total")
            license_name = metadata.get("cardData", {}).get("license") or metadata.get("license")

            # Skip the config.json call when it would 401 (gated + no token) —
            # saves a round trip for the common case.
            can_read_config = not is_gated or bool(headers)
            architecture = self._get_architecture(client, model_id) if can_read_config else None

        return HuggingFaceModelInfo(
            model_id=model_id,
            exists=True,
            is_gated=is_gated,
            param_count_billion=param_count / 1e9 if param_count else None,
            max_context_length=cast(
                "float | None", (architecture or {}).get("max_position_embeddings")
            ),
            num_layers=cast("int | None", (architecture or {}).get("num_hidden_layers")),
            hidden_size=cast("int | None", (architecture or {}).get("hidden_size")),
            num_attention_heads=cast("int | None", (architecture or {}).get("num_attention_heads")),
            num_key_value_heads=cast(
                "int | None",
                (architecture or {}).get("num_key_value_heads")
                or (architecture or {}).get("num_attention_heads"),
            ),
            license=license_name,
        )

    def _get_architecture(self, client: httpx.Client, model_id: str) -> dict[str, object] | None:
        response = client.get(f"{_HUB_API_BASE}/{model_id}/resolve/main/config.json")
        if response.status_code != 200:
            # Gated with an insufficient token, or a non-standard layout
            # (e.g. GGUF-only) — caller falls back to manual GPU sizing.
            return None
        return response.json()

    def _not_found(self, model_id: str) -> HuggingFaceModelInfo:
        return HuggingFaceModelInfo(
            model_id=model_id,
            exists=False,
            is_gated=False,
            param_count_billion=None,
            max_context_length=None,
            num_layers=None,
            hidden_size=None,
            num_attention_heads=None,
            num_key_value_heads=None,
            license=None,
        )

    def _gated_without_access(self, model_id: str) -> HuggingFaceModelInfo:
        return HuggingFaceModelInfo(
            model_id=model_id,
            exists=True,
            is_gated=True,
            param_count_billion=None,
            max_context_length=None,
            num_layers=None,
            hidden_size=None,
            num_attention_heads=None,
            num_key_value_heads=None,
            license=None,
        )
