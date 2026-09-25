"""Adapter for the LLM Gateway (LiteLLM Proxy) — routes requests and manages
API keys & rate limits when calling multiple different LLMs, instead of
calling each vendor's SDK directly. Deployed via
infra/ai-platform-zone/litellm.yaml, config at
infra/ai-platform-zone/litellm-config.yaml.
"""

import json
import os
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import cast

import httpx

from adapters.ai_platform.interfaces import (
    ChatCompletionChunk,
    ChatCompletionResponse,
    ILLMGatewayAdapter,
)


class LiteLLMGatewayAdapter(ILLMGatewayAdapter):
    def __init__(self, base_url: str | None = None, api_key: str | None = None):
        self.base_url = base_url or os.getenv("LITELLM_GATEWAY_URL", "http://localhost:4000")
        self.api_key = api_key or os.getenv("LITELLM_MASTER_KEY", "")

    def chat_completion(
        self, model: str, messages: Sequence[Mapping[str, object]], **kwargs: object
    ) -> ChatCompletionResponse:
        response = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": model, "messages": messages, **kwargs},
            # Generous: when the primary model errors (e.g. Anthropic out of
            # credit), LiteLLM falls back to a local model that can take a
            # while on a long RAG context — 30s was too tight and surfaced as
            # a 500 on /chat.
            timeout=120,
        )
        response.raise_for_status()
        result = response.json()
        # LiteLLM returns per-call cost as a header, not in the JSON body.
        cost_header = response.headers.get("x-litellm-response-cost")
        result["response_cost_usd"] = float(cost_header) if cost_header is not None else None
        return cast(ChatCompletionResponse, result)

    async def chat_completion_stream(
        self, model: str, messages: Sequence[Mapping[str, object]], **kwargs: object
    ) -> AsyncIterator[ChatCompletionChunk]:
        # First async-httpx use in this repo (every other adapter call here
        # is sync httpx.post/get) — kept self-contained to this one method
        # rather than threading async through the whole adapter.
        #
        # Cost isn't attached here the way chat_completion's response_cost_usd
        # is: that reads an `x-litellm-response-cost` response header, which
        # (verified live, both streaming and non-streaming) this LiteLLM
        # deployment doesn't actually send — response_cost_usd is already
        # always None today regardless of streaming, a pre-existing gap
        # unrelated to this method.
        async with (
            httpx.AsyncClient(timeout=120) as client,
            client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": model,
                    "messages": messages,
                    "stream": True,
                    # Otherwise the final chunk has no `usage` at all.
                    "stream_options": {"include_usage": True},
                    **kwargs,
                },
            ) as response,
        ):
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue  # blank keepalive lines between SSE frames
                payload = line[len("data: ") :]
                if payload == "[DONE]":
                    return
                yield cast(ChatCompletionChunk, json.loads(payload))

    def list_models(self) -> list[dict[str, object]]:
        response = httpx.get(
            f"{self.base_url}/models",
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=10,
        )
        response.raise_for_status()
        return response.json().get("data", [])

    def embed(self, model: str, input_texts: list[str]) -> list[list[float]]:
        response = httpx.post(
            f"{self.base_url}/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": model, "input": input_texts},
            timeout=30,
        )
        response.raise_for_status()
        return [item["embedding"] for item in response.json()["data"]]

    def get_spend_report(
        self, start_date: str, end_date: str, group_by: str | None = None
    ) -> list[dict[str, object]]:
        params: dict[str, str] = {"start_date": start_date, "end_date": end_date}
        if group_by is not None:
            params["group_by"] = group_by
        response = httpx.get(
            f"{self.base_url}/global/spend/report",
            headers={"Authorization": f"Bearer {self.api_key}"},
            params=params,
            timeout=10,
        )
        response.raise_for_status()
        return cast(list[dict[str, object]], response.json())
