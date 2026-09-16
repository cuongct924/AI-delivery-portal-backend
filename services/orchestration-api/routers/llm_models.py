"""LLM model names API — lists the `model_name` entries configured in
infra/llm-gateways/litellm-config.yaml via the LiteLLM Gateway's own
OpenAI-compatible `/models` endpoint. Backs the Portal's llmModelPicker
(routers/prompts.py's EvaluatePromptRequest.model / routers/rag.py's
RagEvaluateRequest.model both take one of these names) so a user picks
from what's actually configured instead of typing a model_name blind.
"""

from auth.thunder import get_current_user
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from adapters.factory import get_llm_gateway_adapter

router = APIRouter(prefix="/llm-models", tags=["llm-models"])

llm_gateway_adapter = get_llm_gateway_adapter()


class LlmModelNamesResponse(BaseModel):
    names: list[str]


@router.get("", response_model=LlmModelNamesResponse)
def list_llm_model_names(user: dict = Depends(get_current_user)) -> LlmModelNamesResponse:
    models = llm_gateway_adapter.list_models()
    # LiteLLM's /models mirrors OpenAI's /v1/models shape — "id" is the
    # model_name from litellm-config.yaml, not a vendor-specific field.
    return LlmModelNamesResponse(names=[str(model["id"]) for model in models])
