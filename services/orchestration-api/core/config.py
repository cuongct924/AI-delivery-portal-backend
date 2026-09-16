"""Centralized configuration — read from environment variables (.env at the repo root)."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    anthropic_api_key: str = ""
    mlflow_tracking_uri: str = "http://localhost:5000"

    auth_enabled: bool = False
    # Thunder IdP on the OpenChoreo control plane — no realm concept, fixed
    # paths under this base URL.
    thunder_url: str = "http://thunder.openchoreo.localhost:8080"
    # Expected `aud` claim. Every real caller (Scaffolder actions, the
    # Portal Assistant proxy) is Backstage's backend authenticating via
    # client_credentials as openchoreo-backstage-client — not a
    # "orchestration-api" client, which Thunder never registers.
    thunder_audience: str = "openchoreo-backstage-client"

    litellm_gateway_url: str = "http://localhost:4000"
    litellm_master_key: str = ""
    qdrant_url: str = "http://localhost:6333"

    backstage_base_url: str = "http://localhost:7007"
    # Must match a secret in app-config.yaml's backend.auth.externalAccess.
    backstage_service_token: str = ""

    # Server-side only — never accepted from a Scaffolder form field (see
    # routers/llm_serving.py's PrepareLlmDeployRequest.hf_token_secret_ref
    # docstring). Used by adapters/huggingface_hub_adapter.py to read
    # gated models' config.json for the VRAM estimator; unset means gated
    # models fall back to manual GPU sizing input.
    huggingface_hub_token: str = ""

    # extra="ignore": .env carries vars other tooling uses that this
    # service doesn't declare.
    model_config = SettingsConfigDict(env_file="../../.env", extra="ignore")


settings = Settings()
