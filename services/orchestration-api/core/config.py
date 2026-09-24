"""Centralized configuration — read from environment variables (.env at the repo root)."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    anthropic_api_key: str = ""
    mlflow_tracking_uri: str = "http://localhost:5000"

    auth_enabled: bool = False
    # OpenChoreo's Thunder IdP has no realm concept — fixed base URL only.
    thunder_url: str = "http://thunder.openchoreo.localhost:8080"
    # Backstage's backend authenticates as openchoreo-backstage-client, not
    # orchestration-api — Thunder never registers the latter.
    thunder_audience: str = "openchoreo-backstage-client"

    litellm_gateway_url: str = "http://localhost:4000"
    litellm_master_key: str = ""
    qdrant_url: str = "http://localhost:6333"

    # Repo root the RAG ingest resolves repo-relative source paths against
    # (routers/rag.py) — i.e. the directory that CONTAINS docs/, not docs/
    # itself. Unset falls back to the repo root derived from __file__; the
    # Docker image sets it to /app, where docs/ is copied.
    docs_root: str = ""

    backstage_base_url: str = "http://localhost:7007"
    # Must match a secret in app-config.yaml's backend.auth.externalAccess.
    backstage_service_token: str = ""

    # Server-side only, never from a Scaffolder form field; unset falls
    # back to manual GPU sizing input for gated models.
    huggingface_hub_token: str = ""

    # extra="ignore": .env carries vars other tooling uses that this
    # service doesn't declare.
    model_config = SettingsConfigDict(env_file="../../.env", extra="ignore")


settings = Settings()
