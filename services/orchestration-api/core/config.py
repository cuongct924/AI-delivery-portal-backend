"""Centralized configuration — read from environment variables (.env at the repo root)."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    anthropic_api_key: str = ""
    mlflow_tracking_uri: str = "http://localhost:5000"

    auth_enabled: bool = False
    # Thunder IdP on the OpenChoreo control plane — no realm concept, fixed
    # paths under this base URL.
    thunder_url: str = "http://thunder.openchoreo.localhost:8080"
    # Expected `aud` claim — must match the calling client's Thunder registration.
    thunder_audience: str = "orchestration-api"

    litellm_gateway_url: str = "http://localhost:4000"
    litellm_master_key: str = ""
    qdrant_url: str = "http://localhost:6333"

    backstage_base_url: str = "http://localhost:7007"
    # Must match a secret in app-config.yaml's backend.auth.externalAccess.
    backstage_service_token: str = ""

    # extra="ignore": .env carries vars other tooling uses that this
    # service doesn't declare.
    model_config = SettingsConfigDict(env_file="../../.env", extra="ignore")


settings = Settings()
