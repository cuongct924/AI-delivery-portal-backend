"""MLflow-backed IVersionRegistryAdapter for the "prompt" kind — uses
MLflow's native Prompt Registry (mlflow.genai.*), not the Model Registry (a
prompt is text, not a trained artifact); see routers/prompts.py's docstring
for why this is a separate getter. Scoped to kind="prompt" only —
routers/rag.py's "rag-index" kind keeps JsonFileVersionRegistryAdapter, since
a RAG index pointer has no MLflow Prompt Registry equivalent.
"""

from collections.abc import Mapping

import mlflow
import mlflow.genai
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from adapters.ai_platform._mlflow import get_mlflow_tracking_uri
from adapters.ai_platform.interfaces import DEFAULT_ENVIRONMENT, IVersionRegistryAdapter

# Kept as the bare "active" alias for DEFAULT_ENVIRONMENT specifically (not
# e.g. "active-production") so a prompt version already aliased "active" by
# a pre-environment run of this adapter is still read as production's —
# only non-default environments get their own "active-<environment>" alias.
_ACTIVE_ALIAS = "active"
_SUPPORTED_KIND = "prompt"


def _alias_for(environment: str) -> str:
    return _ACTIVE_ALIAS if environment == DEFAULT_ENVIRONMENT else f"{_ACTIVE_ALIAS}-{environment}"


class MlflowPromptRegistryAdapter(IVersionRegistryAdapter):
    def __init__(self, tracking_uri: str | None = None):
        self.tracking_uri = tracking_uri or get_mlflow_tracking_uri()
        mlflow.set_tracking_uri(self.tracking_uri)
        self.client = MlflowClient(tracking_uri=self.tracking_uri)

    def register_version(self, kind: str, name: str, metadata: Mapping[str, object]) -> str:
        self._check_kind(kind)
        result = mlflow.genai.register_prompt(
            name=name,
            template=str(metadata["content"]),
            tags={"persona": str(metadata["persona"])},
        )
        return str(result.version)

    def list_names(self, kind: str) -> list[str]:
        self._check_kind(kind)
        return [p.name for p in mlflow.genai.search_prompts()]

    def get_version(self, kind: str, name: str, version: str) -> dict[str, object]:
        self._check_kind(kind)
        # get_prompt_version returns None (not an exception) for a missing version.
        prompt_version = self.client.get_prompt_version(name=name, version=version)
        if prompt_version is None:
            raise ValueError(f"{kind}/{name} has no version {version!r}")
        return {
            "persona": (prompt_version.tags or {}).get("persona", ""),
            "content": prompt_version.template,
        }

    def list_versions(self, kind: str, name: str) -> dict[str, dict[str, object]]:
        self._check_kind(kind)
        # mlflow.genai (3.15.1) has no bulk list call — walk versions until None.
        versions: dict[str, dict[str, object]] = {}
        version = 1
        while True:
            try:
                # None = missing version; MlflowException = name never registered — both not found.
                pv = self.client.get_prompt_version(name=name, version=str(version))
            except MlflowException:
                break
            if pv is None:
                break
            versions[str(version)] = {
                "persona": (pv.tags or {}).get("persona", ""),
                "content": pv.template,
            }
            version += 1
        return versions

    def get_active_version(
        self, kind: str, name: str, environment: str = DEFAULT_ENVIRONMENT
    ) -> str | None:
        self._check_kind(kind)
        try:
            pv = self.client.get_prompt_version_by_alias(name=name, alias=_alias_for(environment))
        except MlflowException:
            return None
        return str(pv.version)

    def set_active_version(
        self, kind: str, name: str, version: str, environment: str = DEFAULT_ENVIRONMENT
    ) -> None:
        self._check_kind(kind)
        try:
            mlflow.genai.set_prompt_alias(
                name=name, alias=_alias_for(environment), version=int(version)
            )
        except MlflowException as exc:
            # Any MlflowException here means the version doesn't exist.
            raise ValueError(f"{kind}/{name} has no registered versions") from exc

    @staticmethod
    def _check_kind(kind: str) -> None:
        if kind != _SUPPORTED_KIND:
            raise ValueError(
                f"MlflowPromptRegistryAdapter only supports kind={_SUPPORTED_KIND!r}, got {kind!r}"
            )
