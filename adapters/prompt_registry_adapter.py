"""MLflow-backed IVersionRegistryAdapter for the "prompt" kind — replaces
JsonFileVersionRegistryAdapter for prompts specifically. Uses MLflow's
native Prompt Registry (mlflow.genai.*), not the Model Registry — a prompt
is text, not a trained model artifact. See routers/prompts.py's docstring
for why this is a separate getter from get_registry_adapter().

Scoped to kind="prompt" only. routers/rag.py's "rag-index" kind keeps
using JsonFileVersionRegistryAdapter via get_registry_adapter() — a RAG
index pointer has no equivalent in MLflow's Prompt Registry, so this class
does not attempt to also serve that kind.
"""

import os
from collections.abc import Mapping

import mlflow
import mlflow.genai
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from adapters.interfaces import IVersionRegistryAdapter

_ACTIVE_ALIAS = "active"
_SUPPORTED_KIND = "prompt"


class MlflowPromptRegistryAdapter(IVersionRegistryAdapter):
    def __init__(self, tracking_uri: str | None = None):
        self.tracking_uri = tracking_uri or os.getenv(
            "MLFLOW_TRACKING_URI", "http://localhost:5000"
        )
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
        # get_prompt_version returns None (not a raised exception) when the
        # version doesn't exist — confirmed from the installed SDK's own
        # source (mlflow.tracking.MlflowClient.get_prompt_version).
        prompt_version = self.client.get_prompt_version(name=name, version=version)
        if prompt_version is None:
            raise ValueError(f"{kind}/{name} has no version {version!r}")
        return {
            "persona": (prompt_version.tags or {}).get("persona", ""),
            "content": prompt_version.template,
        }

    def list_versions(self, kind: str, name: str) -> dict[str, dict[str, object]]:
        self._check_kind(kind)
        # mlflow.genai has no bulk "list all versions of a prompt" call as
        # of MLflow 3.15.1 (only search_prompts() over prompt *names*) —
        # walk version numbers until get_prompt_version returns None. Fine
        # at the low version counts a persona prompt realistically has.
        versions: dict[str, dict[str, object]] = {}
        version = 1
        while True:
            try:
                # get_prompt_version returns None for a missing *version* of
                # a name that does exist, but raises MlflowException
                # (RESOURCE_DOES_NOT_EXIST) when `name` itself was never
                # registered — same "any MlflowException means not found"
                # reading as get_active_version above.
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

    def get_active_version(self, kind: str, name: str) -> str | None:
        self._check_kind(kind)
        try:
            pv = self.client.get_prompt_version_by_alias(name=name, alias=_ACTIVE_ALIAS)
        except MlflowException:
            return None
        return str(pv.version)

    def set_active_version(self, kind: str, name: str, version: str) -> None:
        self._check_kind(kind)
        try:
            mlflow.genai.set_prompt_alias(name=name, alias=_ACTIVE_ALIAS, version=int(version))
        except MlflowException as exc:
            # Same reading as get_active_version() above: any MlflowException
            # here means the version doesn't exist.
            raise ValueError(f"{kind}/{name} has no registered versions") from exc

    @staticmethod
    def _check_kind(kind: str) -> None:
        if kind != _SUPPORTED_KIND:
            raise ValueError(
                f"MlflowPromptRegistryAdapter only supports kind={_SUPPORTED_KIND!r}, got {kind!r}"
            )
