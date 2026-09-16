"""Adapter for OpenChoreo's DeploymentPipeline/ProjectReleaseBinding-based
promotion — the "openchoreo" real implementation of IPromotionAdapter,
replacing observability-server's mocked get_promotion_status.

Same `kubernetes.client.CustomObjectsApi` convention as
adapters/openchoreo_inference_adapter.py, against the ProjectReleaseBinding
CRD (see infra/openchoreo/deployment-pipeline.yaml,
infra/openchoreo/environments.yaml,
infra/openchoreo/telco-fraud-detection/projectreleasebinding-development.yaml).

The promotion *path* (development -> staging -> production) is hardcoded
here rather than read from the live DeploymentPipeline object — it mirrors
infra/openchoreo/deployment-pipeline.yaml's `spec.promotionPaths` exactly,
and reading it dynamically would need parsing an arbitrary DAG for a
single, fixed 2-hop chain this repo will only ever have one of. Update both
places together if that file's promotionPaths ever change.

**Why `promote()` alone counts as "manual approval"**: the only real path
that reaches this method is a Dev running the "Evaluate & Deploy Model"
Golden Path template themselves with action=promote (see
templates/register-deploy/template.yaml's `promote` step in the frontend
repo — merged into that template rather than a separate one, same screen
as deploy/rollback) — nothing in agents/mcp-servers/ ever calls it
(get_promotion_status stays the read-only MCP tool; promote() isn't
exposed as a tool at all). A
second approval gate on top (e.g. a stored "pending" state a *different*
human must separately click "approve" on) was considered and rejected:
this repo has no notion of distinct requester/approver identities yet
(every call carries the same Thunder-authenticated user, see
routers/models.py's `Depends(get_current_user)`), so a same-person
"request" then "approve" step would be theater, not a real control. If
that identity model ever exists, revisit this.

`rollback_promotion()` is the staging/prod counterpart to
adapters/deploy_strategies.py's dev-side rollback — undoes the last
`promote()`/`rollback_promotion()` call for one environment by swapping
back to whatever release was there immediately before it, read from the
`_PREVIOUS_RELEASE_ANNOTATION` every `_upsert_binding()` call records on
the binding it's about to overwrite. One level deep (an undo, not a full
history) — `ProjectReleaseBinding` itself only ever held one pointer, so
this is strictly additive, not a redesign.
"""

from typing import Final, cast

from kubernetes import client, config
from kubernetes.client.exceptions import ApiException

from adapters.interfaces import IPromotionAdapter, PromotionStatus

GROUP: Final[str] = "openchoreo.dev"
VERSION: Final[str] = "v1alpha1"
PLURAL: Final[str] = "projectreleasebindings"

# Mirrors infra/openchoreo/deployment-pipeline.yaml's spec.promotionPaths.
_SOURCE_ENVIRONMENT: Final[dict[str, str]] = {
    "staging": "development",
    "production": "staging",
}
_ENVIRONMENTS: Final[tuple[str, ...]] = ("development", "staging", "production")

# ProjectReleaseBinding is a single pointer with no history of its own —
# this annotation is how rollback_promotion() gets one, recorded on every
# _upsert_binding() call that actually changes the pointer. One level deep
# (an undo, not a full log) — enough for "the last promote to this
# environment was a mistake, put it back," which is the actual emergency
# this exists for.
_PREVIOUS_RELEASE_ANNOTATION: Final[str] = "ai-delivery-portal.io/previous-release"


class OpenChoreoPromotionAdapter(IPromotionAdapter):
    """Scoped to the one real Project this repo has — same single-Project
    scoping as OpenChoreoInferenceAdapter's single-Component scoping, for
    the same reason (no dynamic per-model Project provisioning exists)."""

    def __init__(self, namespace: str = "default", project: str = "telco-fraud-detection"):
        config.load_kube_config()
        self.namespace = namespace
        self.project = project
        self.api = client.CustomObjectsApi()

    def get_promotion_status(self) -> PromotionStatus:
        environments = {env: self._bound_release(env) for env in _ENVIRONMENTS}
        return {
            "project": self.project,
            "component": "serving",
            "environments": environments,
            "prod_pending_approval": (
                environments["staging"] is not None
                and environments["staging"] != environments["production"]
            ),
        }

    def promote(self, target_environment: str) -> PromotionStatus:
        source_environment = _SOURCE_ENVIRONMENT.get(target_environment)
        if source_environment is None:
            raise ValueError(
                f"'{target_environment}' isn't a valid promotion target — "
                f"only {sorted(_SOURCE_ENVIRONMENT)} are, per "
                "infra/openchoreo/deployment-pipeline.yaml's promotionPaths"
            )
        release = self._bound_release(source_environment)
        if release is None:
            raise ValueError(
                f"nothing to promote — '{self.project}' has no release bound in "
                f"'{source_environment}' yet"
            )
        self._upsert_binding(target_environment, release)
        return self.get_promotion_status()

    def rollback_promotion(self, environment: str) -> PromotionStatus:
        binding = self._get_binding(environment)
        if binding is None:
            raise ValueError(
                f"nothing to roll back — '{self.project}' has no release bound in "
                f"'{environment}' yet"
            )
        metadata = cast(dict[str, object], binding.get("metadata") or {})
        annotations = cast(dict[str, object], metadata.get("annotations") or {})
        previous = annotations.get(_PREVIOUS_RELEASE_ANNOTATION)
        if previous is None:
            raise ValueError(f"no prior release recorded for '{environment}' to roll back to")
        self._upsert_binding(environment, str(previous))
        return self.get_promotion_status()

    def _get_binding(self, environment: str) -> dict[str, object] | None:
        try:
            return cast(
                dict[str, object],
                self.api.get_namespaced_custom_object(
                    GROUP, VERSION, self.namespace, PLURAL, self._binding_name(environment)
                ),
            )
        except ApiException as exc:
            if exc.status == 404:
                return None
            raise

    def _bound_release(self, environment: str) -> str | None:
        binding = self._get_binding(environment)
        if binding is None:
            return None
        spec = cast(dict[str, object], binding.get("spec") or {})
        release = spec.get("projectRelease")
        return str(release) if release is not None else None

    def _upsert_binding(self, environment: str, project_release: str) -> None:
        name = self._binding_name(environment)
        current = self._get_binding(environment)
        current_metadata = cast(dict[str, object], (current or {}).get("metadata") or {})
        current_spec = cast(dict[str, object], (current or {}).get("spec") or {})
        current_release = current_spec.get("projectRelease")
        current_annotations = cast(dict[str, str], current_metadata.get("annotations") or {})
        annotations = dict(current_annotations)
        if current_release is not None and str(current_release) != project_release:
            annotations[_PREVIOUS_RELEASE_ANNOTATION] = str(current_release)
        body = {
            "apiVersion": f"{GROUP}/{VERSION}",
            "kind": "ProjectReleaseBinding",
            "metadata": {
                "name": name,
                "labels": {
                    "openchoreo.dev/project": self.project,
                    "openchoreo.dev/environment": environment,
                },
                **({"annotations": annotations} if annotations else {}),
            },
            "spec": {
                "environment": environment,
                "owner": {"projectName": self.project},
                "projectRelease": project_release,
            },
        }
        try:
            self.api.patch_namespaced_custom_object(
                GROUP, VERSION, self.namespace, PLURAL, name, body
            )
        except ApiException as exc:
            if exc.status != 404:
                raise
            self.api.create_namespaced_custom_object(GROUP, VERSION, self.namespace, PLURAL, body)

    def _binding_name(self, environment: str) -> str:
        return f"{self.project}-{environment}"
