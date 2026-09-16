"""Mock adapter for IPromotionAdapter — in-memory stand-in for OpenChoreo's
ProjectReleaseBinding-based promotion. Same "development" seed as a demo
run would have after a real deploy (see MockWorkflowAdapter/
MockInferenceAdapter's own seeded-state precedent)."""

from adapters.interfaces import IPromotionAdapter, PromotionStatus

_SOURCE_ENVIRONMENT: dict[str, str] = {
    "staging": "development",
    "production": "staging",
}


class MockPromotionAdapter(IPromotionAdapter):
    def __init__(self, project: str = "telco-fraud-detection") -> None:
        self.project = project
        self._bindings: dict[str, str] = {"development": "1"}
        # Mirrors OpenChoreoPromotionAdapter's annotation-based undo —
        # same one-level-deep contract, just a plain dict here since there's
        # no real CR to annotate.
        self._previous: dict[str, str] = {}

    def get_promotion_status(self) -> PromotionStatus:
        environments = {
            env: self._bindings.get(env) for env in ("development", "staging", "production")
        }
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
                f"only {sorted(_SOURCE_ENVIRONMENT)} are"
            )
        release = self._bindings.get(source_environment)
        if release is None:
            raise ValueError(
                f"nothing to promote — '{self.project}' has no release bound in "
                f"'{source_environment}' yet"
            )
        self._set_binding(target_environment, release)
        return self.get_promotion_status()

    def rollback_promotion(self, environment: str) -> PromotionStatus:
        if environment not in self._bindings:
            raise ValueError(
                f"nothing to roll back — '{self.project}' has no release bound in "
                f"'{environment}' yet"
            )
        previous = self._previous.get(environment)
        if previous is None:
            raise ValueError(f"no prior release recorded for '{environment}' to roll back to")
        self._set_binding(environment, previous)
        return self.get_promotion_status()

    def _set_binding(self, environment: str, release: str) -> None:
        current = self._bindings.get(environment)
        if current is not None and current != release:
            self._previous[environment] = current
        self._bindings[environment] = release
