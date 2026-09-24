"""Security control catalog + pre-flight scan policy for the golden paths.

The platform is the trust boundary: every golden path declares which of the
four control surfaces it must satisfy, and a run is scanned against that
policy before it executes. The four surfaces mirror the platform-engineering
threat model:

- model-governance  — provenance, signing, approval gates (Model Poisoning)
- data-isolation    — classification, PII/DLP, encryption, tenancy (Data Leaks)
- prompt-security   — input/output guardrails, injection detection (Injection)
- inference-audit   — audit trail, explainability, compliance (Shadow AI)

The scan is deterministic and side-effect-free; the router records one audit
event per scan. Business logic lives here, not in the Portal frontend.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

PILLARS: Final[dict[str, str]] = {
    "model-governance": "Model Registry Governance",
    "data-isolation": "Data Isolation",
    "prompt-security": "Prompt Security",
    "inference-audit": "Inference Audit",
}


@dataclass(frozen=True)
class Control:
    """One selectable security control, keyed to the form param that carries it."""

    id: str
    pillar: str
    label: str
    param: str
    kind: str  # "bool" | "enum" | "text" | "int"
    secure_hint: str


CONTROLS: Final[tuple[Control, ...]] = (
    Control(
        id="provenance-tracking",
        pillar="model-governance",
        label="Provenance tracking",
        param="provenanceTracking",
        kind="bool",
        secure_hint="Record dataset version + code commit with the model version.",
    ),
    Control(
        id="model-signing",
        pillar="model-governance",
        label="Model signing",
        param="modelSigning",
        kind="bool",
        secure_hint="Sign the artifact so a tampered registry entry is detectable.",
    ),
    Control(
        id="approval-gate",
        pillar="model-governance",
        label="Approval gate",
        param="approvalGate",
        kind="enum",
        secure_hint="Require a human approval before a non-dev environment.",
    ),
    Control(
        id="data-classification",
        pillar="data-isolation",
        label="Data classification",
        param="dataClassification",
        kind="enum",
        secure_hint="Tag the data public/internal/confidential/restricted.",
    ),
    Control(
        id="pii-scan",
        pillar="data-isolation",
        label="PII detection & masking",
        param="piiScan",
        kind="bool",
        secure_hint="Scan and mask PII before it reaches a model or an index.",
    ),
    Control(
        id="encryption-at-rest",
        pillar="data-isolation",
        label="Encryption at rest",
        param="encryptionAtRest",
        kind="bool",
        secure_hint="Encrypt the artifact/dataset at rest, not only in transit.",
    ),
    Control(
        id="tenant-isolation",
        pillar="data-isolation",
        label="Tenant isolation",
        param="tenantNamespace",
        kind="text",
        secure_hint="Bind the workload to a tenant namespace boundary.",
    ),
    Control(
        id="input-guardrails",
        pillar="prompt-security",
        label="Input guardrails",
        param="inputGuardrails",
        kind="bool",
        secure_hint="Sanitize and length-bound every prompt before inference.",
    ),
    Control(
        id="output-guardrails",
        pillar="prompt-security",
        label="Output guardrails",
        param="outputGuardrails",
        kind="bool",
        secure_hint="Filter model output for leakage and hallucination markers.",
    ),
    Control(
        id="injection-detection",
        pillar="prompt-security",
        label="Injection detection",
        param="injectionDetection",
        kind="bool",
        secure_hint="Detect prompt-injection in live inference streams.",
    ),
    Control(
        id="context-boundary",
        pillar="prompt-security",
        label="Context boundary",
        param="contextBoundary",
        kind="int",
        secure_hint="Cap the context window so retrieved text can't overrun it.",
    ),
    Control(
        id="audit-logging",
        pillar="inference-audit",
        label="Inference audit trail",
        param="auditLogging",
        kind="bool",
        secure_hint="Record every inference with actor, input hash and output.",
    ),
    Control(
        id="explainability",
        pillar="inference-audit",
        label="Explainability",
        param="explainability",
        kind="bool",
        secure_hint="Attach the reasoning/attribution to each audited inference.",
    ),
    Control(
        id="compliance-report",
        pillar="inference-audit",
        label="Compliance reporting",
        param="complianceReport",
        kind="bool",
        secure_hint="Emit a periodic compliance report from the audit trail.",
    ),
)

CONTROLS_BY_ID: Final[dict[str, Control]] = {c.id: c for c in CONTROLS}


@dataclass(frozen=True)
class Rule:
    """A control a golden path must satisfy, optionally only under a condition."""

    control_id: str
    severity: str  # "blocking" | "warning"
    when: Callable[[dict[str, object]], bool] | None = None


_SENSITIVE_CLASSIFICATIONS: Final[frozenset[str]] = frozenset({"confidential", "restricted"})
_NON_DEV_ENVIRONMENTS: Final[frozenset[str]] = frozenset({"staging", "prod", "production"})


def _sensitive_data(params: dict[str, object]) -> bool:
    return str(params.get("dataClassification", "")).lower() in _SENSITIVE_CLASSIFICATIONS


def _targets_non_dev(params: dict[str, object]) -> bool:
    # `environment` is the deploy-side field; `targetEnvironment` is the
    # promote-side one — accept either so the same rule covers both.
    env = params.get("environment") or params.get("targetEnvironment") or ""
    return str(env).lower() in _NON_DEV_ENVIRONMENTS


# Per golden path: the controls it must satisfy. Blocking rules stop the run;
# warnings are surfaced but let it through. A rule with `when` only applies
# when the predicate holds (e.g. PII scan only matters for sensitive data).
POLICY: Final[dict[str, tuple[Rule, ...]]] = {
    "train-track-register": (
        Rule("provenance-tracking", "blocking"),
        Rule("data-classification", "blocking"),
        Rule("pii-scan", "blocking", when=_sensitive_data),
        Rule("model-signing", "warning"),
        Rule("encryption-at-rest", "warning"),
        Rule("tenant-isolation", "warning"),
        Rule("audit-logging", "warning"),
    ),
    "evaluate-deploy-model": (
        Rule("approval-gate", "blocking", when=_targets_non_dev),
        Rule("provenance-tracking", "warning"),
        Rule("audit-logging", "warning"),
        Rule("explainability", "warning"),
    ),
    "setup-model-monitoring": (
        Rule("audit-logging", "warning"),
        Rule("compliance-report", "warning"),
        Rule("model-signing", "warning"),
    ),
    "llm-serve-deploy": (
        Rule("input-guardrails", "blocking"),
        Rule("output-guardrails", "blocking"),
        Rule("injection-detection", "blocking"),
        Rule("tenant-isolation", "blocking"),
        Rule("audit-logging", "blocking"),
        Rule("context-boundary", "warning"),
        Rule("encryption-at-rest", "warning"),
        Rule("explainability", "warning"),
    ),
    "llm-draft-ingest": (
        Rule("data-classification", "blocking"),
        Rule("pii-scan", "blocking", when=_sensitive_data),
        Rule("injection-detection", "warning"),
        Rule("audit-logging", "warning"),
    ),
    "llm-evaluate-activate": (
        Rule("injection-detection", "blocking"),
        Rule("audit-logging", "warning"),
        Rule("compliance-report", "warning"),
    ),
}


@dataclass(frozen=True)
class Finding:
    control: str
    pillar: str
    severity: str
    message: str


@dataclass(frozen=True)
class ControlStatus:
    id: str
    pillar: str
    label: str
    status: str  # "enforced" | "missing" | "not-applicable"


@dataclass(frozen=True)
class ScanResult:
    passed: bool
    score: int
    findings: tuple[Finding, ...]
    controls: tuple[ControlStatus, ...]


def _satisfied(control: Control, params: dict[str, object]) -> bool:
    value = params.get(control.param)
    if control.kind == "bool":
        return value is True
    if control.kind == "enum":
        return value is not None and str(value).lower() not in ("", "none")
    if control.kind == "int":
        return isinstance(value, (int, float)) and value > 0
    return isinstance(value, str) and value.strip() != ""


def scan(golden_path: str, params: dict[str, object]) -> ScanResult:
    """Evaluate a golden path's security policy against the chosen controls.

    Returns a pass/fail, a 0-100 posture score, the blocking/warning findings,
    and the status of every catalog control (so a UI can render the full
    posture, not just the failures).
    """
    rules = POLICY.get(golden_path, ())
    applicable = [r for r in rules if r.when is None or r.when(params)]

    findings: list[Finding] = []
    satisfied_count = 0
    for rule in applicable:
        control = CONTROLS_BY_ID[rule.control_id]
        if _satisfied(control, params):
            satisfied_count += 1
            continue
        findings.append(
            Finding(
                control=control.id,
                pillar=control.pillar,
                severity=rule.severity,
                message=f"{control.label}: {control.secure_hint}",
            )
        )

    required_ids = {r.control_id for r in applicable}
    controls = tuple(
        ControlStatus(
            id=control.id,
            pillar=control.pillar,
            label=control.label,
            status=(
                "not-applicable"
                if control.id not in required_ids
                else "enforced"
                if _satisfied(control, params)
                else "missing"
            ),
        )
        for control in CONTROLS
    )

    passed = not any(f.severity == "blocking" for f in findings)
    score = round(100 * satisfied_count / len(applicable)) if applicable else 100
    return ScanResult(
        passed=passed,
        score=score,
        findings=tuple(findings),
        controls=controls,
    )
