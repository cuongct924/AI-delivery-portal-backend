"""services/orchestration-api/routers/security.py + security/controls.py —
the pre-flight security scan behind every golden path."""

from audit.events import read_audit_events
from routers.security import (
    SecurityScanRequest,
    list_security_controls,
    scan_security,
)
from security.controls import scan


def _scan(golden_path: str, **params: object):
    return scan(golden_path, params)


def test_scan_passes_when_required_controls_are_set() -> None:
    result = _scan(
        "llm-serve-deploy",
        inputGuardrails=True,
        outputGuardrails=True,
        injectionDetection=True,
        tenantNamespace="tenant-a",
        auditLogging=True,
        contextBoundary=4096,
        encryptionAtRest=True,
        explainability=True,
    )
    assert result.passed is True
    assert result.score == 100
    assert result.findings == ()


def test_scan_blocks_on_a_missing_blocking_control() -> None:
    result = _scan(
        "llm-serve-deploy",
        inputGuardrails=True,
        outputGuardrails=True,
        injectionDetection=True,
        tenantNamespace="tenant-a",
        # auditLogging missing — blocking for serving.
    )
    assert result.passed is False
    assert any(f.control == "audit-logging" and f.severity == "blocking" for f in result.findings)


def test_pii_scan_only_required_for_sensitive_data() -> None:
    public = _scan(
        "train-track-register",
        provenanceTracking=True,
        dataClassification="public",
    )
    assert not any(f.control == "pii-scan" for f in public.findings)

    restricted = _scan(
        "train-track-register",
        provenanceTracking=True,
        dataClassification="restricted",
    )
    assert any(f.control == "pii-scan" and f.severity == "blocking" for f in restricted.findings)


def test_approval_gate_only_required_off_dev() -> None:
    dev = _scan("evaluate-deploy-model", environment="dev")
    assert not any(f.control == "approval-gate" for f in dev.findings)

    prod = _scan("evaluate-deploy-model", environment="prod")
    assert any(f.control == "approval-gate" and f.severity == "blocking" for f in prod.findings)


def test_score_reflects_satisfied_ratio() -> None:
    result = _scan(
        "llm-serve-deploy",
        inputGuardrails=True,
        outputGuardrails=True,
        injectionDetection=True,
        tenantNamespace="tenant-a",
        auditLogging=True,
        contextBoundary=4096,
        encryptionAtRest=True,
        explainability=True,
    )
    assert result.score == 100

    partial = _scan("llm-serve-deploy", inputGuardrails=True)
    assert 0 < partial.score < 100


def test_controls_status_marks_enforced_missing_and_not_applicable() -> None:
    result = _scan(
        "llm-serve-deploy",
        inputGuardrails=True,
        outputGuardrails=True,
        injectionDetection=True,
        tenantNamespace="tenant-a",
        auditLogging=True,
    )
    by_id = {c.id: c.status for c in result.controls}
    assert by_id["input-guardrails"] == "enforced"
    assert by_id["context-boundary"] == "missing"
    # Not part of the serving policy at all.
    assert by_id["pii-scan"] == "not-applicable"


def test_scan_endpoint_records_an_audit_event() -> None:
    before = len(read_audit_events())
    response = scan_security(
        SecurityScanRequest(
            golden_path="llm-serve-deploy",
            stage="run",
            artifact="llama-3-8b",
            params={"inputGuardrails": True},
        )
    )
    assert response.passed is False
    events = read_audit_events()
    assert len(events) == before + 1
    event = events[-1]
    assert event["action"] == "security.scan"
    assert event["category"] == "security"
    assert event["result"] == "denied"
    resource = event["resource"]
    assert isinstance(resource, dict)
    assert resource["name"] == "llama-3-8b"


def test_scan_endpoint_passes_and_records_success() -> None:
    response = scan_security(
        SecurityScanRequest(
            golden_path="llm-evaluate-activate",
            params={"injectionDetection": True},
        )
    )
    assert response.passed is True
    assert read_audit_events()[-1]["result"] == "success"


def test_controls_endpoint_lists_the_catalog() -> None:
    response = list_security_controls()
    ids = {c.id for c in response.controls}
    assert {"provenance-tracking", "pii-scan", "injection-detection", "audit-logging"} <= ids
    assert all(c.pillar_label for c in response.controls)
