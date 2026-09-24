"""Security control-surface API — the pre-flight scan behind every golden
path's `orchestration:security-scan` step and the wizard's live posture panel.

The platform is the trust boundary: a run is scanned against the four control
surfaces (Model Registry Governance, Data Isolation, Prompt Security,
Inference Audit) before it executes, and one audit event is recorded per scan
so the Inference Audit surface has a trail even for runs that never ship.
"""

from audit.events import record_audit_event
from auth.thunder import get_current_user
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from security.controls import CONTROLS, PILLARS, scan

router = APIRouter(prefix="/security", tags=["security"])


class SecurityScanRequest(BaseModel):
    golden_path: str
    stage: str = "run"
    artifact: str = ""
    params: dict[str, object] = {}


class SecurityFindingResponse(BaseModel):
    control: str
    pillar: str
    severity: str
    message: str


class SecurityControlStatusResponse(BaseModel):
    id: str
    pillar: str
    label: str
    status: str


class SecurityScanResponse(BaseModel):
    passed: bool
    score: int
    findings: list[SecurityFindingResponse]
    controls: list[SecurityControlStatusResponse]


class SecurityControlResponse(BaseModel):
    id: str
    pillar: str
    pillar_label: str
    label: str
    param: str
    kind: str
    secure_hint: str


class SecurityControlsResponse(BaseModel):
    controls: list[SecurityControlResponse]


@router.get("/controls", response_model=SecurityControlsResponse)
def list_security_controls(
    user: dict = Depends(get_current_user),
) -> SecurityControlsResponse:
    """The full control catalog, so a UI can render every surface even before
    a scan runs."""
    del user
    return SecurityControlsResponse(
        controls=[
            SecurityControlResponse(
                id=c.id,
                pillar=c.pillar,
                pillar_label=PILLARS[c.pillar],
                label=c.label,
                param=c.param,
                kind=c.kind,
                secure_hint=c.secure_hint,
            )
            for c in CONTROLS
        ]
    )


@router.post("/scan", response_model=SecurityScanResponse)
def scan_security(
    request: SecurityScanRequest, user: dict = Depends(get_current_user)
) -> SecurityScanResponse:
    """Scan a golden-path run against its security policy.

    Blocking findings mean the run should stop; warnings are advisory. The
    scan itself is recorded as an audit event (category "security") so the
    Inference Audit surface captures the decision either way.
    """
    result = scan(request.golden_path, request.params)
    # `user` is a plain dict through FastAPI but the `Depends` sentinel when a
    # test calls this function directly.
    actor = (
        user.get("sub") or user.get("preferred_username") or "unknown"
        if isinstance(user, dict)
        else "unknown"
    )
    record_audit_event(
        action="security.scan",
        category="security",
        result="success" if result.passed else "denied",
        resource={
            "type": "GoldenPath",
            "name": request.artifact or request.golden_path,
            "golden_path": request.golden_path,
            "stage": request.stage,
        },
        metadata={
            "score": result.score,
            "blocking": sum(1 for f in result.findings if f.severity == "blocking"),
            "warnings": sum(1 for f in result.findings if f.severity == "warning"),
            "actor": actor,
        },
    )
    return SecurityScanResponse(
        passed=result.passed,
        score=result.score,
        findings=[
            SecurityFindingResponse(
                control=f.control,
                pillar=f.pillar,
                severity=f.severity,
                message=f.message,
            )
            for f in result.findings
        ],
        controls=[
            SecurityControlStatusResponse(id=c.id, pillar=c.pillar, label=c.label, status=c.status)
            for c in result.controls
        ],
    )
