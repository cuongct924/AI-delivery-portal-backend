VENV := .venv
PYTHON312 := python3.12
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
UV := $(VENV)/bin/uv
RUFF := $(VENV)/bin/ruff
PYRIGHT := $(VENV)/bin/pyright
PYTEST := $(VENV)/bin/pytest
DVC := $(VENV)/bin/dvc

# Auto-loads .env (AWS_ACCESS_KEY_ID, KEYCLOAK_URL, ...) into every recipe's
# environment — no more manually exporting before dvc/uvicorn/etc.
ifneq (,$(wildcard .env))
include .env
export
endif

# Each service's requirements.txt -> its own requirements.lock.txt; not
# required to agree with each other.
SERVICE_REQS := requirements-dev.txt \
	adapters/requirements.txt \
	services/orchestration-api/requirements.txt \
	agents/mcp-servers/ai-observability-server/requirements.txt \
	agents/mcp-servers/llmops-golden-paths-server/requirements.txt \
	agents/mcp-servers/golden-path-guide-server/requirements.txt \
	agents/mcp-servers/mlops-golden-paths-server/requirements.txt \
	infra/argo-workflows/training-image/requirements.txt

.PHONY: venv install lock hooks lint format format-check typecheck test check \
	gitleaks checkov trivy security \
	run-orchestration-api run-ai-observability-mcp run-llmops-golden-paths-mcp run-golden-path-guide-mcp \
	run-mlops-golden-paths-mcp \
	dvc-pull dvc-push \
	clean-venv

# Pinned to 3.12 to match the Dockerfile base image. Install: brew install python@3.12
venv:
	$(PYTHON312) -m venv $(VENV)
	$(PIP) install --upgrade pip uv -q

## Installs the merged dev.lock.txt into the shared local .venv (not used by
## any Dockerfile) and the pre-commit hook.
install: venv
	$(UV) pip install -r dev.lock.txt
	$(VENV)/bin/pre-commit install

## Regenerates dev.lock.txt (--universal, any dev OS) and each service's own
## *.lock.txt (pinned linux/x86_64, matches Docker/CI). --index-strategy
## unsafe-best-match + --emit-index-url: needed for torch's extra index to
## resolve without breaking unrelated packages. Commit the *.lock.txt diff.
lock: venv
	$(UV) pip compile $(SERVICE_REQS) --python-version 3.12 --universal --index-strategy unsafe-best-match --emit-index-url -o dev.lock.txt
	@for f in $(SERVICE_REQS); do \
		echo "Locking $$f..."; \
		$(UV) pip compile "$$f" --python-version 3.12 --python-platform x86_64-unknown-linux-gnu --index-strategy unsafe-best-match --emit-index-url -o "$${f%.txt}.lock.txt"; \
	done

hooks:
	$(VENV)/bin/pre-commit install

lint:
	$(RUFF) check .

format:
	$(RUFF) format .

## Non-mutating version of `format`, for CI — fails instead of rewriting files.
format-check:
	$(RUFF) format --check .

typecheck:
	$(PYRIGHT)

test:
	$(PYTEST)

check: lint format-check typecheck test

# Security scanners — same as CI's jobs, run from native CLIs (faster, no
# Docker daemon needed). Install once: brew install gitleaks checkov trivy

## Secret scanning — scans full git history, same as CI.
gitleaks:
	gitleaks detect --source . --redact -v

## IaC misconfig scan — same framework/skip-path as CI.
checkov:
	checkov --directory . --framework dockerfile,argo_workflows,github_actions \
		--skip-path .venv --compact --quiet

## Builds and Trivy-scans each service image with its own Dockerfile —
## needs Docker running. golden-path-guide-server predates this target and
## was never added; not this rename's concern to fix.
trivy:
	docker build -t orchestration-api:local -f services/orchestration-api/Dockerfile .
	docker build -t ai-observability-server:local -f agents/mcp-servers/ai-observability-server/Dockerfile .
	docker build -t llmops-golden-paths-server:local -f agents/mcp-servers/llmops-golden-paths-server/Dockerfile .
	docker build -t mlops-golden-paths-server:local -f agents/mcp-servers/mlops-golden-paths-server/Dockerfile .
	docker build -t training-image:local -f infra/argo-workflows/training-image/Dockerfile .
	@for img in orchestration-api ai-observability-server llmops-golden-paths-server mlops-golden-paths-server training-image; do \
		echo "=== Trivy scan: $$img ==="; \
		trivy image --severity CRITICAL,HIGH --ignore-unfixed --exit-code 1 "$$img:local"; \
	done

## Everything security — secrets + IaC + image CVEs.
security: gitleaks checkov trivy

run-orchestration-api:
	cd services/orchestration-api && PYTHONPATH=$(CURDIR) $(CURDIR)/$(PY) -m uvicorn main:app --reload

run-ai-observability-mcp:
	bash scripts/run-mcp-local.sh ai-observability

run-llmops-golden-paths-mcp:
	bash scripts/run-mcp-local.sh llmops-golden-paths

run-golden-path-guide-mcp:
	bash scripts/run-mcp-local.sh golden-path-guide

run-mlops-golden-paths-mcp:
	bash scripts/run-mcp-local.sh mlops-golden-paths

## Pulls/pushes the DVC-tracked dataset against the local MinIO remote.
dvc-pull:
	$(DVC) pull

dvc-push:
	$(DVC) push

clean-venv:
	rm -rf $(VENV)
