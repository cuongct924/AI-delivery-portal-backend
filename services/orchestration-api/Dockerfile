# Multi-stage — dependencies resolved in `builder`, only the installed
# site-packages copied into the final image; uv and pip's build cache never
# ship in the final image.
FROM python:3.12-slim AS builder
WORKDIR /app
RUN pip install --no-cache-dir uv
# Copied before the app code so this layer (the slow one) is cache-hit on
# every rebuild that only touches app code, not either requirements.lock.txt.
# adapters/ has its own separately-locked requirements.lock.txt (it's a
# standalone installable unit per CLAUDE.md's SERVICE_REQS convention) —
# both are resolved together in one `uv pip install` call, not two separate
# ones, so uv's resolver can catch/report a real version conflict between
# them instead of the second install silently overwriting the first's files.
COPY services/orchestration-api/requirements.lock.txt orchestration-api-requirements.lock.txt
COPY adapters/requirements.lock.txt adapters-requirements.lock.txt
COPY services/orchestration-api/container-overrides.txt .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --target /deps \
        -r orchestration-api-requirements.lock.txt \
        -r adapters-requirements.lock.txt \
        --overrides container-overrides.txt

FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get upgrade -y && rm -rf /var/lib/apt/lists/*
COPY --from=builder /deps /usr/local/lib/python3.12/site-packages
# `uv pip install --target` lays out console-script entry points (uvicorn
# et al.) under <target>/bin, not <target> itself — that landed at
# .../site-packages/bin, off $PATH, so `CMD ["uvicorn", ...]` below
# couldn't find it. Put it on PATH instead of restructuring the COPY above.
ENV PATH="/usr/local/lib/python3.12/site-packages/bin:${PATH}"
RUN useradd --create-home --uid 1000 appuser
# routers/models.py and evaluations/llm_judge.py import from adapters/, which
# lives outside this service's own directory — needs repo-root build context.
COPY --chown=appuser:appuser adapters/ ./adapters/
# get_feature_store_adapter() constructs FeastAdapter() eagerly at import
# time (routers/models.py module scope, not lazily per-request), and it
# defaults to the relative path "infra/feature-store" (FEAST_REPO_PATH env
# var overrides it) — without this, the app can't even boot. The 2 .db
# files here are baked in as a frozen snapshot as of image build time, not
# shared live with the host's own infra/feature-store/ (which the local
# dev `uvicorn` process keeps mutating) — fine for this container's own
# purposes, not a live sync.
COPY --chown=appuser:appuser infra/feature-store/ ./infra/feature-store/
COPY --chown=appuser:appuser services/orchestration-api/ .
USER appuser
EXPOSE 8000
ENV PYTHONPATH=/app
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz')" || exit 1
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
