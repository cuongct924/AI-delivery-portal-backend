# Multi-stage — dependencies resolved in `builder`, only the installed
# site-packages copied into the final image; uv and pip's build cache never
# ship in the final image.
FROM python:3.12-slim AS builder
WORKDIR /app
RUN pip install --no-cache-dir uv
# adapters/ has its own separately-locked requirements.lock.txt — resolved
# together in one `uv pip install` call so a real version conflict between
# the two fails loudly instead of one silently overwriting the other's files.
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
# `uv pip install --target` puts console scripts (uvicorn et al.) under
# <target>/bin, off $PATH by default.
ENV PATH="/usr/local/lib/python3.12/site-packages/bin:${PATH}"
RUN useradd --create-home --uid 1000 appuser
# adapters/ lives outside this service's own directory — needs repo-root context.
COPY --chown=appuser:appuser adapters/ ./adapters/
# FeastAdapter() is constructed eagerly at import time and defaults to this
# relative path — the app can't boot without it. Frozen snapshot, not a
# live sync with the host's own infra/feature-store/.
COPY --chown=appuser:appuser infra/feature-store/ ./infra/feature-store/
COPY --chown=appuser:appuser services/orchestration-api/ .
USER appuser
EXPOSE 8000
ENV PYTHONPATH=/app
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz')" || exit 1
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
