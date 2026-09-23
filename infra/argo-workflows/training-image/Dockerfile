# Multi-stage — deps resolved in `builder`, only site-packages ship in the
# final image.
FROM python:3.12-slim AS builder
WORKDIR /app
RUN pip install --no-cache-dir uv
# Copied before app code so this slow layer stays cached across *.py edits.
COPY infra/argo-workflows/training-image/requirements.lock.txt .
# unsafe-best-match: lock file trusts both PyPI and the PyTorch CPU index.
RUN uv pip install --system --no-cache --index-strategy unsafe-best-match \
    --target /deps -r requirements.lock.txt

# checkov:skip=CKV_DOCKER_2: batch job, no long-running process to health-check.
FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get upgrade -y \
    # libgomp1: LightGBM needs it at import.
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /deps /usr/local/lib/python3.12/site-packages
COPY infra/argo-workflows/training-image/*.py .
RUN useradd --create-home --uid 1000 appuser
USER appuser
ENV PYTHONPATH=/app
CMD ["python", "train.py"]
