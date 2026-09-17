# data

Datasets tracked with [DVC](https://dvc.org) — actual files live in an S3-compatible
remote (`minio` service in `docker-compose.yml`, self-hosted for local dev; swap in
real S3 credentials later if needed, only `.env`'s `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`
change, `.dvc/config` stays the same).

Only the small `.dvc` pointer files (md5 hash + size) are committed to git —
the real data goes to the `storage` remote configured in `.dvc/config`.

Split by which golden path / training path each dataset demos —
`ranking-package-recommendation-ranking/` is a separate Golden Path
(`routers/recommendations.py`, its own docstring calls it "Golden Path #3":
different dataset contract and training trigger shape from Golden Path #1,
even though it also trains via classical algorithms, not Deep Learning);
`deep-learning/` is Golden Path #1's `architecture=mlp/lstm/nlp/cv` (needs
`sequenceLength`/`hiddenLayers`/..., none of which `sklearn` does).

Golden Path #1's `architecture=sklearn` datasets each get their own
`<taskType>-<useCase>/` directory (one dataset per directory) instead of a
shared `traditional-ml/` — named to match the `train-track-register`
Scaffolder template's `useCase` field, so it's obvious at a glance which
demo a directory is for.

## Golden Path #1, architecture=sklearn — one directory per (taskType, useCase)

| Directory | File | Task type | Notes |
|---|---|---|---|
| `classification-telco-fraud-detection/` | `telco-fraud-detection-sample.csv` | classification | has an ID column (`transaction_id`) — pass it as `idColumns` |
| `regression-house-price-prediction/` | `house-price-sample.csv` | regression | no ID column |
| `clustering-customer-segmentation/` | `customer-segmentation-sample.csv` | clustering | 3 synthetic segments, no target column |
| `anomaly-detection-network-anomaly-detection/` | `network-anomaly-sample.csv` | anomaly-detection | `is_anomaly` column is optional — leave `targetColumn` empty for a purely unsupervised run, set it to get precision/recall/f1 too |
| `regression-revenue-forecast/` | `revenue-forecast-sample.csv` | regression | `date`/`region`/`revenue`, 2 regions × 365 days — set `timeColumn=date` for a chronological split instead of a random one |

Clustering test runs reuse the classification or regression file with
`targetColumn` left empty, same as before.

## `deep-learning/` — architecture=mlp/lstm/nlp/cv (Golden Path #1)

| File | Task type | Notes |
|---|---|---|
| `sensor-timeseries-sample.csv` | regression | synthetic (trend+seasonality+noise, numpy, not downloaded), 500 rows, `timestamp` column — `classification-telco-fraud-detection/`/`regression-house-price-prediction/` are too small (~10-15 rows) for MLP/LSTM to learn a real signal, and `timeColumn=timestamp` is required for `architecture=lstm`'s sequence windowing |
| `shapes-sample.zip` | classification | synthetic geometric shapes (circle/square/triangle/...) for `architecture=cv` — image classification |

## `ranking-package-recommendation-ranking/` — Golden Path #3 (`routers/recommendations.py`)

| File | Task type | Notes |
|---|---|---|
| `ranking-sample.csv` | ranking | 119 rows, 30 users × 20 items. Learning-to-rank sample: `query_id` holds user ids, `item_id` the items, `relevance` the graded 0-3 label (pass it as `ratingColumn` for SVD/KNN); also has `price`/`category_score`/`past_purchases`. No event-time column, so `train_rec.py`'s temporal split takes `timestampColumn=price` as an ordering key. Content-based (`tfidf_cosine`) additionally needs an item-features file — not shipped, pass your own `itemFeaturesUri`. |

`recommend-train-register`'s Scaffolder defaults point at this file via
`file:///mnt/data/ranking-package-recommendation-ranking/ranking-sample.csv`
— the same path `orchestration-api` reads (`docker-compose.yml` mounts
`./data` at `/mnt/data`) and the k3d training pod's `hostPath` exposes.

## `future-pipelines/` — reference shapes, not wired to anything yet

Not DVC-tracked (no `.dvc` files, committed directly) and not listed by
`LocalFileObjectStorageAdapter` on purpose — see its own README for what
each subdirectory is for and why.

## First time (pull the demo datasets)

```bash
docker compose up -d minio
# create the bucket once — minio doesn't auto-create it
docker run --rm --network host minio/mc alias set local http://localhost:9000 minioadmin minioadmin
docker run --rm --network host minio/mc mb local/mlops-datasets
.venv/bin/dvc push   # uploads every tracked dataset to the remote
.venv/bin/dvc pull   # (on another machine) downloads them back
```

## Adding a new dataset version

```bash
.venv/bin/dvc add data/<directory>/<file>
git add data/<directory>/<file>.dvc data/<directory>/.gitignore
.venv/bin/dvc push
```

`dvc add` prints an md5 hash into the resulting `data/.../<file>.dvc` — the
training step embeds that hash in a `mlflow.data` `Dataset`'s digest/name and
calls `mlflow.log_input()` on it, so `IModelRegistryAdapter.get_dataset_lineage()`
(`adapters/mlflow_adapter.py`) can trace a model version back to the exact
dataset file(s) that trained it — a run can log more than one.
