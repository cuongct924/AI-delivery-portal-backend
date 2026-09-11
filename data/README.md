# data

Datasets tracked with [DVC](https://dvc.org) — actual files live in an S3-compatible
remote (`minio` service in `docker-compose.yml`, self-hosted for local dev; swap in
real S3 credentials later if needed, only `.env`'s `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`
change, `.dvc/config` stays the same).

Only the small `.dvc` pointer files (md5 hash + size) are committed to git —
the real data goes to the `storage` remote configured in `.dvc/config`.

Split into 3 directories by which golden path / training path each dataset
demos — `recsys/` is a separate Golden Path (`routers/recommendations.py`,
its own docstring calls it "Golden Path #3": different dataset contract and
training trigger shape from Golden Path #1, even though it also trains via
classical algorithms, not Deep Learning); `traditional-ml/` vs
`deep-learning/` splits Golden Path #1 by `architecture` on
`TriggerTrainingRequest` — classical `sklearn` needs none of the
DL-specific fields (`sequenceLength`, `hiddenLayers`, ...), MLP/LSTM/NLP/CV do.

## `traditional-ml/` — architecture=sklearn (Golden Path #1)

| File | Task type | Notes |
|---|---|---|
| `fraud-detection-sample.csv` | classification | has an ID column (`transaction_id`) — pass it as `idColumns` |
| `house-price-sample.csv` | regression | no ID column |

Clustering test runs reuse either file with `targetColumn` left empty.

## `deep-learning/` — architecture=mlp/lstm/nlp/cv (Golden Path #1)

| File | Task type | Notes |
|---|---|---|
| `sensor-timeseries-sample.csv` | regression | synthetic (trend+seasonality+noise, numpy, not downloaded), 500 rows, `timestamp` column — the `traditional-ml/` datasets are too small (~10-15 rows) for MLP/LSTM to learn a real signal, and `timeColumn=timestamp` is required for `architecture=lstm`'s sequence windowing |
| `shapes-sample.zip` | classification | synthetic geometric shapes (circle/square/triangle/...) for `architecture=cv` — image classification |

## `recsys/` — Golden Path #3 (`routers/recommendations.py`)

| File | Task type | Notes |
|---|---|---|
| `interactions-sample.csv` | ranking | 30 users × interactions (Pareto-distributed) — collaborative algorithms (SVD/KNN) use this alone |
| `item-features-sample.csv` | ranking | 20 items with features — combined with `interactions-sample.csv` for content-based algorithms (TF-IDF cosine) |

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
.venv/bin/dvc add data/<traditional-ml|deep-learning|recsys>/<file>
git add data/<traditional-ml|deep-learning|recsys>/<file>.dvc data/<traditional-ml|deep-learning|recsys>/.gitignore
.venv/bin/dvc push
```

`dvc add` prints an md5 hash into the resulting `data/.../<file>.dvc` — the
training step embeds that hash in a `mlflow.data` `Dataset`'s digest/name and
calls `mlflow.log_input()` on it, so `IModelRegistryAdapter.get_dataset_lineage()`
(`adapters/mlflow_adapter.py`) can trace a model version back to the exact
dataset file(s) that trained it — a run can log more than one.
