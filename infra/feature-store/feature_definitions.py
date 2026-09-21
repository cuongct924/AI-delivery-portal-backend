"""Feast feature repo definitions — `adapters/ai_platform/feature_store_adapter.py`'s
`FeastAdapter` connects here by default (`FEAST_REPO_PATH=infra/feature-store`).

`join_keys=["entity_id"]` is not a naming choice — it matches the column
name FeastAdapter itself hardcodes in its entity_df/entity_rows, so any
feature view added here must key on that same column to be reachable
through the adapter. If a real dataset's identifier column has a
different name, remap it via FileSource(field_mapping=...) here — never
in the adapter.
"""

import os
from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64, String
from feast.value_type import ValueType

entity = Entity(name="entity", join_keys=["entity_id"], value_type=ValueType.STRING)

# `or`, not .get() default — docker-compose passes a blank var as "", not unset.
# Unset/blank falls back to local parquet; real deploys set both FEAST_* vars to s3://.
transaction_features_source = FileSource(
    name="transaction_features_source",
    path=os.environ.get("FEAST_OFFLINE_STORE_PATH") or "data/transaction_features.parquet",
    s3_endpoint_override=os.environ.get("FEAST_S3_ENDPOINT_URL") or None,
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

transaction_features = FeatureView(
    name="transaction_features",
    entities=[entity],
    ttl=timedelta(days=365),
    schema=[
        Field(name="amount", dtype=Float32),
        Field(name="merchant_category", dtype=String),
        Field(name="hour_of_day", dtype=Int64),
    ],
    online=True,
    source=transaction_features_source,
    tags={"golden_path": "fraud-detection"},
)
