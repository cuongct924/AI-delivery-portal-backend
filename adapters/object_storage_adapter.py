"""Adapter for the versioned dataset object store — MinIO locally (see
docker-compose.yml's `minio` service and `.dvc/config`'s remote "storage"),
any S3-compatible bucket in general.
"""

import os

import boto3

from adapters.interfaces import DatasetInfo, IObjectStorageAdapter


class MinioObjectStorageAdapter(IObjectStorageAdapter):
    def __init__(
        self,
        endpoint_url: str | None = None,
        bucket: str | None = None,
        mount_path: str = "/mnt/data",
    ):
        self.endpoint_url = endpoint_url or os.getenv("MINIO_ENDPOINT_URL", "http://localhost:9000")
        self.bucket = bucket or os.getenv("MINIO_DATASETS_BUCKET", "mlops-datasets")
        self.mount_path = mount_path
        self.client = boto3.client(
            "s3",
            endpoint_url=self.endpoint_url,
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "minioadmin"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin"),
        )

    def list_datasets(self, prefix: str = "") -> list[DatasetInfo]:
        response = self.client.list_objects_v2(Bucket=self.bucket, Prefix=prefix)
        return [
            DatasetInfo(
                name=obj["Key"],
                # The training pod reads from the kind cluster's hostPath
                # mount, not straight from the bucket — see this module's
                # docstring — so the picker must hand back a path the
                # WorkflowTemplate can actually open, keyed off the same
                # basename the object was pushed under.
                uri=f"file://{self.mount_path}/{obj['Key'].rsplit('/', 1)[-1]}",
                size_bytes=obj["Size"],
                source="s3",
            )
            for obj in response.get("Contents", [])
        ]
