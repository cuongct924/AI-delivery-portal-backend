"""Adapter for the versioned dataset object store — MinIO locally (see
docker-compose.yml's `minio` service and `.dvc/config`'s remote "storage"),
any S3-compatible bucket in general.
"""

import os
from pathlib import Path

import boto3
import yaml

from adapters.interfaces import DatasetInfo, IObjectStorageAdapter

_REPO_ROOT = Path(__file__).resolve().parent.parent


class MinioObjectStorageAdapter(IObjectStorageAdapter):
    def __init__(
        self,
        endpoint_url: str | None = None,
        bucket: str | None = None,
        mount_path: str = "/mnt/data",
        local_datasets_path: str | None = None,
    ):
        self.endpoint_url = endpoint_url or os.getenv("MINIO_ENDPOINT_URL", "http://localhost:9000")
        self.bucket = bucket or os.getenv("MINIO_DATASETS_BUCKET", "mlops-datasets")
        self.mount_path = mount_path
        # Same resolution as LocalFileObjectStorageAdapter's root_path —
        # see _dvc_hash_to_relative_path's docstring for why this adapter
        # needs it too.
        self.local_datasets_path = Path(
            local_datasets_path or os.getenv("LOCAL_DATASETS_PATH") or _REPO_ROOT / "data"
        )
        self.client = boto3.client(
            "s3",
            endpoint_url=self.endpoint_url,
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "minioadmin"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin"),
        )

    def _dvc_hash_to_relative_path(self) -> dict[str, str]:
        """Every dataset `.dvc` pointer file under `local_datasets_path`,
        keyed by its md5 content hash.

        `dvc push` uploads objects content-addressed
        (`<remote-prefix>/files/md5/<hash[:2]>/<hash[2:]>`) — an S3
        object's key alone carries no trace of the dataset's real
        filename or directory, unlike a plain `mc cp <file>` upload. The
        only place that mapping still exists is the same `.dvc` pointer
        files `LocalFileObjectStorageAdapter` already reads (each one is
        `{outs: [{md5, path, ...}]}`), so recovering a usable
        name/uri means reading them here too.
        """
        mapping: dict[str, str] = {}
        if not self.local_datasets_path.is_dir():
            return mapping
        for dvc_file in self.local_datasets_path.rglob("*.dvc"):
            try:
                spec = yaml.safe_load(dvc_file.read_text())
                out = spec["outs"][0]
            except Exception:
                continue
            relative_dir = dvc_file.parent.relative_to(self.local_datasets_path)
            mapping[out["md5"]] = str(relative_dir / out["path"])
        return mapping

    def list_datasets(self, prefix: str = "") -> list[DatasetInfo]:
        response = self.client.list_objects_v2(Bucket=self.bucket, Prefix=prefix)
        hash_to_path = self._dvc_hash_to_relative_path()
        datasets: list[DatasetInfo] = []
        for obj in response.get("Contents", []):
            # Parse the content-addressed key backwards: ".../files/md5/
            # <hash[:2]>/<hash[2:]>". Anything else (a stray object, or one
            # whose .dvc pointer isn't checked out locally) has no
            # recoverable filename and no local mount path a training pod
            # could actually open — skip it rather than surface a
            # dataset picker entry that 404s the moment someone picks it.
            key_parts = obj["Key"].rsplit("/", 2)
            if len(key_parts) != 3 or key_parts[0].rsplit("/", 1)[-1] != "md5":
                continue
            relative_path = hash_to_path.get(key_parts[1] + key_parts[2])
            if relative_path is None:
                continue
            datasets.append(
                DatasetInfo(
                    name=relative_path,
                    # Same mount-path convention as LocalFileObjectStorageAdapter
                    # — the training pod's hostPath mount, keyed off the
                    # dataset's real relative path now, not the S3 key.
                    uri=f"file://{self.mount_path}/{relative_path}",
                    size_bytes=obj["Size"],
                    source="s3",
                )
            )
        return datasets
