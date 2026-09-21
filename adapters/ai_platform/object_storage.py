"""Adapters for the dataset picker's two object-storage sources, plus the
fan-out wrapper `factory.get_object_storage_adapter()` composes them with.
Merged into one file — all three are small and tightly coupled (Composite
wraps the other two directly), not a case of duplicated logic.
"""

import logging
import os
from pathlib import Path

import boto3
import yaml

from adapters.ai_platform.interfaces import DatasetInfo, IObjectStorageAdapter

# 3 parents: adapters/ai_platform/object_storage.py -> ai_platform -> adapters -> repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]

logger = logging.getLogger(__name__)


class LocalFileObjectStorageAdapter(IObjectStorageAdapter):
    """Datasets already checked out on the local filesystem — DVC's
    working-tree copy under `data/` (see .dvc/config's remote "storage").
    Works both for a dev machine running the orchestration API directly (no
    MinIO needed) and for the orchestration-api pod, which bind-mounts
    `./data` read-only via a hostPath volume — anywhere else `data/` doesn't
    exist on disk, so `list_datasets` just returns `[]`, the same fail-open
    contract `CompositeObjectStorageAdapter` relies on.
    """

    def __init__(self, root_path: str | None = None, mount_path: str = "/mnt/data"):
        self.root_path = Path(root_path or os.getenv("LOCAL_DATASETS_PATH") or _REPO_ROOT / "data")
        # k3d training pod sees these via hostPath mount, not this process's filesystem.
        self.mount_path = mount_path

    def list_datasets(self, prefix: str = "") -> list[DatasetInfo]:
        if not self.root_path.is_dir():
            return []
        datasets: list[DatasetInfo] = []
        for path in sorted(self.root_path.rglob(f"{prefix}*")):
            # A `.dvc` pointer file next to it marks it as tracked.
            if not path.is_file() or not Path(f"{path}.dvc").exists():
                continue
            relative_path = path.relative_to(self.root_path)
            datasets.append(
                DatasetInfo(
                    name=str(relative_path),
                    uri=f"file://{self.mount_path}/{relative_path.as_posix()}",
                    size_bytes=path.stat().st_size,
                    source="local",
                )
            )
        return datasets


class MinioObjectStorageAdapter(IObjectStorageAdapter):
    """The versioned dataset object store — MinIO in the AI Platform zone
    (infra/ai-platform-zone/minio.yaml and `.dvc/config`'s remote "storage"),
    any S3-compatible bucket in general.
    """

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
        # Same resolution as LocalFileObjectStorageAdapter's root_path — see
        # _dvc_hash_to_relative_path below.
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
        """Every `.dvc` pointer file under `local_datasets_path`, keyed by its
        md5 content hash.

        `dvc push` uploads objects content-addressed
        (`<remote-prefix>/files/md5/<hash[:2]>/<hash[2:]>`) — the S3 key alone
        carries no trace of the dataset's filename/directory. The only place
        that mapping still exists is the `.dvc` pointer files, each one
        `{outs: [{md5, path, ...}]}`.
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
            # Skip keys with no matching .dvc pointer — avoids a 404'ing picker entry.
            key_parts = obj["Key"].rsplit("/", 2)
            if len(key_parts) != 3 or key_parts[0].rsplit("/", 1)[-1] != "md5":
                continue
            relative_path = hash_to_path.get(key_parts[1] + key_parts[2])
            if relative_path is None:
                continue
            datasets.append(
                DatasetInfo(
                    name=relative_path,
                    # Same mount-path convention as LocalFileObjectStorageAdapter's hostPath mount.
                    uri=f"file://{self.mount_path}/{relative_path}",
                    size_bytes=obj["Size"],
                    source="s3",
                )
            )
        return datasets


class CompositeObjectStorageAdapter(IObjectStorageAdapter):
    def __init__(self, adapters: list[IObjectStorageAdapter]):
        self.adapters = adapters

    def list_datasets(self, prefix: str = "") -> list[DatasetInfo]:
        datasets: list[DatasetInfo] = []
        for adapter in self.adapters:
            try:
                datasets.extend(adapter.list_datasets(prefix))
            except Exception:
                # One source down (e.g. MinIO) shouldn't hide datasets others can list.
                logger.warning("%s failed to list datasets", type(adapter).__name__, exc_info=True)
        return datasets
