"""Adapters for the dataset picker's two object-storage sources, plus the
fan-out wrapper `factory.get_object_storage_adapter()` composes them with.
Merged into one file — all three are small and tightly coupled (Composite
wraps the other two directly), not a case of duplicated logic.
"""

import logging
import os
from pathlib import Path

import boto3

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
    """The AI Platform zone's MinIO (infra/ai-platform-zone/minio.yaml) — a
    filesystem gateway over the repo's own `./data`, not a `dvc push` target.
    scripts/setup-3node-infra.sh `docker cp`s `./data`'s contents onto
    worker2's hostPath as-is (see that manifest's own top comment), so every
    top-level dataset-category folder under `data/` shows up as its own
    bucket, holding the same files — and `.dvc` pointers — under their real
    names. Never DVC's content-addressed `files/md5/<hash>` keys: nothing
    ever runs `dvc push` against this MinIO, so those keys would never exist
    here even though `.dvc/config` still names it as a remote.
    """

    def __init__(self, endpoint_url: str | None = None, mount_path: str = "/mnt/data"):
        self.endpoint_url = endpoint_url or os.getenv("MINIO_ENDPOINT_URL", "http://localhost:9000")
        self.mount_path = mount_path
        self.client = boto3.client(
            "s3",
            endpoint_url=self.endpoint_url,
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "minioadmin"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin"),
        )

    def list_datasets(self, prefix: str = "") -> list[DatasetInfo]:
        datasets: list[DatasetInfo] = []
        for bucket in self.client.list_buckets().get("Buckets", []):
            bucket_name = bucket["Name"]
            response = self.client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
            contents = response.get("Contents", [])
            keys = {obj["Key"] for obj in contents}
            for obj in contents:
                key = obj["Key"]
                # Same "has a sibling .dvc pointer" convention as
                # LocalFileObjectStorageAdapter — marks it as a tracked dataset.
                if key.endswith(".dvc") or f"{key}.dvc" not in keys:
                    continue
                relative_path = f"{bucket_name}/{key}"
                datasets.append(
                    DatasetInfo(
                        name=relative_path,
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
