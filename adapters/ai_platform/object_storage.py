"""Adapters for the dataset picker's two object-storage sources, plus the
fan-out wrapper `factory.get_object_storage_adapter()` composes them with.
Merged into one file — all three are small and tightly coupled (Composite
wraps the other two directly), not a case of duplicated logic.
"""

import logging
import os
from functools import lru_cache
from pathlib import Path

import boto3
from botocore.config import Config

from adapters.ai_platform.interfaces import DatasetInfo, IObjectStorageAdapter

# 3 parents: adapters/ai_platform/object_storage.py -> ai_platform -> adapters -> repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]

# The path the training pod sees datasets at (hostPath mount) — see
# LocalFileObjectStorageAdapter. Overridable for a non-default mount.
_DEFAULT_MOUNT_PATH = "/mnt/data"

logger = logging.getLogger(__name__)


def _local_data_root() -> Path:
    return Path(os.getenv("LOCAL_DATASETS_PATH") or _REPO_ROOT / "data")


def resolve_dataset_path(uri: str) -> Path:
    """Resolves a dataset URI to a path *this* process can actually read.

    Datasets are addressed by the path the training pod sees
    (`file:///mnt/data/<name>`, see LocalFileObjectStorageAdapter) — but the
    orchestration-api also runs directly on a dev host, where `/mnt/data`
    doesn't exist and the same files live under the repo's `data/`. Falls
    back to that repo-relative path when the mount path isn't present, so
    /datasets/validate|columns|preview|enrich-features work in both places
    instead of 500ing with FileNotFoundError on the host. A URI that already
    points at a real path (e.g. a test's tmp_path) is returned unchanged.
    """
    path = Path(uri.strip().removeprefix("file://"))
    if path.exists():
        return path
    mount_path = Path(os.getenv("DATASET_MOUNT_PATH", _DEFAULT_MOUNT_PATH))
    try:
        relative = path.relative_to(mount_path)
    except ValueError:
        return path
    return _local_data_root() / relative


def dataset_uri_for_path(path: Path) -> str:
    """Inverse of resolve_dataset_path: turns a local path back into the
    mount-path URI the training pod understands, so an endpoint that writes
    a derived file (enrich-features) hands back a URI the workflow can read
    rather than a host-only path. A path outside the local data root keeps
    its own `file://` form."""
    try:
        relative = path.relative_to(_local_data_root())
    except ValueError:
        return f"file://{path}"
    mount_path = os.getenv("DATASET_MOUNT_PATH", _DEFAULT_MOUNT_PATH)
    return f"file://{mount_path}/{relative.as_posix()}"


@lru_cache
def _minio_adapter() -> "MinioObjectStorageAdapter":
    # Cached so the preview/columns/validate endpoints don't build a fresh
    # boto3 client (and connection pool) on every keystroke-driven call.
    return MinioObjectStorageAdapter()


def read_dataset_bytes(uri: str, source: str | None = None) -> bytes:
    """Reads a dataset's raw bytes, honouring which source the picker said it
    came from.

    `source="s3"` reads the object straight out of MinIO/S3 — the realistic
    path a dev should see the preview come from — and falls back to the local
    `data/` checkout if the bucket is unreachable (same fail-open spirit as
    CompositeObjectStorageAdapter.list_datasets). Any other source (or none,
    e.g. a hand-typed `file://` path) reads the local file directly.
    """
    if source == "s3":
        try:
            return _minio_adapter().read_dataset(uri)
        except Exception:
            logger.warning(
                "MinIO read failed for %s — falling back to local data/", uri, exc_info=True
            )
    return resolve_dataset_path(uri).read_bytes()


class LocalFileObjectStorageAdapter(IObjectStorageAdapter):
    """Datasets already checked out on the local filesystem — DVC's
    working-tree copy under `data/` (see .dvc/config's remote "storage").
    Works both for a dev machine running the orchestration API directly (no
    MinIO needed) and for the orchestration-api pod, which bind-mounts
    `./data` read-only via a hostPath volume — anywhere else `data/` doesn't
    exist on disk, so `list_datasets` just returns `[]`, the same fail-open
    contract `CompositeObjectStorageAdapter` relies on.
    """

    def __init__(self, root_path: str | None = None, mount_path: str = _DEFAULT_MOUNT_PATH):
        self.root_path = Path(root_path) if root_path else _local_data_root()
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

    def read_dataset(self, uri: str) -> bytes:
        return resolve_dataset_path(uri).read_bytes()


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
        # Bounded retries/timeouts — without these, botocore's default
        # backoff stalls every dataset-picker call ~8s when the MinIO
        # port-forward is down, before Composite's fail-open can serve
        # the local listing.
        self.client = boto3.client(
            "s3",
            endpoint_url=self.endpoint_url,
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "minioadmin"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin"),
            config=Config(connect_timeout=2, read_timeout=5, retries={"max_attempts": 2}),
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

    def _bucket_and_key(self, uri: str) -> tuple[str, str]:
        # list_datasets builds uri as file://<mount_path>/<bucket>/<key>, so
        # the first path segment under the mount is the bucket and the rest is
        # the object key.
        relative = Path(uri.strip().removeprefix("file://")).relative_to(self.mount_path)
        bucket, _, key = relative.as_posix().partition("/")
        return bucket, key

    def read_dataset(self, uri: str) -> bytes:
        bucket, key = self._bucket_and_key(uri)
        return bytes(self.client.get_object(Bucket=bucket, Key=key)["Body"].read())


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

    def read_dataset(self, uri: str) -> bytes:
        last_error: Exception | None = None
        for adapter in self.adapters:
            try:
                return adapter.read_dataset(uri)
            except Exception as error:
                last_error = error
        if last_error is not None:
            raise last_error
        raise FileNotFoundError(uri)
