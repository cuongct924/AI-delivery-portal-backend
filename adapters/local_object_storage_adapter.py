"""Adapter for datasets already checked out on the local filesystem — DVC's
working-tree copy under `data/` (see .dvc/config's remote "storage"). Works
both for a dev machine running the orchestration API directly (no MinIO
needed) and for the docker-compose container, which bind-mounts `./data`
read-only (see docker-compose.yml's `orchestration-api` service) — anywhere
else `data/` doesn't exist on disk, so `list_datasets` just returns `[]`,
the same fail-open contract `CompositeObjectStorageAdapter` relies on.
"""

import os
from pathlib import Path

from adapters.interfaces import DatasetInfo, IObjectStorageAdapter

_REPO_ROOT = Path(__file__).resolve().parent.parent


class LocalFileObjectStorageAdapter(IObjectStorageAdapter):
    def __init__(self, root_path: str | None = None, mount_path: str = "/mnt/data"):
        self.root_path = Path(root_path or os.getenv("LOCAL_DATASETS_PATH") or _REPO_ROOT / "data")
        # Where the k3d training pod sees these files (hostPath mount, synced
        # via `docker cp`) — not this process's filesystem, so uris are always
        # built from mount_path, never root_path (see the hostPath comment in
        # infra/argo-workflows/train-register-cluster-template.yaml).
        self.mount_path = mount_path

    def list_datasets(self, prefix: str = "") -> list[DatasetInfo]:
        if not self.root_path.is_dir():
            return []
        datasets: list[DatasetInfo] = []
        for path in sorted(self.root_path.rglob(f"{prefix}*")):
            # A ".dvc" pointer file next to it marks a *tracked dataset*, as
            # opposed to any other file living under data/.
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
