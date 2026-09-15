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
        # Where the training pod (k3d, not this process) sees these same
        # files — same reasoning as MinioObjectStorageAdapter's own
        # mount_path. `root_path` is wherever *this* process happens to
        # run (the repo checkout, or docker-compose's ./data bind-mount at
        # /app/data) — neither is the k3d training pod's filesystem, so a
        # URI built from root_path (e.g. path.as_uri()) would 404 there
        # even though it resolves fine right here. `/mnt/data` is synced
        # onto the k3d server node's filesystem separately (docker cp —
        # see infra/argo-workflows/train-register-template.yaml's hostPath
        # volume comment and top-level README.md); this only ever
        # constructs the URI train.py will read, never touches the file.
        self.mount_path = mount_path

    def list_datasets(self, prefix: str = "") -> list[DatasetInfo]:
        if not self.root_path.is_dir():
            return []
        datasets: list[DatasetInfo] = []
        for path in sorted(self.root_path.rglob(f"{prefix}*")):
            # A ".dvc" pointer file sitting next to it is what makes a file
            # a *dataset* DVC tracks, as opposed to README.md/.gitignore or
            # any other non-tracked file that happens to live under data/.
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
