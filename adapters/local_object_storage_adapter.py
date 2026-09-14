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
    def __init__(self, root_path: str | None = None):
        self.root_path = Path(root_path or os.getenv("LOCAL_DATASETS_PATH") or _REPO_ROOT / "data")

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
            datasets.append(
                DatasetInfo(
                    name=str(path.relative_to(self.root_path)),
                    uri=path.as_uri(),
                    size_bytes=path.stat().st_size,
                    source="local",
                )
            )
        return datasets
