"""Tests adapters/ai_platform/object_storage.py (LocalFileObjectStorageAdapter,
MinioObjectStorageAdapter, CompositeObjectStorageAdapter)."""

from pathlib import Path
from unittest.mock import MagicMock

from adapters.ai_platform.interfaces import DatasetInfo, IObjectStorageAdapter
from adapters.ai_platform.object_storage import (
    _REPO_ROOT,
    CompositeObjectStorageAdapter,
    LocalFileObjectStorageAdapter,
    MinioObjectStorageAdapter,
)


def test_repo_root_resolves_above_the_adapters_package() -> None:
    # Regression: object_storage.py lives 2 directories under the repo root
    # (adapters/ai_platform/) — _REPO_ROOT must walk up 3 parents, not 2, or
    # LocalFileObjectStorageAdapter's default (unset LOCAL_DATASETS_PATH)
    # root_path silently resolves to adapters/data instead of <repo-root>/data.
    assert (_REPO_ROOT / "data").is_dir()
    assert (_REPO_ROOT / "adapters").is_dir()


def test_local_adapter_lists_dvc_tracked_files_with_source_local(tmp_path: Path) -> None:
    (tmp_path / "traditional-ml").mkdir()
    dataset = tmp_path / "traditional-ml" / "fraud-detection-sample.csv"
    dataset.write_text("a,b\n1,2\n")
    (tmp_path / "traditional-ml" / "fraud-detection-sample.csv.dvc").write_text("md5: abc")

    adapter = LocalFileObjectStorageAdapter(root_path=str(tmp_path))
    datasets = adapter.list_datasets()

    # uri is the k3d training pod's own mount path (/mnt/data), not
    # tmp_path — this process and that pod are different filesystems, so
    # a URI built from root_path would 404 there even though it resolves
    # fine right here (see the adapter's own docstring on `mount_path`).
    assert datasets == [
        DatasetInfo(
            name="traditional-ml/fraud-detection-sample.csv",
            uri="file:///mnt/data/traditional-ml/fraud-detection-sample.csv",
            size_bytes=8,
            source="local",
        )
    ]


def test_local_adapter_skips_files_without_a_dvc_pointer(tmp_path: Path) -> None:
    (tmp_path / "fraud-detection-sample.csv").write_text("data")
    (tmp_path / "fraud-detection-sample.csv.dvc").write_text("md5: abc")
    # Neither of these has a matching ".dvc" sibling, so neither is a
    # DVC-tracked dataset — just files that happen to live under data/.
    (tmp_path / "README.md").write_text("# data/")
    (tmp_path / ".gitignore").write_text("*.csv")

    adapter = LocalFileObjectStorageAdapter(root_path=str(tmp_path))

    assert [d["name"] for d in adapter.list_datasets()] == ["fraud-detection-sample.csv"]


def test_local_adapter_returns_empty_list_when_root_missing(tmp_path: Path) -> None:
    adapter = LocalFileObjectStorageAdapter(root_path=str(tmp_path / "does-not-exist"))

    assert adapter.list_datasets() == []


def _minio_adapter() -> MinioObjectStorageAdapter:
    adapter = MinioObjectStorageAdapter()
    adapter.client = MagicMock()
    return adapter


def test_minio_adapter_lists_dvc_tracked_files_across_every_bucket() -> None:
    # Reproduces the exact bug report: infra/ai-platform-zone/minio.yaml
    # `docker cp`s ./data's contents onto MinIO's hostPath as-is, so every
    # dataset-category folder is its own bucket holding the file under its
    # real name — never a single fixed bucket, never DVC's content-addressed
    # files/md5/<hash> keys (nothing runs `dvc push` against this MinIO).
    adapter = _minio_adapter()
    adapter.client.list_buckets.return_value = {
        "Buckets": [{"Name": "classification-telco-fraud-detection"}]
    }
    adapter.client.list_objects_v2.return_value = {
        "Contents": [
            {"Key": "telco-fraud-detection-sample.csv", "Size": 321},
            {"Key": "telco-fraud-detection-sample.csv.dvc", "Size": 42},
        ]
    }

    datasets = adapter.list_datasets()

    assert datasets == [
        DatasetInfo(
            name="classification-telco-fraud-detection/telco-fraud-detection-sample.csv",
            uri="file:///mnt/data/classification-telco-fraud-detection/telco-fraud-detection-sample.csv",
            size_bytes=321,
            source="s3",
        )
    ]
    adapter.client.list_objects_v2.assert_called_once_with(
        Bucket="classification-telco-fraud-detection", Prefix=""
    )


def test_minio_adapter_skips_files_without_a_dvc_pointer() -> None:
    # No matching ".dvc" sibling object — just a file that happens to live
    # in the bucket, not a DVC-tracked dataset.
    adapter = _minio_adapter()
    adapter.client.list_buckets.return_value = {
        "Buckets": [{"Name": "classification-telco-fraud-detection"}]
    }
    adapter.client.list_objects_v2.return_value = {
        "Contents": [{"Key": "random-upload.csv", "Size": 99}]
    }

    assert adapter.list_datasets() == []


def test_minio_adapter_returns_empty_when_no_buckets_exist() -> None:
    adapter = _minio_adapter()
    adapter.client.list_buckets.return_value = {"Buckets": []}

    assert adapter.list_datasets() == []


class _StubAdapter(IObjectStorageAdapter):
    def __init__(self, datasets: list[DatasetInfo] | None = None, error: Exception | None = None):
        self._datasets = datasets or []
        self._error = error

    def list_datasets(self, prefix: str = "") -> list[DatasetInfo]:
        if self._error:
            raise self._error
        return self._datasets


def test_composite_merges_datasets_from_every_source() -> None:
    local = DatasetInfo(name="a.csv", uri="file:///a.csv", size_bytes=1, source="local")
    s3 = DatasetInfo(name="b.csv", uri="file:///mnt/data/b.csv", size_bytes=2, source="s3")
    composite = CompositeObjectStorageAdapter([_StubAdapter([local]), _StubAdapter([s3])])

    assert composite.list_datasets() == [local, s3]


def test_composite_skips_a_source_that_raises_and_keeps_the_rest() -> None:
    s3 = DatasetInfo(name="b.csv", uri="file:///mnt/data/b.csv", size_bytes=2, source="s3")
    composite = CompositeObjectStorageAdapter(
        [_StubAdapter(error=ConnectionError("MinIO unreachable")), _StubAdapter([s3])]
    )

    assert composite.list_datasets() == [s3]
