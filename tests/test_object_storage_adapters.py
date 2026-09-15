"""Tests adapters/local_object_storage_adapter.py and
adapters/composite_object_storage_adapter.py."""

from pathlib import Path

from adapters.composite_object_storage_adapter import CompositeObjectStorageAdapter
from adapters.interfaces import DatasetInfo, IObjectStorageAdapter
from adapters.local_object_storage_adapter import LocalFileObjectStorageAdapter


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
