"""Tests adapters/local_object_storage_adapter.py,
adapters/object_storage_adapter.py, and
adapters/composite_object_storage_adapter.py."""

from pathlib import Path
from unittest.mock import MagicMock

from adapters.composite_object_storage_adapter import CompositeObjectStorageAdapter
from adapters.interfaces import DatasetInfo, IObjectStorageAdapter
from adapters.local_object_storage_adapter import LocalFileObjectStorageAdapter
from adapters.object_storage_adapter import MinioObjectStorageAdapter


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


def _minio_adapter_with_local_dvc_files(tmp_path: Path) -> MinioObjectStorageAdapter:
    (tmp_path / "classification-telco-fraud-detection").mkdir()
    (
        tmp_path / "classification-telco-fraud-detection" / "telco-fraud-detection-sample.csv.dvc"
    ).write_text(
        "outs:\n"
        "- md5: 7fc207cd8dfefb2942162f38c2d1497b\n"
        "  size: 321\n"
        "  path: telco-fraud-detection-sample.csv\n"
    )
    adapter = MinioObjectStorageAdapter(local_datasets_path=str(tmp_path))
    adapter.client = MagicMock()
    return adapter


def test_minio_adapter_recovers_real_name_and_uri_from_dvc_pointer(tmp_path: Path) -> None:
    # Reproduces the exact bug report: `dvc push` uploads content-addressed
    # (`.../files/md5/<hash[:2]>/<hash[2:]>`), not under the dataset's real
    # filename — the S3 key alone carries no usable name, and a uri built
    # straight from it 404s (that hash never exists at the training pod's
    # mount path). Only the local .dvc pointer still has the mapping.
    adapter = _minio_adapter_with_local_dvc_files(tmp_path)
    adapter.client.list_objects_v2.return_value = {
        "Contents": [
            {
                "Key": "fraud-detection/files/md5/7f/c207cd8dfefb2942162f38c2d1497b",
                "Size": 321,
            }
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


def test_minio_adapter_skips_object_whose_hash_has_no_local_dvc_pointer(tmp_path: Path) -> None:
    # An object in the bucket with no matching .dvc file checked out
    # locally has no recoverable name/uri — surfacing it anyway would just
    # be a dataset picker entry that 404s the moment someone picks it.
    adapter = _minio_adapter_with_local_dvc_files(tmp_path)
    adapter.client.list_objects_v2.return_value = {
        "Contents": [
            {"Key": "fraud-detection/files/md5/ab/cdef0000000000000000000000000000", "Size": 99}
        ]
    }

    assert adapter.list_datasets() == []


def test_minio_adapter_skips_object_not_shaped_like_a_dvc_content_address(tmp_path: Path) -> None:
    adapter = _minio_adapter_with_local_dvc_files(tmp_path)
    adapter.client.list_objects_v2.return_value = {
        "Contents": [{"Key": "some/other/random-upload.csv", "Size": 99}]
    }

    assert adapter.list_datasets() == []


def test_minio_adapter_returns_empty_when_local_datasets_path_missing(tmp_path: Path) -> None:
    adapter = MinioObjectStorageAdapter(local_datasets_path=str(tmp_path / "does-not-exist"))
    adapter.client = MagicMock()
    adapter.client.list_objects_v2.return_value = {
        "Contents": [
            {"Key": "fraud-detection/files/md5/7f/c207cd8dfefb2942162f38c2d1497b", "Size": 321}
        ]
    }

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
