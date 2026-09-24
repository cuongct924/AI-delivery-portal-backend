"""Tests adapters/ai_platform/object_storage.py (LocalFileObjectStorageAdapter,
MinioObjectStorageAdapter, CompositeObjectStorageAdapter)."""

from pathlib import Path
from unittest.mock import MagicMock

from adapters.ai_platform import object_storage
from adapters.ai_platform.interfaces import DatasetInfo, IObjectStorageAdapter
from adapters.ai_platform.object_storage import (
    _REPO_ROOT,
    CompositeObjectStorageAdapter,
    LocalFileObjectStorageAdapter,
    MinioObjectStorageAdapter,
    dataset_uri_for_path,
    read_dataset_bytes,
    resolve_dataset_path,
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


def test_resolve_dataset_path_maps_mount_uri_to_local_data_root(
    tmp_path: Path, monkeypatch
) -> None:
    # Reproduces the host-side 500: the picker hands back the training pod's
    # own /mnt/data URI, which doesn't exist on a dev machine — the resolver
    # must fall back to the repo's data/ (here redirected to tmp_path).
    monkeypatch.setenv("LOCAL_DATASETS_PATH", str(tmp_path))
    monkeypatch.setenv("DATASET_MOUNT_PATH", "/nonexistent-mount-xyz")
    (tmp_path / "sample.csv").write_text("a,b\n1,2\n")

    assert resolve_dataset_path("file:///nonexistent-mount-xyz/sample.csv") == (
        tmp_path / "sample.csv"
    )


def test_resolve_dataset_path_keeps_an_existing_path(tmp_path: Path) -> None:
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("a,b\n1,2\n")

    assert resolve_dataset_path(f"file://{csv_path}") == csv_path


def test_dataset_uri_for_path_maps_local_data_root_back_to_mount_uri(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("LOCAL_DATASETS_PATH", str(tmp_path))
    monkeypatch.setenv("DATASET_MOUNT_PATH", "/nonexistent-mount-xyz")

    assert (
        dataset_uri_for_path(tmp_path / "sample.csv") == "file:///nonexistent-mount-xyz/sample.csv"
    )


def test_dataset_uri_for_path_keeps_a_path_outside_the_data_root(tmp_path: Path) -> None:
    assert dataset_uri_for_path(tmp_path / "elsewhere.csv") == (
        f"file://{tmp_path / 'elsewhere.csv'}"
    )


def test_read_dataset_bytes_reads_local_file_when_source_is_local(tmp_path: Path) -> None:
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("a,b\n1,2\n")

    assert read_dataset_bytes(f"file://{csv_path}", "local") == b"a,b\n1,2\n"


def test_read_dataset_bytes_reads_from_minio_when_source_is_s3(tmp_path: Path, monkeypatch) -> None:
    # The picker said this dataset lives in MinIO/S3 — the read must go to the
    # bucket, not silently fall back to a same-named local file.
    (tmp_path / "data.csv").write_text("local-copy")
    monkeypatch.setenv("LOCAL_DATASETS_PATH", str(tmp_path))
    minio = _minio_adapter()
    minio.client.get_object.return_value = {"Body": MagicMock(read=lambda: b"from-minio")}
    monkeypatch.setattr(object_storage, "_minio_adapter", lambda: minio)

    assert read_dataset_bytes("file:///mnt/data/bucket/data.csv", "s3") == b"from-minio"
    minio.client.get_object.assert_called_once_with(Bucket="bucket", Key="data.csv")


def test_read_dataset_bytes_falls_back_to_local_when_minio_unreachable(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("LOCAL_DATASETS_PATH", str(tmp_path))
    monkeypatch.setenv("DATASET_MOUNT_PATH", "/nonexistent-mount-xyz")
    (tmp_path / "data.csv").write_text("local-copy")
    minio = _minio_adapter()
    minio.client.get_object.side_effect = ConnectionError("MinIO unreachable")
    monkeypatch.setattr(object_storage, "_minio_adapter", lambda: minio)

    assert read_dataset_bytes("file:///nonexistent-mount-xyz/data.csv", "s3") == b"local-copy"


def test_minio_adapter_read_dataset_parses_bucket_and_key_from_uri() -> None:
    adapter = _minio_adapter()
    adapter.client.get_object.return_value = {"Body": MagicMock(read=lambda: b"payload")}

    assert adapter.read_dataset("file:///mnt/data/my-bucket/nested/data.csv") == b"payload"
    adapter.client.get_object.assert_called_once_with(Bucket="my-bucket", Key="nested/data.csv")


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


def test_minio_adapter_bounds_retries_and_timeouts() -> None:
    # Regression: botocore's default backoff stalled every dataset-picker
    # call ~8s when the MinIO port-forward was down — fail-open must be fast.
    adapter = MinioObjectStorageAdapter(endpoint_url="http://localhost:9")
    config = adapter.client.meta.config

    assert config.retries["total_max_attempts"] == 3
    assert config.connect_timeout == 2
    assert config.read_timeout == 5


class _StubAdapter(IObjectStorageAdapter):
    def __init__(self, datasets: list[DatasetInfo] | None = None, error: Exception | None = None):
        self._datasets = datasets or []
        self._error = error

    def list_datasets(self, prefix: str = "") -> list[DatasetInfo]:
        if self._error:
            raise self._error
        return self._datasets

    def read_dataset(self, uri: str) -> bytes:
        if self._error:
            raise self._error
        return uri.encode()


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


def test_composite_read_dataset_falls_through_to_the_next_source() -> None:
    composite = CompositeObjectStorageAdapter(
        [_StubAdapter(error=FileNotFoundError("not local")), _StubAdapter()]
    )

    assert composite.read_dataset("file:///mnt/data/b.csv") == b"file:///mnt/data/b.csv"
