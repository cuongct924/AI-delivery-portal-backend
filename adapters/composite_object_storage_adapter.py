"""Merges dataset listings from every configured IObjectStorageAdapter
source (local filesystem + S3/MinIO) into the one list the Dataset URI
picker renders — Composite layered on the Adapter pattern so the picker
never needs to know how many storage backends exist.
"""

import logging

from adapters.interfaces import DatasetInfo, IObjectStorageAdapter

logger = logging.getLogger(__name__)


class CompositeObjectStorageAdapter(IObjectStorageAdapter):
    def __init__(self, adapters: list[IObjectStorageAdapter]):
        self.adapters = adapters

    def list_datasets(self, prefix: str = "") -> list[DatasetInfo]:
        datasets: list[DatasetInfo] = []
        for adapter in self.adapters:
            try:
                datasets.extend(adapter.list_datasets(prefix))
            except Exception:
                # One source being unreachable (e.g. MinIO down) shouldn't
                # hide datasets the other source(s) can still list.
                logger.warning("%s failed to list datasets", type(adapter).__name__, exc_info=True)
        return datasets
