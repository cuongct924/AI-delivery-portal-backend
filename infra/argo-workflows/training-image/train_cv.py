"""Image classification training — torchvision transfer learning: a frozen
ImageNet-pretrained resnet18 backbone with a fresh linear head, fine-tuned
only on that head (CPU-friendly). Same role as train_dl.py/train_nlp.py: a
separate script train.py dispatches into.
"""

import base64
import io
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast

import pandas as pd
import torch
from metrics import compute_metrics
from optimizers import build_optimizer
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import ResNet18_Weights, resnet18

_IMAGE_SIZE = 224
_TEST_FRACTION = 0.2
_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((_IMAGE_SIZE, _IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class CVModel:
    """Wraps the fine-tuned backbone + class names for serving — the object
    `pyfunc_wrapper.GenericPyfuncWrapper` delegates `.predict()` to."""

    def __init__(self, backbone: nn.Module, class_names: list[str]) -> None:
        self._backbone = backbone
        self._class_names = class_names

    def predict(self, model_input: pd.DataFrame) -> list[str]:
        """`model_input` has exactly 1 column of base64-encoded image bytes
        (a serving-friendly, JSON-transportable encoding — the training-
        time input is files on disk instead)."""
        column = model_input.columns[0]
        images = [
            _TRANSFORM(Image.open(io.BytesIO(base64.b64decode(value))).convert("RGB"))
            for value in model_input[column]
        ]
        batch = torch.stack(cast(list[torch.Tensor], images))
        self._backbone.eval()
        with torch.no_grad():
            logits = self._backbone(batch)
        predicted_indices = logits.argmax(dim=1).tolist()
        return [self._class_names[i] for i in predicted_indices]


def train_and_evaluate(
    dataset_zip_path: Path, hyperparameters: dict[str, object]
) -> tuple[CVModel, dict[str, float]]:
    """Extracts the zip (`<class_name>/<file>` layout, `ImageFolder`-
    compatible), fine-tunes only the replaced final layer, and evaluates on
    a held-out 20%.

    Args:
        dataset_zip_path: Path to the (already-downloaded) dataset zip.
        hyperparameters: `learning_rate`, `epochs`, `batch_size` — same
            names/meaning as the DL path, reused as-is. Optional
            `optimizer` ("adam"/"sgd", default "adam") — same Dev-facing
            choice as train_dl.py, applied to just the replaced head's
            parameters since the backbone stays frozen.

    Returns:
        (CVModel, metrics) — `metrics` from the same
        `compute_metrics("classification", ...)` the sklearn/DL/NLP paths
        use.
    """
    learning_rate = float(cast(float, hyperparameters["learning_rate"]))
    epochs = int(cast(int, hyperparameters["epochs"]))
    batch_size = int(cast(int, hyperparameters["batch_size"]))

    with TemporaryDirectory() as extract_dir:
        with zipfile.ZipFile(dataset_zip_path) as zf:
            zf.extractall(extract_dir)
        full_dataset = ImageFolder(extract_dir, transform=_TRANSFORM)
        class_names = full_dataset.classes

        test_size = max(1, int(len(full_dataset) * _TEST_FRACTION))
        train_size = len(full_dataset) - test_size
        generator = torch.Generator().manual_seed(42)
        train_dataset, test_dataset = random_split(
            full_dataset, [train_size, test_size], generator=generator
        )

        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        for param in backbone.parameters():
            param.requires_grad = False
        backbone.fc = nn.Linear(backbone.fc.in_features, len(class_names))

        optimizer_name = str(hyperparameters.get("optimizer", "adam"))
        optimizer = build_optimizer(optimizer_name, backbone.fc.parameters(), learning_rate)
        criterion = nn.CrossEntropyLoss()
        train_loader = DataLoader(cast(Any, train_dataset), batch_size=batch_size, shuffle=True)

        backbone.train()
        for _epoch in range(epochs):
            for images, labels in train_loader:
                optimizer.zero_grad()
                loss = criterion(backbone(images), labels)
                loss.backward()
                optimizer.step()

        backbone.eval()
        test_loader = DataLoader(cast(Any, test_dataset), batch_size=batch_size)
        predicted_labels: list[int] = []
        true_labels: list[int] = []
        with torch.no_grad():
            for images, labels in test_loader:
                predicted_labels.extend(backbone(images).argmax(dim=1).tolist())
                true_labels.extend(labels.tolist())

    metrics = compute_metrics("classification", true_labels, predicted_labels)
    return CVModel(backbone, class_names), metrics
