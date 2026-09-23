"""Text classification training — fine-tunes a pretrained HuggingFace
sequence-classification model with `transformers.Trainer`. Same role as
`train_dl.py`: a separate script `train.py` dispatches into, not a rewrite
of the shared split/gate/register flow.
"""

from typing import Any, Final, cast

import numpy as np
import pandas as pd
from datasets import Dataset
from metrics import compute_metrics
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# Maps optimizers.py's "adam"/"sgd" vocabulary to Trainer's `optim` string —
# "adam" maps to AdamW since plain Adam isn't a supported optimizer name.
_OPTIM_NAMES: Final[dict[str, str]] = {"adam": "adamw_torch", "sgd": "sgd"}


def train_and_evaluate(
    train_text: pd.Series,
    test_text: pd.Series,
    train_labels: pd.Series,
    test_labels: pd.Series,
    hyperparameters: dict[str, object],
) -> tuple[Any, dict[str, float]]:
    """Fine-tunes `hyperparameters["base_model_name"]` for single-label text
    classification and evaluates it on the held-out split.

    Args:
        train_text: Raw text column, train split — never run through
            `train.py`'s `_encode_categoricals()`, which would corrupt it
            into category codes.
        test_text: Raw text column, test split.
        train_labels: Label column (string classes), train split.
        test_labels: Label column, test split.
        hyperparameters: `base_model_name` (HuggingFace Hub model id),
            `learning_rate`, `epochs`, `batch_size`. Optional `optimizer`
            ("adam"/"sgd", default "adam").

    Returns:
        (transformers model, metrics) — `metrics` from the same
        `compute_metrics("classification", ...)` the sklearn/DL paths use.
    """
    base_model_name = cast(str, hyperparameters["base_model_name"])
    optimizer_name = str(hyperparameters.get("optimizer", "adam"))
    if optimizer_name not in _OPTIM_NAMES:
        raise ValueError(
            f"unknown optimizer {optimizer_name!r} — must be one of {sorted(_OPTIM_NAMES)}"
        )
    # Both splits' label sets combined so a class only present in the test
    # split still gets a stable code, and codes agree between splits.
    label_dtype = pd.CategoricalDtype(categories=sorted(set(train_labels) | set(test_labels)))
    train_label_ids = train_labels.astype(label_dtype).cat.codes
    test_label_ids = test_labels.astype(label_dtype).cat.codes
    num_labels = len(label_dtype.categories)

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name, num_labels=num_labels
    )

    def _tokenize(batch: dict[str, list[str]]) -> Any:
        return tokenizer(batch["text"], truncation=True, padding=True)

    train_dataset = Dataset.from_dict(
        {"text": train_text.tolist(), "label": train_label_ids.tolist()}
    ).map(_tokenize, batched=True)
    test_dataset = Dataset.from_dict(
        {"text": test_text.tolist(), "label": test_label_ids.tolist()}
    ).map(_tokenize, batched=True)

    training_args = TrainingArguments(
        output_dir="/tmp/nlp-trainer",
        num_train_epochs=int(cast(int, hyperparameters["epochs"])),
        per_device_train_batch_size=int(cast(int, hyperparameters["batch_size"])),
        per_device_eval_batch_size=int(cast(int, hyperparameters["batch_size"])),
        learning_rate=float(cast(float, hyperparameters["learning_rate"])),
        optim=_OPTIM_NAMES[optimizer_name],
        eval_strategy="epoch",
        logging_strategy="epoch",
        # train.py already owns the mlflow.start_run() for this job — the
        # Trainer's own MLflow integration would open a conflicting 2nd run.
        report_to=[],
        disable_tqdm=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=tokenizer,
    )
    trainer.train()

    # Cast: stub types predictions as a tuple for multi-output; this model has 1 head.
    predictions = trainer.predict(cast(Any, test_dataset))
    predicted_ids = cast(np.ndarray, predictions.predictions).argmax(axis=-1)
    metrics = compute_metrics("classification", test_label_ids, predicted_ids)
    return {"model": model, "tokenizer": tokenizer}, metrics
