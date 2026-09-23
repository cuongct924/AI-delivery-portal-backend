"""Generic `mlflow.pyfunc.PythonModel` wrapper — any object exposing
`.predict(X)` becomes servable through the same KServe `"mlflow"`
servingRuntime the sklearn/DL paths already use, no per-preset serving
code. Reused unchanged by CV and RecSys.
"""

from typing import Any

from mlflow import pyfunc as mlflow_pyfunc


class GenericPyfuncWrapper(
    mlflow_pyfunc.PythonModel  # pyright: ignore[reportPrivateImportUsage]
):
    """Delegates `predict()` to any wrapped model exposing that method.

    `model_input` is deliberately untyped (not `pd.DataFrame`) — a wrapped
    model's own `.predict()` may expect any input shape the Dev designed
    for, and an `mlflow.pyfunc.PythonModel` subclass with a typed
    `model_input` gets its `predict` auto-wrapped by MLflow with type-hint-
    based input coercion (`PythonModel.__init_subclass__`), which would
    silently reshape input never intended for that validation.
    """

    def __init__(self, model: Any) -> None:
        self._model = model

    def predict(self, context: Any, model_input: Any, params: dict[str, Any] | None = None) -> Any:
        return self._model.predict(model_input)
