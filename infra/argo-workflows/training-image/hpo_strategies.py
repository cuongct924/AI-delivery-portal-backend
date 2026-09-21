"""Hyperparameter Search Strategy — Grid/Random/Bayesian search over DL
hyperparameters, on top of the existing single-value default.

Lives inside training-image, not `adapters/delivery/interfaces.py` — this
container only has the `*.py` files the Dockerfile copies in, no access to
the top-level `adapters/` package (a different deployable). Same Strategy
Pattern shape as `adapters/delivery/interfaces.py`'s
`IDeployTrafficStrategy`/`IReleaseStrategy` though, plus `trial_count()`
(Grid needs to tell the caller how many trials the grid actually contains)
and `report_result()` (Bayesian needs each trial's outcome fed back before
it can suggest the next one).
"""

import itertools
from dataclasses import dataclass
from typing import Any, Protocol

import optuna

# Only this module's own Optuna progress spam — study/trial info is
# already visible via the nested MLflow runs hpo_runner.py creates.
optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass(frozen=True)
class SearchSpace:
    """One hyperparameter's search range/set — mutually exclusive shapes.

    Attributes:
        param_name: Hyperparameter name, e.g. "learning_rate".
        low: Range lower bound (with `high`) — numeric ranges only.
        high: Range upper bound (with `low`).
        choices: Discrete/categorical value set — mutually exclusive with
            low/high.
        is_int: Sample an int in [low, high] instead of a float. Ignored
            when `choices` is set.
    """

    param_name: str
    low: float | int | None = None
    high: float | int | None = None
    choices: list[Any] | None = None
    is_int: bool = False


class IHyperparameterSearchStrategy(Protocol):
    def trial_count(self, requested_trials: int, spaces: list[SearchSpace]) -> int:
        """How many trials this strategy actually runs.

        Random/Bayesian honor `requested_trials` (the Dev's budget); Grid
        ignores it — trial count is fixed by the size of the grid itself.
        """
        ...

    def suggest_trial(self, trial_number: int, spaces: list[SearchSpace]) -> dict[str, Any]:
        """Samples one hyperparameter combination for this trial."""
        ...

    def report_result(self, trial_number: int, value: float) -> None:
        """Feeds a completed trial's objective value back to the strategy
        — Bayesian search uses this to inform later suggestions; Grid/Fixed
        are no-ops."""
        ...


class FixedStrategy:
    """The existing default — 1 trial, the single fixed value from each
    space's `choices[0]`. Not wired into `train.py`'s dispatch
    (searchStrategy="fixed" skips the HPO loop entirely) — kept so the
    Strategy enum stays complete for any future generic caller."""

    def trial_count(self, requested_trials: int, spaces: list[SearchSpace]) -> int:
        return 1

    def suggest_trial(self, trial_number: int, spaces: list[SearchSpace]) -> dict[str, Any]:
        return {space.param_name: space.choices[0] for space in spaces if space.choices}

    def report_result(self, trial_number: int, value: float) -> None:
        pass


class GridSearchStrategy:
    """Exhaustively tries every combination in the Cartesian product of
    each space's `choices` — no ranges allowed."""

    def __init__(self) -> None:
        self._param_names: list[str] = []
        self._combinations: list[tuple[Any, ...]] | None = None

    def trial_count(self, requested_trials: int, spaces: list[SearchSpace]) -> int:
        self._build_grid(spaces)
        return len(self._combinations) if self._combinations else 0

    def suggest_trial(self, trial_number: int, spaces: list[SearchSpace]) -> dict[str, Any]:
        if self._combinations is None:
            self._build_grid(spaces)
        assert self._combinations is not None
        combo = self._combinations[trial_number]
        return dict(zip(self._param_names, combo, strict=True))

    def report_result(self, trial_number: int, value: float) -> None:
        pass

    def _build_grid(self, spaces: list[SearchSpace]) -> None:
        for space in spaces:
            if space.choices is None:
                raise ValueError(
                    f"GridSearchStrategy requires discrete choices for {space.param_name!r} "
                    "— low/high ranges aren't supported for grid search"
                )
        self._param_names = [space.param_name for space in spaces]
        choice_lists = (space.choices for space in spaces if space.choices)
        self._combinations = list(itertools.product(*choice_lists))


class _OptunaStrategy:
    """Shared `optuna.Study.ask()`/`tell()` plumbing for Random/Bayesian —
    only the sampler differs between the two."""

    def __init__(self, sampler: optuna.samplers.BaseSampler, direction: str) -> None:
        self._study = optuna.create_study(sampler=sampler, direction=direction)
        self._trials: dict[int, optuna.trial.Trial] = {}

    def trial_count(self, requested_trials: int, spaces: list[SearchSpace]) -> int:
        return requested_trials

    def suggest_trial(self, trial_number: int, spaces: list[SearchSpace]) -> dict[str, Any]:
        trial = self._study.ask()
        self._trials[trial_number] = trial
        sampled: dict[str, Any] = {}
        for space in spaces:
            if space.choices is not None:
                sampled[space.param_name] = trial.suggest_categorical(
                    space.param_name, space.choices
                )
            elif space.is_int:
                sampled[space.param_name] = trial.suggest_int(
                    space.param_name, int(space.low or 0), int(space.high or 0)
                )
            else:
                sampled[space.param_name] = trial.suggest_float(
                    space.param_name, float(space.low or 0.0), float(space.high or 0.0)
                )
        return sampled

    def report_result(self, trial_number: int, value: float) -> None:
        trial = self._trials.pop(trial_number)
        self._study.tell(trial, value)


class RandomSearchStrategy(_OptunaStrategy):
    """Optuna's `RandomSampler` — uniform random draws within budget
    `numTrials`, no use of past trial outcomes."""

    def __init__(self, direction: str, seed: int = 42) -> None:
        super().__init__(optuna.samplers.RandomSampler(seed=seed), direction)


class BayesianSearchStrategy(_OptunaStrategy):
    """Optuna's TPE sampler — uses past trials' outcomes to pick smarter
    later ones. Optuna's optional trial-pruning isn't wired in: it needs
    per-epoch intermediate values from `train_dl.py`'s own loop, not just
    this strategy/SearchSpace layer — left for later."""

    def __init__(self, direction: str) -> None:
        super().__init__(optuna.samplers.TPESampler(), direction)


def build_search_strategy(name: str, direction: str) -> IHyperparameterSearchStrategy:
    """Factory keyed by the Scaffolder form's `searchStrategy` value.

    Raises:
        ValueError: `name` isn't one of fixed/grid/random/bayesian.
    """
    if name == "fixed":
        return FixedStrategy()
    if name == "grid":
        return GridSearchStrategy()
    if name == "random":
        return RandomSearchStrategy(direction)
    if name == "bayesian":
        return BayesianSearchStrategy(direction)
    raise ValueError(f"unknown search strategy {name!r} — must be fixed/grid/random/bayesian")
