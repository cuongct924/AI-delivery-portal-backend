---
name: python-typing-expert
description: Python OOP and modern typing standards for this repo (adapters/, agents/, services/orchestration-api/) — toolchain is ruff + pyright + pytest, all configured in the root pyproject.toml. Use when writing or reviewing Python code in this repo.
---

# Python Typing & OOP Expert

## Toolchain
*   **`ruff`** — lint + format. Config: `pyproject.toml` → `[tool.ruff]`.
*   **`pyright`** — static type checking. Config: `pyproject.toml` → `[tool.pyright]`.
*   **`pytest`** — tests. Config: `pyproject.toml` → `[tool.pytest.ini_options]`.
*   Install: `make install`. Run everything: `make check` (lint + typecheck + test).

## 1. Modern Type Hints (Python 3.12+)
*   **Built-in generics:** `list`, `dict`, not `typing.List`, `typing.Dict`.
*   **Union & Optional:** the `|` operator (e.g. `User | None`), never `typing.Union`/`typing.Optional`.
*   **Type aliases:** the Python 3.12 `type` statement (e.g. `type UserId = int`).
*   **Advanced typing:**
    *   `Self` for method chaining and factory methods.
    *   `Protocol` for structural typing without inheritance.
    *   `TypedDict` for type-safe dict structures.
    *   `ClassVar` for class attributes vs. `Final` for constants.
*   **Narrowing over suppressing:** when pyright reports a real union-type or `Optional` issue, narrow the type (an `if x is None: raise` guard, an `isinstance`/discriminant check) instead of reaching for `# type: ignore` or `cast` — reserve `cast` for cases where you truly know more than the checker (e.g. a third-party stub is wider than the real runtime contract).

## 2. Design Principles
*   **Composition over inheritance:** prefer "has-a" relationships over deep inheritance trees.
*   **Encapsulation:** `@property` instead of Java-style getters/setters; distinguish public from private (`_prefix`).
*   **Abstract base classes:** declare abstract methods with `@abstractmethod` (see `adapters/delivery/interfaces.py` or `adapters/ai_platform/interfaces.py` for the pattern used throughout this repo).
*   **SOLID:**
    *   **S**ingle responsibility.
    *   **O**pen for extension, closed for modification.
    *   **L**iskov substitution — subclasses must be seamlessly interchangeable.
    *   **I**nterface segregation — small, focused `Protocol`s/ABCs.
    *   **D**ependency inversion — depend on abstractions (`Protocol`/ABC), not concrete classes.

## 3. Static Type Checking
*   Run `pyright` (or `make typecheck`) before considering Python work done.
*   Third-party packages without type stubs (`kubernetes`, `mlflow`, `qdrant_client`, `mcp`) will surface real errors when their return types are wider than expected — read the actual error before dismissing it; it is frequently a genuine edge case (e.g. a nullable field, a broader SDK return-type union), not noise.
