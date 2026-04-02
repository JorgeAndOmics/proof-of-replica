---
globs: ["src/**/*.py", "tests/**/*.py"]
---

# Python Code Style

## Naming Conventions

- Classes: `PascalCase`
- Functions and methods: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Private members: `_single_leading_underscore`
- Type variables: `PascalCase` + `T` suffix (e.g., `ItemT`)
- Modules and packages: `short_snake_case`

## Imports

- Group order: stdlib, third-party, local (ruff handles this via `I` rules)
- Prefer explicit imports over star imports
- Use `from __future__ import annotations` only if targeting < 3.10
- For type-checking-only imports, use `TYPE_CHECKING` guard:
  ```python
  from __future__ import annotations
  from typing import TYPE_CHECKING
  if TYPE_CHECKING:
      from expensive_module import HeavyType
  ```

## Function Design

- Maximum 5 parameters per function. Use dataclasses/Pydantic models for more.
- Maximum function length: ~30 lines. If longer, extract helpers.
- One return type per function (no `Union[str, int]` returns unless modeling real domain variants).
- Use `@staticmethod` only when the method genuinely needs no instance/class state. Prefer module-level functions.
- Prefer early returns over deep nesting.

## Type Hints

- Annotate ALL function signatures (parameters + return types).
- Use modern syntax: `list[str]`, `dict[str, int]`, `str | None` (not `Optional`, `List`, `Dict`).
- Use `Protocol` over abstract base classes when defining structural interfaces.
- Define reusable type aliases in `types.py`:
  ```python
  type UserId = int
  type JsonDict = dict[str, Any]
  ```

## Docstrings (Google Style)

- Required on: all public modules, classes, functions, and methods.
- Not required on: private helpers, test functions, obvious one-liners.
- Format:
  ```python
  def process(data: list[str], threshold: float = 0.5) -> dict[str, float]:
      """Filter and score input data against threshold.

      Args:
          data: Raw string inputs to process.
          threshold: Minimum score to include. Defaults to 0.5.

      Returns:
          Mapping of passing items to their scores.

      Raises:
          ValueError: If data is empty.
      """
  ```

## String Formatting

- Use f-strings for interpolation (not `.format()` or `%`).
- Use triple-quoted strings for multi-line.
- Use `pathlib.Path` for all file path operations (never `os.path`).

## Collections and Iteration

- Prefer comprehensions over `map`/`filter` for readability.
- Use `itertools` and `more-itertools` for complex iteration.
- Use `set` for membership checks.
- Prefer `dict.get(key, default)` over `KeyError` handling for simple lookups.
- Use `enumerate()` instead of manual index tracking.
- Use `zip(..., strict=True)` when lengths must match.
