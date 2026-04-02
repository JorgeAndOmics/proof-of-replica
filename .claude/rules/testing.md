---
globs: ["tests/**/*.py", "src/**/test_*.py"]
---

# Testing Standards (Strict TDD)

## TDD Workflow (Red-Green-Refactor)

1. **RED**: Write a failing test that defines the desired behavior.
2. **GREEN**: Write the minimum code to make the test pass.
3. **REFACTOR**: Clean up both test and implementation while keeping tests green.
4. Repeat. Never write implementation code without a failing test first.

## Test Organization

```
tests/
  conftest.py              # Shared fixtures, marks, plugins
  unit/                    # Fast tests, no I/O, no network
    test_<module>.py       # Mirrors src/ structure
  integration/             # Tests with real dependencies
    test_<feature>.py
    conftest.py            # Integration-specific fixtures
```

## Test Naming

- Files: `test_<module_name>.py`
- Functions: `test_<unit>_<scenario>_<expected_behavior>`
- Examples:
  ```python
  def test_parse_config_with_missing_key_raises_value_error(): ...
  def test_calculate_score_with_empty_input_returns_zero(): ...
  def test_user_create_with_duplicate_email_fails(): ...
  ```

## Test Structure (Arrange-Act-Assert)

```python
def test_calculate_discount_for_premium_user_applies_twenty_percent():
    # Arrange
    user = User(tier="premium", balance=100.0)
    calculator = DiscountCalculator()

    # Act
    result = calculator.apply(user)

    # Assert
    assert result.final_price == 80.0
    assert result.discount_applied == 20.0
```

- One logical assertion per test (multiple `assert` is fine if testing one behavior).
- No logic in tests (no `if`, `for`, `try/except`).
- Tests must be independent and order-insensitive.

## Fixtures

- Use `conftest.py` for shared fixtures. Scope them appropriately:
  - `function` (default): fresh per test
  - `module`: shared across test file
  - `session`: shared across entire run (use sparingly)
- Prefer factory fixtures over complex setup:
  ```python
  @pytest.fixture
  def make_user():
      def _make(name: str = "test", tier: str = "basic") -> User:
          return User(name=name, tier=tier)
      return _make
  ```

## Parametrize for Variants

```python
@pytest.mark.parametrize("input_val, expected", [
    ("hello", 5),
    ("", 0),
    ("  spaces  ", 10),
])
def test_count_chars(input_val: str, expected: int):
    assert count_chars(input_val) == expected
```

## Mocking Guidelines

- Mock at the boundary, not the internals. Mock I/O, network, time, randomness.
- Never mock the unit under test.
- Prefer dependency injection over `unittest.mock.patch`.
- When patching is necessary, patch where the name is looked up, not where it is defined.
- Use `pytest-mock`'s `mocker` fixture over `unittest.mock` directly.

## Coverage

- Target: 90%+ line coverage on `src/`.
- Run: `uv run pytest --cov=src --cov-report=term-missing`
- Coverage is a floor, not a goal. 100% coverage does not mean correct code.
- Never write tests solely to increase coverage numbers.

## Marks for Categorization

```python
@pytest.mark.slow          # Tests > 1s
@pytest.mark.integration   # Needs external resources
@pytest.mark.network       # Needs network access
```

Configure in `pyproject.toml`:
```toml
[tool.pytest.ini_options]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: marks tests requiring external resources",
    "network: marks tests requiring network access",
]
```
