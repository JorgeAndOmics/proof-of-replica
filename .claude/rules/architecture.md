---
globs: ["src/**/*.py"]
---

# Architecture and Design Patterns

## SOLID Principles (Adapted for Python)

- **Single Responsibility**: Each module/class does one thing. If you need "and" to describe it, split it.
- **Open/Closed**: Extend behavior via composition, protocols, and strategy patterns. Avoid modifying existing working code.
- **Liskov Substitution**: Subtypes must be usable wherever their parent type is expected. If you override a method, preserve the contract.
- **Interface Segregation**: Keep protocols small and focused. No client should depend on methods it does not use.
- **Dependency Inversion**: High-level modules depend on abstractions (protocols), not concrete implementations. Inject dependencies; do not instantiate them internally.

## Dependency Injection

- Prefer constructor injection:
  ```python
  class OrderService:
      def __init__(self, repo: OrderRepository, notifier: Notifier) -> None:
          self._repo = repo
          self._notifier = notifier
  ```
- For simple cases, use default arguments:
  ```python
  def process(data: str, parser: Parser = JsonParser()) -> Result: ...
  ```
- Avoid service locators and global registries.

## Module Organization

- Each module should have a clear, single purpose.
- Keep `__init__.py` minimal: only re-export public API.
- Use `__all__` to define the public surface explicitly.
- Internal helpers go in `_private_module.py` (leading underscore).

## Data Modeling

- Use `dataclasses` for plain data containers:
  ```python
  @dataclass(frozen=True, slots=True)
  class Coordinate:
      lat: float
      lon: float
  ```
- Use `Pydantic` models for validation at system boundaries (API, config, external data).
- Prefer `frozen=True` (immutable) by default. Only make mutable when mutation is needed.
- Use `NamedTuple` for lightweight, immutable records with positional semantics.
- Use `enum.StrEnum` for string-valued enumerations.

## Anti-Patterns to Avoid

- **God classes/modules**: No file > 400 lines. Split by responsibility.
- **Circular imports**: If two modules import each other, extract shared types to a third.
- **Premature abstraction**: Do not create interfaces until you have 2+ implementations. Start concrete.
- **Global mutable state**: No module-level mutable variables. Use explicit parameter passing.
- **String-typed APIs**: Use enums, Literals, or NewType instead of raw strings for fixed vocabularies.
- **Deep inheritance**: Max 2 levels. Prefer composition and protocols.

## Performance Considerations

- Profile before optimizing. Use `cProfile` or `py-spy`.
- Use generators for large datasets (lazy evaluation).
- Cache expensive computations with `functools.lru_cache` or `functools.cache`.
- Avoid repeated attribute lookups in tight loops.
- Use `__slots__` on classes instantiated many times.
- Prefer `collections.deque` over `list` for queue/stack patterns.
