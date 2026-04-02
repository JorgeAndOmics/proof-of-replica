---
globs: ["src/**/*.py"]
---

# Error Handling and Robustness

## Exception Hierarchy

- Define project-specific exceptions inheriting from a base:
  ```python
  class ProjectError(Exception):
      """Base exception for proof_of_replica."""

  class ConfigurationError(ProjectError):
      """Raised when configuration is invalid or missing."""

  class ValidationError(ProjectError):
      """Raised when input validation fails."""
  ```
- Never raise bare `Exception`. Always use specific types.
- Never use bare `except:` or `except Exception:` without re-raising or logging.

## Error Messages

- Be specific: `"Expected positive integer, got {value!r}"` not `"Invalid input"`.
- Include context: `"Failed to parse config file {path}: {error}"`.
- Suggest fixes when possible: `"Key 'api_url' missing from config. Add it to settings.toml."`.

## Defensive Patterns

- Validate at boundaries (public API entry points), trust internally.
- Use `assert` only for internal invariants (never for input validation).
- Prefer returning `Result` types or raising early over deeply nested error checks.
- For nullable values, handle `None` explicitly. Never let `None` propagate silently.

## Logging

- Use `logging` stdlib (or `structlog` if configured).
- Log levels:
  - `DEBUG`: Internal state useful for development
  - `INFO`: Expected operations (startup, shutdown, request processed)
  - `WARNING`: Unexpected but recoverable situations
  - `ERROR`: Failures that need attention
  - `CRITICAL`: System-level failures
- Never use `print()` for operational output. Use logging.
- Include structured context in log messages:
  ```python
  logger.error("Payment failed", extra={"user_id": uid, "amount": amt})
  ```

## Resource Management

- ALWAYS use context managers (`with`) for files, connections, locks.
- Implement `__enter__`/`__exit__` or use `@contextmanager` for custom resources.
- Use `atexit` or `try/finally` for cleanup that must run.

## Retry and Resilience

- Use `tenacity` for retries (exponential backoff, jitter).
- Set explicit timeouts on all I/O operations.
- Fail fast on unrecoverable errors; retry only transient failures.
