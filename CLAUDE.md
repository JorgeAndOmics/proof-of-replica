# Proof of Replica

CLI tool that generates statistically faithful synthetic replicas of tabular datasets.

## Tech Stack

- **Language**: Python 3.12+
- **Package manager**: uv (never pip, conda, poetry, or requirements.txt)
- **Testing**: pytest (strict TDD)
- **Linting/formatting**: ruff
- **Type checking**: mypy (strict mode)
- **Pre-commit**: pre-commit (installed via `uvx pre-commit install`)
- **CI**: GitHub Actions

## Project Layout

```
src/proof_of_replica/    # All source code (src layout)
  __init__.py
  py.typed               # PEP 561 marker
  core/                  # Domain logic
  utils/                 # Shared helpers
  types.py               # Type aliases and protocols
tests/
  unit/                  # Fast, isolated tests
  integration/           # Tests with external dependencies
  conftest.py            # Shared fixtures
docs/                    # Architecture, ADRs, reference docs
.claude/rules/           # Detailed coding rules (auto-loaded)
```

## Commands (always prefix with `uv run`)

```bash
uv run pytest                          # Run all tests
uv run pytest tests/unit               # Unit tests only
uv run pytest --cov=src                # Tests with coverage
uv run ruff check .                    # Lint
uv run ruff check --fix .              # Lint + auto-fix
uv run ruff format .                   # Format
uv run mypy src                        # Type check
make check                             # Run all quality gates
make test                              # Run tests
```

## Development Workflow

1. **TDD cycle**: Write failing test -> Implement minimum code -> Refactor -> Repeat
2. **Commits**: Conventional Commits (`feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `ci:`, `chore:`)
3. **Before committing**: Run `make check` (runs ruff + mypy + pytest)
4. **Never commit**: Failing tests, type errors, or lint violations

## Critical Rules

- ALWAYS use `uv run` to execute tools. Never call `python`, `pytest`, `ruff` directly.
- ALWAYS use `uv add <pkg>` to add dependencies. Never edit pyproject.toml by hand for deps.
- NEVER create `requirements.txt`, `setup.py`, or `setup.cfg`.
- NEVER add `# type: ignore` without a specific error code and comment explaining why.
- NEVER use `Any` type without explicit justification in a comment.
- ALWAYS write tests BEFORE implementation (strict TDD).
- ALWAYS run `make check` before suggesting a commit.
- Every function must have type annotations.
- Every public function must have a Google-style docstring.
- Prefer explicit over implicit. Prefer simple over clever.

## Agent Behavior

- Ask before making destructive changes (deleting files, overwriting data).
- When uncertain about requirements, ask a clarifying question before implementing.
- After completing a task, run `make check` before reporting done.
- Prefer small, incremental changes over large rewrites.
- Lead with the solution, then explain reasoning.
- When showing code changes, explain what changed and why.
- If a task will take multiple steps, outline the plan first and confirm before proceeding.

## Architecture Notes

Profile-centric design: `real data -> por profile -> profile.json -> por generate -> replica`.
The profile JSON is the single source of truth. Polars is the primary DataFrame engine.
Generation pipeline: load profile -> generate group cols -> per-column generators -> group effects -> correlations -> constraints -> missingness -> noise -> privacy check -> validate -> write.
See @docs/architecture.md for detailed component map and data flow.

## What NOT to Do

- Do not create or activate virtual environments manually. uv manages `.venv/`.
- Do not install packages globally.
- Do not suppress errors silently (bare `except:` is forbidden).
- Do not use mutable default arguments.
- Do not import from `__init__.py` within the same package (use explicit module paths).
