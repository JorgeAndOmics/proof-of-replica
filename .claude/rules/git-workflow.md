# Git Workflow and Commit Standards

## Conventional Commits

Format: `<type>(<scope>): <subject>`

### Types
- `feat`: New feature or capability
- `fix`: Bug fix
- `refactor`: Code change that neither fixes a bug nor adds a feature
- `test`: Adding or updating tests
- `docs`: Documentation only changes
- `ci`: CI/CD configuration changes
- `chore`: Build process, dependency updates, tooling
- `perf`: Performance improvement
- `style`: Formatting, whitespace (no code logic change)

### Rules
- Subject line: imperative mood, lowercase, no period, max 72 chars.
- Body (optional): Explain WHAT changed and WHY, not HOW.
- Footer (optional): `BREAKING CHANGE:` or `Closes #123`.
- Examples:
  ```
  feat(parser): add support for YAML config files
  fix(auth): prevent token refresh race condition
  refactor(core): extract validation logic into dedicated module
  test(api): add integration tests for /users endpoint
  ```

## Commit Granularity

- One logical change per commit. If you can split it, do.
- Tests and implementation in the SAME commit (they prove each other).
- Refactors in SEPARATE commits from feature/fix work.
- Never commit generated files, `.env`, secrets, or `.venv/`.

## Branch Strategy

- `main`: Always deployable. Protected.
- `feat/<description>`: Feature branches from main.
- `fix/<description>`: Bug fix branches from main.
- Squash-merge feature branches to main for clean history.

## Pre-Commit Checks

Before every commit, ensure:
1. `uv run ruff check .` passes (no lint errors)
2. `uv run ruff format --check .` passes (formatting clean)
3. `uv run mypy src` passes (no type errors)
4. `uv run pytest` passes (all tests green)

The `make check` command runs all four. Pre-commit hooks enforce this automatically.

## Never Commit

- Failing tests
- Type check errors
- Lint violations
- Commented-out code (delete it; git has history)
- TODO comments without a linked issue number
- Debug print statements
