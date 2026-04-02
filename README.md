# Proof of Replica

CLI tool that generates statistically faithful synthetic replicas of tabular datasets. Embargoed or access-restricted data cannot be shared with AI agents or collaborators, but analysis scripts developed against a high-quality replica can be rerun on the real data without modification.

## Quick Start

```bash
# Clone and enter the project
git clone https://github.com/JorgeAndOmics/proof-of-replica
cd proof-of-replica

# Install dependencies (requires uv: https://docs.astral.sh/uv/)
make dev

# Run tests
make test

# Run all quality checks
make check
```

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (package manager)

## Project Structure

```
src/proof_of_replica/    # Source code
tests/                   # Test suite
  unit/                  # Fast, isolated tests
  integration/           # Tests with external dependencies
docs/                    # Architecture docs, ADRs
```

## Development

### Setup

```bash
make dev                 # Install all deps + pre-commit hooks
```

### Daily Workflow

```bash
make test                # Run tests
make lint                # Check lint
make format              # Auto-format code
make typecheck           # Run mypy
make check               # All of the above (run before committing)
```

### Testing

This project follows strict TDD (test-driven development). Write tests first, then implement.

```bash
make test                # All tests
make test-unit           # Unit tests only
make test-cov            # Tests with coverage report
make test-fast           # Parallel execution
```

### Commit Convention

This project uses [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(scope): add new feature
fix(scope): fix a bug
refactor(scope): restructure without behavior change
test(scope): add or update tests
docs(scope): documentation changes
```

## License

MIT
