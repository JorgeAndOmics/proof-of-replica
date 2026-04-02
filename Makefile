.PHONY: help install dev test test-unit test-integration lint format typecheck check clean build

# ──────────────────────────────────────────────
# Project variables
# ──────────────────────────────────────────────
PACKAGE := proof_of_replica
SRC_DIR := src
TEST_DIR := tests

# ──────────────────────────────────────────────
# Default target
# ──────────────────────────────────────────────
help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ──────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────
install: ## Install production dependencies
	uv sync --no-dev

dev: ## Install all dependencies (including dev)
	uv sync
	uvx pre-commit install

# ──────────────────────────────────────────────
# Quality gates
# ──────────────────────────────────────────────
lint: ## Run linter
	uv run ruff check $(SRC_DIR) $(TEST_DIR)

format: ## Format code
	uv run ruff format $(SRC_DIR) $(TEST_DIR)
	uv run ruff check --fix $(SRC_DIR) $(TEST_DIR)

format-check: ## Check formatting without modifying
	uv run ruff format --check $(SRC_DIR) $(TEST_DIR)

typecheck: ## Run type checker
	uv run mypy $(SRC_DIR)

# ──────────────────────────────────────────────
# Testing
# ──────────────────────────────────────────────
test: ## Run all tests
	uv run pytest

test-unit: ## Run unit tests only
	uv run pytest $(TEST_DIR)/unit

test-integration: ## Run integration tests only
	uv run pytest $(TEST_DIR)/integration -m integration

test-cov: ## Run tests with coverage report
	uv run pytest --cov=$(SRC_DIR) --cov-report=term-missing --cov-report=html

test-fast: ## Run tests in parallel
	uv run pytest -x -n auto

# ──────────────────────────────────────────────
# Combined checks
# ──────────────────────────────────────────────
check: lint format-check typecheck test ## Run ALL quality gates (use before commit)

# ──────────────────────────────────────────────
# Build and clean
# ──────────────────────────────────────────────
build: check ## Build distribution (after checks pass)
	uv build

clean: ## Remove build artifacts and caches
	rm -rf dist/ build/ .eggs/ *.egg-info/
	rm -rf .pytest_cache/ .mypy_cache/ .ruff_cache/
	rm -rf htmlcov/ .coverage coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
