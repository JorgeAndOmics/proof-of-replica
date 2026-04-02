# ──────────────────────────────────────────────
# Stage 1: Build
# ──────────────────────────────────────────────
FROM python:${PYTHON_VERSION}-slim AS builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# Copy dependency files first (cache layer)
COPY pyproject.toml uv.lock ./

# Install production dependencies only
RUN uv sync --frozen --no-dev --no-editable

# Copy source code
COPY src/ src/

# ──────────────────────────────────────────────
# Stage 2: Runtime
# ──────────────────────────────────────────────
FROM python:${PYTHON_VERSION}-slim AS runtime

# Security: run as non-root
RUN groupadd --gid 1000 app && \
    useradd --uid 1000 --gid app --shell /bin/bash --create-home app

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy source code
COPY --from=builder /app/src /app/src

# Ensure venv binaries are on PATH
ENV PATH="/app/.venv/bin:$PATH"

# Switch to non-root user
USER app

# Health check (customize for your app)
# HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
#     CMD python -c "import proof_of_replica; print('ok')"

# Entry point
CMD ["python", "-m", "proof_of_replica"]
