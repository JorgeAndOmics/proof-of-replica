---
globs: ["src/**/*.py", "tests/**/*.py", "*.toml", "*.yml", "*.yaml"]
---

# Security and Safety

## Secrets Management

- NEVER hardcode secrets, API keys, tokens, passwords, or connection strings.
- Use environment variables or a `.env` file (loaded via `python-dotenv` or `pydantic-settings`).
- `.env` must be in `.gitignore`. Always.
- Provide a `.env.example` with placeholder values for documentation.

## Input Validation

- Validate ALL external input (user input, API responses, file content, env vars).
- Use Pydantic models for structured validation at boundaries.
- Sanitize file paths to prevent directory traversal.
- Set size limits on file uploads, request bodies, and collection sizes.

## Dependency Safety

- Pin dependencies via `uv.lock` (committed to git).
- Review new dependencies before adding: check maintenance status, license, security advisories.
- Prefer well-maintained packages with active security response.
- Run `uv audit` periodically to check for known vulnerabilities.

## Safe Defaults

- File permissions: read/write for owner only on sensitive files.
- Network: prefer HTTPS. Validate TLS certificates.
- Serialization: never use `pickle` for untrusted data. Use JSON or msgpack.
- Subprocess: never use `shell=True` with user-provided input.
- SQL: always use parameterized queries. Never string-format SQL.
