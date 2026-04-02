# CLI Design Rules

## Entry Point

- The CLI is `por`, implemented with Click (`proof_of_replica.cli.main:app`)
- Each subcommand (`profile`, `generate`, `validate`, `diff`, `report`, `template`, `scaffold`, `replicate`) is a Click command registered on the main group

## Argument Conventions

- Positional arguments for required file paths (INPUT_FILE, PROFILE, REPLICA)
- `-o, --output` for output path with sensible defaults (`<stem>.profile.json`, `replica.<format>`)
- `-v, --verbose` for verbosity (supports `-v`, `-vv`)
- `--format` for output format selection (csv, tsv, parquet, text, json, html)
- `--seed` for reproducibility (default: 42)
- `--rows` for row count overrides

## Output Conventions

- Quiet by default: no output except errors unless `-v` is set
- Errors to stderr, data to stdout or file
- Exit code 0 on success, non-zero on failure
- `--validate fail` causes non-zero exit on validation failure

## Profile Handling

- Every command that reads a profile validates it against the JSON Schema first
- Validation errors include JSON path, problem description, and fix suggestion
- Missing optional fields silently fall back to defaults
