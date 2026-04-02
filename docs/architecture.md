# Architecture Reference

> This document is loaded on demand via `@docs/architecture.md`. Do not duplicate in CLAUDE.md.

## Overview

Proof of Replica is a CLI tool that extracts statistical profiles from tabular datasets and generates synthetic replicas that faithfully reproduce the original's statistical properties. The profile JSON is the single portable artifact bridging real data and replica generation.

## Component Map

```
src/proof_of_replica/
  cli/                # Click-based CLI entry points
    main.py           # Click group: por profile, por generate, por validate, etc.
  core/               # Domain logic, no I/O
    profiler.py       # Type inference, distribution fitting, stats extraction
    schema.py         # Profile JSON schema, Pydantic models, validation
  engines/            # Orchestration engines
    generation.py     # Generation pipeline: profile -> synthetic DataFrame
    validation.py     # Validation checks (KS test, chi-squared, range, etc.)
    reporting.py      # Fidelity report generation (HTML/PDF)
  generators/         # Per-type column generators
    numeric.py        # Continuous and integer distribution sampling
    categorical.py    # Weighted choice from value_counts
    identifier.py     # Sequential, UUID, prefix_increment
    boolean.py        # Bernoulli draw from true_fraction
    date.py           # Date/timestamp generation within range
    string.py         # Regex, alphabet, placeholder generators
  utils/              # Shared helpers
    io.py             # File I/O (CSV, TSV, Parquet via Polars)
    noise.py          # Noise model (additive Gaussian, label flip, etc.)
    privacy.py        # Nearest-neighbor privacy distance
  types.py            # Shared type aliases and protocols
  exceptions.py       # Project exception hierarchy
```

## Architectural Decisions

### Profile-Centric Design

```
Real data  -->  por profile  -->  profile.json  -->  por generate  -->  replica
```

- The **profile JSON** is the single source of truth for generation. All configuration lives there.
- CLI flags set initial values during profiling; the persisted profile is canonical.
- Every command that reads a profile validates it against the JSON Schema first.

### Layered Architecture

```
CLI           (presentation: Click commands)
    |
Engines       (orchestration: profiling, generation, validation, reporting)
    |
Core + Generators  (domain logic: stats, distributions, column generators)
    |
Utils         (I/O, noise, privacy)
```

- **Core has no dependencies on CLI or engines.** Generators are pure functions.
- **Polars is the primary DataFrame engine.** Pandas only appears as an I/O bridge if required.
- **Determinism:** Same profile + seed always produces identical output.

### Generation Pipeline

1. Load and validate profile
2. Initialize RNG with seed
3. Generate group column(s) first
4. For each column: dispatch to type-specific generator
5. Apply group effects (shift + scale)
6. Apply correlation structure (Cholesky/copula, if specified)
7. Apply per-column constraints (clamp, integer enforcement)
8. Inject missingness (MCAR/MAR/MNAR)
9. Apply noise perturbation
10. Privacy check (no real row survives unchanged)
11. Validation pass (if enabled)
12. Write output

### Key Decisions Log

| Decision | Rationale | Date |
|---|---|---|
| src layout | Prevents accidental imports from project root | Init |
| Strict mypy | Catches type bugs before runtime | Init |
| Polars over pandas | Faster, lower memory, strong type system, native Arrow | Init |
| Click for CLI | Spec leaves choice open; Click is mature and composable | Init |
| Profile as JSON | Human-readable, version-controllable, schema-validatable | Init |
| Per-type generators | Each dtype has distinct generation logic; clean separation | Init |

## Data Flow

1. Input enters via CLI (`por profile`, `por generate`, etc.)
2. Profile JSON validated against schema at boundary
3. Engines orchestrate core logic (profiling, generation, validation)
4. Generators produce per-column synthetic data as Polars Series
5. Engines compose columns into a DataFrame, apply cross-column effects
6. Utils handle file I/O (Polars native read/write)
7. Output written as CSV/TSV/Parquet

## Extension Points

When adding new features, follow this checklist:

- [ ] Define domain types in `types.py` or `core/schema.py`
- [ ] Write generator logic in `generators/`
- [ ] Write failing tests first in `tests/unit/`
- [ ] Add engine orchestration if new pipeline step is needed
- [ ] Add CLI subcommand in `cli/main.py`
- [ ] Add integration tests in `tests/integration/`
- [ ] Update this document if architecture changes
