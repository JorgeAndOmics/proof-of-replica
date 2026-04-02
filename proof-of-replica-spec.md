# Proof of Replica — Technical Specification

**Version:** 0.2.0-draft
**Date:** 2026-04-01
**Author:** [Your Name]
**Status:** Draft — ready for implementation in Claude Code

---

## 1. Purpose

Proof of Replica is a CLI tool that generates statistically faithful synthetic replicas of tabular datasets. It exists to solve a specific workflow problem: embargoed or access-restricted data cannot be shared with AI agents or collaborators, but analysis scripts developed against a high-quality replica can be rerun on the real data without modification.

The tool is domain-agnostic. It operates on the structural and statistical properties of any tabular input, not on biological or domain-specific semantics.

### 1.1 Core Workflow

```
Real data (restricted)
        │
        ▼
  ┌─────────────┐
  │  por profile │ ──► profile.json (shareable, version-controlled)
  └─────────────┘
        │
        ▼
  ┌──────────────┐
  │ por generate  │ ──► replica.parquet / replica.csv
  └──────────────┘
        │
        ▼
  Agent develops analysis scripts against replica
        │
        ▼
  Scripts rerun on real data by authorized user
```

The agent never sees the real data. The profile JSON is the portable, version-controllable bridge between real data and replica generation.

### 1.2 Feature Tiers

Features are organized into two tiers throughout this spec:

- **Core (v1):** Required for first useful release. The tool is usable with only these features.
- **Stretch (v1.x+):** Implemented after core is stable. Each stretch feature is independently valuable and can be built incrementally.

Labels appear inline as `[Core]` or `[Stretch]` throughout.

---

## 2. Design Principles

1. **Profile is the single source of truth.** All configuration — schema, statistical properties, generation parameters, overrides, seed, row count — lives in the profile JSON. CLI flags can set initial values, but the persisted profile is canonical.

2. **Faithful by default, flexible on demand.** Out of the box, the tool reproduces univariate distributions, value ranges, column types, and missingness. Users can opt into correlation preservation, group-level effects, and other higher-fidelity features via per-column overrides.

3. **Privacy-conscious.** No real row should survive unchanged in the replica. The tool should report a nearest-neighbor distance metric between the real and synthetic data as a best-effort privacy signal.

4. **Deterministic.** Same profile (including seed) always produces the identical output.

5. **Quiet by default.** No output except errors unless verbosity is requested.

6. **Extensible without forking.** User-defined post-processing hooks allow custom logic after generation without modifying tool internals.

---

## 3. Installation

Packaged with `pyproject.toml`. Installable via:

```bash
pip install .                  # from cloned repo
pip install proof-of-replica   # future PyPI publication
```

### 3.1 Requirements

- Python 3.12+
- Core dependencies: polars, numpy, scipy, scikit-learn, statsmodels
- CLI framework: click or typer (implementer's choice)
- Reporting: matplotlib or plotly (for HTML fidelity reports)

### 3.2 Why Polars

Polars is the primary dataframe engine. Rationale:

- Significantly faster than pandas for column-type-aware operations
- Lower memory footprint via Apache Arrow memory model
- Lazy evaluation enables efficient processing of large datasets
- Strong native type system aligns well with the profiling task
- Native Parquet and CSV I/O

Pandas should only appear as an I/O bridge if a specific format requires it.

---

## 4. CLI Surface

### 4.1 `por profile` — Extract a statistical profile from real data `[Core]`

```
por profile <INPUT_FILE> [OPTIONS]

Arguments:
  INPUT_FILE              Path to real data file (.csv, .tsv, .parquet)

Options:
  -o, --output PATH       Output profile path [default: <input_stem>.profile.json]
  --separator TEXT         Column separator for CSV/TSV [default: auto-detect]
  --rows INT              Number of rows to sample for profiling (0 = all) [default: 0]
  --seed INT              Random seed for row sampling [default: 42]
  --correlations          Compute and store pairwise correlation matrix for numeric columns
  --group-column TEXT     Column name defining groups (e.g., "condition") to capture
                          group-level statistics separately
  --dp                    Enable differential privacy mode [Stretch]
  --epsilon FLOAT         DP epsilon parameter [default: 1.0] [Stretch]
  --delta FLOAT           DP delta parameter [default: 1e-5] [Stretch]
  -v, --verbose           Increase verbosity (-v, -vv)
```

**Output:** A JSON profile file (see Section 5) that fully describes the dataset without containing any real values.

### 4.2 `por generate` — Generate a synthetic replica `[Core]`

```
por generate <PROFILE> [OPTIONS]

Arguments:
  PROFILE                 Path to profile JSON

Options:
  -o, --output PATH       Output replica path [default: replica.<format>]
  --format TEXT           Output format: csv, tsv, parquet [default: parquet]
  --rows INT              Number of rows to generate [default: value from profile]
  --seed INT              Override the seed stored in the profile
  --override JSON_STR     Inline JSON overrides merged onto the profile before generation
  --override-file PATH    Path to a JSON file of overrides merged onto the profile
  --validate TEXT         Validation strictness: off, warn, fail [default: warn]
  --hook PATH             Python file containing a post_generate(df) -> df function [Stretch]
  --variants INT          Generate N replica variants with incremented seeds [Stretch]
  --split FLOAT           Train/test split ratio (e.g., 0.8); outputs two files [Stretch]
  --augment PATH          Append synthetic rows to this existing file [Stretch]
  --add-columns PATH      JSON defining new synthetic columns to add to existing data [Stretch]
  --chunk-size INT        Generate in chunks of this size (memory-efficient mode) [Stretch]
  -v, --verbose           Increase verbosity (-v, -vv)
```

**Output:** A synthetic tabular file and optionally a validation summary (see `--validate`).

### 4.3 `por validate` — Check replica fidelity against a profile `[Core]`

```
por validate <REPLICA> <PROFILE> [OPTIONS]

Arguments:
  REPLICA                 Path to the synthetic replica file
  PROFILE                 Path to the profile JSON

Options:
  --strictness TEXT       Validation level: warn, fail [default: warn]
  --report PATH           Write validation report to file [default: stdout]
  --format TEXT           Report format: text, json, html [default: text]
  -v, --verbose           Increase verbosity (-v, -vv)
```

**Checks performed:**

- Column names and types match the profile
- Value ranges are within expected bounds
- Univariate distribution similarity (KS test for continuous, chi-squared for categorical)
- Missingness rates within tolerance
- Cardinality of categorical columns within tolerance
- If correlations stored in profile: correlation matrix similarity (Frobenius norm)
- Privacy check: nearest-neighbor distance between replica and original (if original is accessible)
- Quasi-identifier detection: warn if column combinations could re-identify individuals `[Stretch]`
- k-anonymity check on the generated replica `[Stretch]`

### 4.4 `por diff` — Compare replica against original data `[Core]`

```
por diff <ORIGINAL> <REPLICA> [OPTIONS]

Arguments:
  ORIGINAL                Path to the real data file
  REPLICA                 Path to the synthetic replica file

Options:
  --report PATH           Write diff report to file [default: stdout]
  --format TEXT           Report format: text, json, html [default: text]
  --privacy               Include nearest-neighbor privacy distance metric
  -v, --verbose           Increase verbosity (-v, -vv)
```

**Output:** A column-by-column comparison of distributions, summary statistics, correlations, and optionally privacy metrics.

### 4.5 `por report` — Generate a fidelity report `[Core]`

```
por report <REPLICA> <PROFILE> [OPTIONS]

Arguments:
  REPLICA                 Path to the synthetic replica file
  PROFILE                 Path to the profile JSON

Options:
  -o, --output PATH       Output report path [default: report.html]
  --format TEXT           Report format: html, pdf [default: html]
  --original PATH         Path to original data for side-by-side comparison
  -v, --verbose           Increase verbosity (-v, -vv)
```

**Output:** An auto-generated HTML or PDF document containing:

- Per-column distribution plots (original vs synthetic, if original provided)
- Statistical test suite summary table (KS / chi-squared p-values per column)
- Correlation matrix heatmap comparison (if correlations in profile)
- Missingness pattern visualization
- Privacy distance histogram (if original provided)
- Overall fidelity score summary

### 4.6 `por profile-diff` — Compare two profiles `[Stretch]`

```
por profile-diff <PROFILE_A> <PROFILE_B> [OPTIONS]

Options:
  --report PATH           Write comparison to file [default: stdout]
  --format TEXT           Report format: text, json [default: text]
```

Useful for comparing profiles across dataset versions, cohorts, or sites.

### 4.7 `por profile-merge` — Merge profiles from multiple sources `[Stretch]`

```
por profile-merge <PROFILE_A> <PROFILE_B> [PROFILE_C ...] [OPTIONS]

Options:
  -o, --output PATH       Output merged profile path
  --weights FLOAT...      Relative weights for each profile [default: equal]
```

Combines profiles from multiple sites or cohorts into a single profile. Weighted averaging of statistics, union of categorical values, pooled correlation matrices.

### 4.8 `por profile-redact` — Strip sensitive information from a profile `[Stretch]`

```
por profile-redact <PROFILE> [OPTIONS]

Options:
  -o, --output PATH       Output redacted profile path
  --drop-columns TEXT...  Remove these columns entirely
  --drop-stats TEXT...    Remove specific stat fields (e.g., "percentiles", "min", "max")
  --round-to INT          Round all numeric stats to N significant figures [default: 3]
```

Produces a sanitized profile safe for broader sharing.

### 4.9 `por convert` — Format conversion utility `[Stretch]`

```
por convert <INPUT> -o <OUTPUT>

Supported conversions: csv <-> tsv <-> parquet
```

A convenience utility. No mutation, just read and write.

### 4.10 `por template` — Output an annotated example profile `[Core]`

```
por template [OPTIONS]

Options:
  -o, --output PATH       Output path [default: stdout]
  --minimal               Output a minimal profile with only required fields
  --full                  Output a comprehensive profile with all optional sections
```

Outputs a well-commented example profile JSON that serves as a starting point for hand-authoring. Comments explain every field, its purpose, allowed values, and defaults. The `--minimal` variant includes only the fields needed for basic generation; `--full` includes every optional section (correlations, constraints, structure, hooks, injections, privacy).

### 4.11 `por scaffold` — Generate a starter profile from a CSV header `[Core]`

```
por scaffold <INPUT_FILE> [OPTIONS]

Arguments:
  INPUT_FILE              Path to a data file (.csv, .tsv, .parquet)

Options:
  -o, --output PATH       Output profile path [default: <input_stem>.scaffold.json]
  --rows INT              Peek at this many rows for type inference (0 = header only) [default: 100]
  --separator TEXT         Column separator for CSV/TSV [default: auto-detect]
```

Reads column names and optionally a small sample of rows to infer types, then outputs a skeleton profile with column names, detected dtypes, and placeholder stats that the user can fill in. Unlike `por profile`, this does **not** compute real statistics from the data, making it safe to use when you only have header access or want to define stats manually.

### 4.12 `por init` — Interactive profile wizard `[Stretch]`

```
por init [OPTIONS]

Options:
  -o, --output PATH       Output profile path [default: profile.json]
  --from PATH             Start from an existing file (CSV header or partial profile)
```

A step-by-step interactive wizard that walks the user through defining a profile:

1. Ask for dataset name and row count
2. For each column: name, type, role, key statistics
3. Ask about group columns and group effects
4. Ask about correlations, missingness, constraints
5. Write the profile with validation

Can optionally start from a CSV header (via `--from`) to pre-populate column names and types.

### 4.13 `por edit` — Edit a profile with validation `[Stretch]`

```
por edit <PROFILE>
```

Opens the profile in `$EDITOR` (or `$VISUAL`). On save, the tool validates the profile against the JSON Schema and reports errors with suggestions before accepting the changes. If validation fails, the user is offered the choice to re-edit or discard changes.

### 4.14 `por replicate` — One-command data-in, replica-out

The primary convenience entrypoint. Takes a real dataset and produces a synthetic replica in a single command, with three levels of user involvement controlled by `--guided`.

```
por replicate <INPUT_FILE> [OPTIONS]

Arguments:
  INPUT_FILE              Path to real data file (.csv, .tsv, .parquet)

Options:
  -o, --output PATH       Output replica path [default: <input_stem>.replica.<format>]
  --format TEXT           Output format: csv, tsv, parquet [default: parquet]
  --guided TEXT           Interaction level: auto, summary, full [default: auto]
  --save-profile PATH     Also save the generated profile to this path
  --rows INT              Override row count [default: match original]
  --seed INT              Random seed [default: 42]
  --correlations          Preserve correlation structure
  --group-column TEXT     Column defining groups for group-level effects
  --report PATH           Also generate a fidelity report [default: none]
  --noise-level FLOAT     Global noise level [default: 0.05]
  -v, --verbose           Increase verbosity (-v, -vv)
```

**Guided modes:**

**`--guided auto`** `[Core]`

Zero interaction. Profiles the data with sensible defaults, generates the replica, and optionally produces a fidelity report. The entire pipeline runs without prompts.

```bash
# Simplest possible usage
por replicate patients.csv

# With report and saved profile
por replicate patients.csv --report report.html --save-profile patients.profile.json
```

Internally equivalent to `por profile` + `por generate` + optionally `por report`, chained with default settings. The profile is ephemeral unless `--save-profile` is specified.

**`--guided summary`** `[Stretch]`

Auto-profiles the data, then pauses to show a summary before generating:

```
Profiled patients.csv (1500 rows, 12 columns)

  sample_id      string     identifier    unique, sequential
  age            float64    feature       normal(52.3, 14.7), 2% null
  condition      category   group         3 levels: healthy/stage_1/stage_2
  gene_expr_TP53 float64    feature       lognormal(s=0.8), group effects detected
  measurement    date       feature       2020-01-15 to 2025-11-30, 5% null
  sequence       string     structured    auto-detected: alphabet(ACGT, len~150)
  ...

  Global: noise_level=0.05, correlations=off

Adjust? [Enter to proceed / e to edit globals / s to save profile and exit]
```

The user can accept defaults, tweak global parameters, or save the profile for manual editing before generation.

**`--guided full`** `[Stretch]`

Interactive column-by-column wizard. After auto-profiling, walks through each column:

```
Column 3/12: gene_expr_TP53
  Detected:  float64, lognormal(s=0.8, loc=0.0, scale=12.5)
  Role:      feature
  Nulls:     1.3%
  Group effects detected for "condition":
    healthy:  baseline
    stage_1:  shift=+2.3, scale=1.1x
    stage_2:  shift=+5.7, scale=1.4x

  Accept? [Enter=yes / d=change distribution / r=change role / n=adjust noise / s=skip column / ?=help]
```

For each column, the user can accept the auto-detected settings, modify specific properties, change the distribution family, adjust the role, or skip. After all columns are reviewed, the tool generates the replica.

This mode also prompts for:

- Global settings (noise level, seed, row count)
- Whether to enable correlation preservation
- Whether to enable cross-column constraints (if patterns are detected)
- Output format and path

**Implementation notes:**

- All three modes use the same underlying profiling and generation engines
- `--save-profile` is recommended in all modes so the user has a reproducible artifact
- In `auto` mode, the tool should print a one-line summary of what it did (e.g., "Generated 1500-row replica with 12 columns -> replica.parquet") unless `--quiet` is set
- The `summary` and `full` modes require a TTY; if stdin is not a terminal, they fall back to `auto` with a warning

---

## 5. Profile JSON Schema

The profile is the single artifact that bridges real data and synthetic generation. It must be self-contained: anyone with the profile and the tool can generate a replica without access to the original data.

```jsonc
{
  // Metadata
  "tool": "proof-of-replica",
  "version": "0.2.0",
  "created_at": "2026-04-01T14:00:00Z",
  "source_hash": "sha256:abc123...",   // Hash of source file for integrity tracking
  "seed": 42,
  "row_count": 1500,                   // Default rows to generate

  // Global generation defaults
  "defaults": {
    "noise_level": 0.05,               // Global perturbation strength [0.0 - 1.0]
    "preserve_nulls": true
  },

  // Profiler thresholds (used by por profile)
  "profiler": {
    "categorical_max_cardinality": 50,  // Max unique values before treating as free-text
    "categorical_max_fraction": 0.05,   // Or max cardinality as fraction of row count
    "correlation_threshold": 0.05,      // Only store correlations with |r| above this
    "distribution_fit_pvalue": 0.05,    // Goodness-of-fit threshold; below = use empirical
    "sparsity_threshold": 0.70          // Zero-fraction above which sparse mode activates
  },

  // Validation thresholds (used by por validate, por generate --validate, por report)
  "validation": {
    "ks_pvalue": 0.05,                  // KS test p-value threshold for continuous columns
    "chisq_pvalue": 0.05,              // Chi-squared p-value threshold for categorical columns
    "null_tolerance": 0.02,            // Acceptable absolute deviation in null fraction
    "range_tolerance": 0.05,           // Acceptable deviation in value ranges (fraction)
    "boolean_tolerance": 0.05,         // Acceptable absolute deviation in true fraction
    "correlation_frobenius": 0.1,      // Max Frobenius norm of correlation matrix difference
    "k_anonymity_k": 5,               // Minimum group size for k-anonymity check [Stretch]
    "quasi_id_max_k": 5               // Warn if any quasi-ID combo has group < this [Stretch]
  },

  // Column definitions (ordered)
  "columns": [
    {
      "name": "sample_id",
      "dtype": "string",
      "role": "identifier",             // identifier | feature | group | outcome | index
      "generator": {
        "method": "sequential",         // sequential | uuid | prefix_increment
        "prefix": "SAMPLE_",
        "zero_pad": 4                   // SAMPLE_0001, SAMPLE_0002, ...
      }
    },
    {
      "name": "age",
      "dtype": "float64",
      "role": "feature",
      "stats": {
        "mean": 52.3,
        "std": 14.7,
        "min": 18.0,
        "max": 89.0,
        "median": 51.0,
        "skewness": 0.23,
        "kurtosis": -0.41,
        "null_fraction": 0.02,
        "percentiles": {
          "5": 28.1,
          "25": 42.0,
          "75": 62.5,
          "95": 76.8
        }
      },
      "distribution": {
        "family": "normal",             // normal | lognormal | uniform | beta | gamma |
                                        // empirical_kde | empirical_histogram
        "params": {
          "loc": 52.3,
          "scale": 14.7
        }
      },
      "constraints": {
        "min": 18.0,                    // Hard floor after generation
        "max": 100.0,                   // Hard ceiling after generation
        "integer_valued": false
      },
      // Per-column overrides (merged over global defaults)
      "overrides": {
        "noise_level": 0.1
      }
    },
    {
      "name": "condition",
      "dtype": "categorical",
      "role": "group",
      "stats": {
        "cardinality": 3,
        "null_fraction": 0.0,
        "value_counts": {
          "healthy": 0.40,
          "stage_1": 0.35,
          "stage_2": 0.25
        }
      },
      "generator": {
        "method": "weighted_choice"     // Draw from value_counts proportions
      }
    },
    {
      "name": "gene_expr_TP53",
      "dtype": "float64",
      "role": "feature",
      "stats": { "..." : "..." },
      "distribution": {
        "family": "lognormal",
        "params": { "s": 0.8, "loc": 0.0, "scale": 12.5 }
      },
      "group_effects": {
        "group_column": "condition",
        "effects": {
          "healthy":  { "shift": 0.0, "scale_factor": 1.0 },
          "stage_1":  { "shift": 2.3, "scale_factor": 1.1 },
          "stage_2":  { "shift": 5.7, "scale_factor": 1.4 }
        }
      }
    },
    {
      "name": "measurement_date",
      "dtype": "date",
      "role": "feature",
      "stats": {
        "min": "2020-01-15",
        "max": "2025-11-30",
        "null_fraction": 0.05
      },
      "distribution": {
        "family": "uniform",
        "params": {
          "start": "2020-01-15",
          "end": "2025-11-30"
        }
      }
    },
    {
      "name": "notes",
      "dtype": "string",
      "role": "feature",
      "stats": {
        "null_fraction": 0.60,
        "mean_length": 42,
        "max_length": 256
      },
      "generator": {
        "method": "placeholder",        // placeholder | markov | lorem
        "placeholder_value": "[REDACTED]"
      }
    },
    {
      "name": "gene_id",
      "dtype": "string",
      "role": "identifier",
      "generator": {
        "method": "regex",              // Structured string: generate from regex [Core]
        "pattern": "ENSG[0-9]{11}",
        "unique": true
      }
    },
    {
      "name": "locus",
      "dtype": "string",
      "role": "feature",
      "generator": {
        "method": "template",           // Structured string: composable template [Stretch]
        "template": "{chrom}:{start}-{end}",
        "parts": {
          "chrom": { "type": "choice", "values": ["chr1","chr2","chr3","chr17","chrX"] },
          "start": { "type": "integer_range", "min": 1000000, "max": 250000000 },
          "end":   { "type": "relative", "base": "start", "offset_min": 100, "offset_max": 50000 }
        },
        "unique": false
      }
    },
    {
      "name": "sequence",
      "dtype": "string",
      "role": "feature",
      "generator": {
        "method": "alphabet",           // Structured string: random from alphabet [Core]
        "chars": "ACGT",
        "length": { "distribution": "normal", "mean": 150, "std": 20, "min": 50 }
      }
    },
    {
      "name": "is_control",
      "dtype": "boolean",
      "role": "feature",
      "stats": {
        "true_fraction": 0.40,
        "null_fraction": 0.0
      }
    }
  ],

  // Optional: pairwise correlation matrix (numeric columns only)
  "correlations": {
    "columns": ["age", "gene_expr_TP53"],
    "matrix": [
      [1.0, -0.12],
      [-0.12, 1.0]
    ],
    "method": "pearson"                 // pearson | spearman | kendall
  },

  // Optional: missingness structure
  "missingness": {
    "pattern": "MCAR",                  // MCAR | MAR | MNAR | observed
    "co_missing": {
      "notes,measurement_date": 0.8     // P(both null | either null)
    }
  },

  // Optional: cross-column constraints [Stretch]
  "constraints": {
    "conditional_nulls": [
      {
        "column": "dosage",
        "condition": "treatment == 'placebo'",
        "action": "set_null"
      }
    ],
    "temporal_ordering": [
      {
        "before": "admission_date",
        "after": "discharge_date"
      }
    ],
    "arithmetic_invariants": [
      {
        "derived": "bmi",
        "expression": "weight / (height / 100) ** 2",
        "tolerance": 0.01
      }
    ],
    "composite_uniqueness": [
      {
        "columns": ["patient_id", "visit_number"],
        "unique": true
      }
    ]
  },

  // Optional: post-processing hooks [Stretch]
  "hooks": {
    "post_generate": "hooks/my_postprocess.py"
  },

  // Optional: differential privacy metadata [Stretch]
  "privacy": {
    "mechanism": "differential_privacy",
    "epsilon": 1.0,
    "delta": 1e-5,
    "bounds_provided": true,
    "composition": "sequential"
  },

  // Optional: signal injections [Stretch]
  "injections": [
    {
      "column": "gene_expr_TP53",
      "type": "group_difference",
      "group_column": "condition",
      "groups": ["stage_2"],
      "effect_size": 1.5,
      "effect_type": "cohens_d"
    }
  ],

  // Optional: data structure metadata [Stretch]
  "structure": {
    "type": "flat",                      // flat | longitudinal | hierarchical | sparse
    "longitudinal": {
      "subject_column": "patient_id",
      "time_column": "visit_date",
      "expected_timepoints_per_subject": { "mean": 4, "std": 1.2 }
    },
    "hierarchical": {
      "levels": ["region", "hospital", "patient_id"],
      "counts_per_level": {
        "region": 5,
        "hospital_per_region": { "mean": 3, "std": 1 },
        "patient_per_hospital": { "mean": 50, "std": 15 }
      }
    },
    "sparse": {
      "zero_fraction": 0.92,
      "zero_inflation_method": "zero_inflated_negative_binomial"
    }
  }
}
```

### 5.1 Profile Design Notes

- The `source_hash` allows integrity tracking without storing data. It is informational only and not used during generation.
- `distribution.family: "empirical_kde"` and `"empirical_histogram"` store a KDE bandwidth or histogram bin edges/counts respectively. These enable high-fidelity reproduction of non-standard distributions.
- `group_effects` captures systematic differences between groups. During generation, the base distribution is drawn first, then shift/scale are applied per group membership.
- Top-level `constraints` defines cross-column relationships enforced as a post-generation pass. Per-column `constraints` defines value-level clamping.
- Free-text columns use placeholder generators by default. The `markov` method is a stretch goal for more realistic text.
- The `structure` block is advisory for flat datasets but drives generation logic for longitudinal, hierarchical, and sparse modes.
- All thresholds in `profiler` and `validation` blocks have sensible defaults. Users only need to include fields they want to override. Missing fields fall back to defaults.
- Per-column `overrides` can also override validation thresholds for that column (e.g., a column with known high variance might need a looser `range_tolerance`).

### 5.2 Formal JSON Schema `[Core]`

The tool ships with a JSON Schema file (`profile.schema.json`) that formally specifies the profile format. This enables:

- **Editor autocomplete and inline validation** in VS Code (via the JSON Schema Store or a local `$schema` reference), JetBrains IDEs, and other editors
- **Programmatic validation** using standard JSON Schema validators
- **Documentation generation** from the schema itself

The profile JSON can reference the schema directly:

```json
{
  "$schema": "./profile.schema.json",
  "tool": "proof-of-replica",
  "version": "0.2.0"
}
```

The schema should be generated from the tool's internal type definitions (e.g., Pydantic models) to ensure it stays in sync with the code. It is published alongside the package and versioned with it.

### 5.3 Profile Validation `[Core]`

Every command that reads a profile (`generate`, `validate`, `diff`, `report`) automatically validates the profile against the JSON Schema before proceeding. Validation errors include:

- **Location:** JSON path to the offending field (e.g., `columns[2].distribution.params.loc`)
- **Problem:** What's wrong (e.g., "Expected float, got string")
- **Suggestion:** How to fix it (e.g., "Change '52.3' to 52.3 (remove quotes)")

Example output:

```
ERROR  columns[2].stats.mean: missing required field
  → Add "mean" to the stats block, or remove the distribution to use defaults.

ERROR  validation.ks_pvalue: value 1.5 is out of range [0.0, 1.0]
  → Set to a value between 0.0 and 1.0 (default: 0.05).

WARN   columns[0].generator.method: "sequential" generates predictable IDs
  → Consider "uuid" if identifier uniqueness across datasets matters.
```

Validation is strict on types and structure, but lenient on optional fields (missing optional sections are silently filled with defaults).

---

## 6. Type Inference and Profiling Engine `[Core]`

When `por profile` receives a real dataset, it must automatically infer column types and extract statistical properties.

### 6.1 Type Detection Priority

The profiler should attempt detection in this order:

1. **Boolean** — two unique non-null values interpretable as true/false
2. **Integer count** — all non-null values are whole numbers, no decimal points in source
3. **Float / continuous** — numeric with decimal values
4. **Date / timestamp** — parseable as ISO 8601 or common date formats
5. **Categorical** — string column with cardinality below a threshold (default: 50 or 5% of row count, whichever is smaller)
6. **Identifier** — string column with cardinality equal to row count (all unique)
7. **Free-text string** — string column with high cardinality, variable lengths

### 6.2 Distribution Fitting

For continuous columns, the profiler should fit candidate distributions and select the best fit:

- Candidates: normal, lognormal, gamma, beta, uniform, exponential
- Selection: lowest AIC or BIC across candidates
- Fallback: if no parametric family fits well (goodness-of-fit p-value below threshold), use `empirical_kde` or `empirical_histogram`

For integer count columns:

- Candidates: Poisson, negative binomial, zero-inflated Poisson
- Same selection logic applies

For categorical columns:

- Store observed proportions directly in `value_counts`

### 6.3 Correlation Capture (optional, via `--correlations`)

- Compute pairwise Pearson (or Spearman if specified) correlation matrix for all numeric columns
- Store only correlations with |r| above `profiler.correlation_threshold` (default: 0.05) to keep profile compact
- During generation, use a Gaussian copula or Cholesky decomposition to induce the target correlation structure

### 6.4 Sparsity Detection `[Stretch]`

For datasets with high zero-fractions (common in single-cell, proteomics, count matrices):

- Detect if the zero fraction exceeds `profiler.sparsity_threshold` (default: 0.70)
- If so, set `structure.type` to `"sparse"` and record the zero fraction
- Fit a zero-inflated distribution (e.g., zero-inflated negative binomial) rather than a standard parametric family
- Profile the non-zero values separately for higher fidelity

---

## 7. Generation Engine `[Core]`

### 7.1 Pipeline

```
Load profile
    │
    ▼
Initialize RNG with seed
    │
    ▼
Generate structural skeleton [Stretch: longitudinal/hierarchical]
    │
    ▼
Generate group column(s) first (if any)
    │
    ▼
For each non-group column:
    ├── Identifier  → sequential / UUID / prefix generator
    ├── Boolean     → Bernoulli draw from true_fraction
    ├── Categorical → weighted random choice from value_counts
    ├── Date        → uniform or specified distribution within range
    ├── Continuous  → draw from fitted distribution family + params
    ├── Integer     → draw from count distribution, round
    └── Free-text   → placeholder / lorem / markov
    │
    ▼
Apply group effects (shift + scale per group membership)
    │
    ▼
Apply signal injections [Stretch]
    │
    ▼
Apply correlation structure (copula / Cholesky, if specified)
    │
    ▼
Apply per-column constraints (clamp to min/max, enforce integer, etc.)
    │
    ▼
Apply cross-column constraints [Stretch]
    ├── Conditional nulls
    ├── Temporal ordering (swap if violated)
    ├── Arithmetic invariants (recompute derived columns)
    └── Composite uniqueness (deduplicate or regenerate)
    │
    ▼
Inject missingness (MCAR/MAR/MNAR pattern + co-missingness)
    │
    ▼
Apply noise perturbation (global noise_level + per-column overrides)
    │
    ▼
Run post-generation hooks (if specified) [Stretch]
    │
    ▼
Privacy check: verify no real row survived unchanged (if original accessible)
    │
    ▼
Quasi-identifier warning (if enabled) [Stretch]
    │
    ▼
k-anonymity check (if enabled) [Stretch]
    │
    ▼
Validation pass (if --validate != off)
    │
    ▼
Write output file(s)
```

### 7.2 Noise Model

The `noise_level` parameter (0.0 to 1.0) controls perturbation strength:

- **Continuous columns:** additive Gaussian noise scaled to `noise_level * column_std`
- **Categorical columns:** random label flipping with probability `noise_level`
- **Integer columns:** additive discrete noise drawn from a rounded normal, scaled similarly
- **Dates:** additive noise in days, scaled to `noise_level * date_range_days`
- **Booleans:** bit-flip with probability `noise_level`

Noise is applied after distribution-based generation as an additional privacy/mutation layer.

### 7.3 Privacy Verification

After generation, if the original data is accessible:

- Compute per-row nearest-neighbor distance (Euclidean on normalized numeric columns) between the replica and original
- Report the minimum, mean, and 5th percentile distances
- Warn if any synthetic row is identical to a real row (distance = 0)

This is best-effort, not a formal differential privacy guarantee.

### 7.4 Post-Generation Hooks `[Stretch]`

Users can supply a Python file with a `post_generate` function:

```python
# hooks/my_postprocess.py
import polars as pl

def post_generate(df: pl.DataFrame) -> pl.DataFrame:
    """Arbitrary post-processing. Receives and returns a Polars DataFrame."""
    # Example: enforce a business rule
    df = df.with_columns(
        pl.when(pl.col("age") < 18)
          .then(pl.lit(None))
          .otherwise(pl.col("consent_date"))
          .alias("consent_date")
    )
    return df
```

The hook runs after all built-in generation steps but before validation. This allows users to inject domain-specific logic without modifying the tool.

---

## 8. Schema-Driven Mode (No Real Data) `[Core]`

When no real dataset exists, users can author a profile JSON by hand (or generate one via an LLM in the future). The profile schema in Section 5 is designed to be human-writable:

- `stats` blocks can be partially filled; the generator will use reasonable defaults for missing fields
- `distribution` can be omitted; the generator will fall back to normal for continuous, uniform for dates, weighted choice for categorical
- `group_effects` is entirely optional
- `correlations` is entirely optional

Minimal viable column definition:

```json
{
  "name": "age",
  "dtype": "float64",
  "stats": { "mean": 50, "std": 15, "min": 18, "max": 90 }
}
```

---

## 9. Validation Engine `[Core]`

Validation can run automatically after generation (`por generate --validate`) or standalone (`por validate`).

### 9.1 Checks and Tolerances

All thresholds below are configurable via the `validation` block in the profile JSON. The "Config Key" column shows the field name; defaults apply when omitted.

| Check | Method | Config Key | Default | Tier |
|---|---|---|---|---|
| Column names and order | Exact match | — | — | Core |
| Column dtypes | Exact match | — | — | Core |
| Value range (continuous) | min/max deviation | `range_tolerance` | 0.05 (5%) | Core |
| Univariate distribution (continuous) | KS test p-value | `ks_pvalue` | 0.05 | Core |
| Univariate distribution (categorical) | Chi-squared test p-value | `chisq_pvalue` | 0.05 | Core |
| Null fraction | Absolute difference | `null_tolerance` | 0.02 | Core |
| Categorical cardinality | Exact count match | — | — | Core |
| Boolean true fraction | Absolute difference | `boolean_tolerance` | 0.05 | Core |
| Correlation matrix similarity | Frobenius norm | `correlation_frobenius` | 0.1 | Core |
| Cross-column constraints | Per-constraint check | — | 100% | Stretch |
| Quasi-identifier detection | Column-combo uniqueness | `quasi_id_max_k` | 5 | Stretch |
| k-anonymity | Min group size | `k_anonymity_k` | 5 | Stretch |

Per-column overrides: any column can include a `"validation_overrides"` block that takes precedence over the global `validation` settings for that column. This is useful for columns with known unusual distributions that would otherwise trigger false warnings.

### 9.2 Strictness Modes

- **off:** no validation
- **warn:** run all checks, print warnings for failures, still write output
- **fail:** run all checks, exit with non-zero code on any failure, do not write output

---

## 10. Fidelity Reporting `[Core]`

The `por report` command generates a visual fidelity report as HTML or PDF.

### 10.1 Report Contents

**Per-column section:**

- Distribution overlay plot (histogram + KDE for continuous, bar chart for categorical)
- If original data provided: side-by-side original vs synthetic
- Summary statistics table (mean, std, median, min, max, null fraction)
- KS test or chi-squared test result with p-value
- Missingness visualization

**Global section:**

- Correlation matrix heatmap (original vs synthetic, if correlations captured)
- Privacy distance histogram (if original provided)
- Overall fidelity summary table: pass/warn/fail per column per check
- Cross-column constraint satisfaction summary `[Stretch]`

### 10.2 Implementation Notes

- Use matplotlib or plotly for static plots embedded in HTML
- PDF generation via HTML-to-PDF conversion (e.g., weasyprint) or direct matplotlib PDF backend
- Report should be self-contained (inline CSS, embedded images) for easy sharing

---

## 11. Structured String Generation

Columns containing structured text (accessions, locus notation, ontology terms, sequences, compound identifiers, URIs, formatted codes) require specialized generation beyond free-text or simple identifiers. The tool supports four approaches, tiered by complexity.

### 11.1 Regex Generator `[Core]`

Generate strings by sampling from a regular expression pattern using a regex-to-string library (e.g., `rstr` or `exrex`).

```jsonc
{
  "name": "ensembl_id",
  "dtype": "string",
  "generator": {
    "method": "regex",
    "pattern": "ENSG[0-9]{11}",
    "unique": true                      // Default: false
  }
}
```

Covers: accession IDs, ontology terms, formatted codes, simple compound identifiers, and any pattern expressible as a regex.

**Uniqueness enforcement:** When `unique: true`, the generator tracks emitted values and retries on collision. For large datasets with narrow patterns, it should warn if the pattern's entropy is insufficient for the requested row count.

### 11.2 Alphabet Generator `[Core]`

Generate random strings from a fixed character set with a configurable length distribution. Designed for biological sequences but general-purpose.

```jsonc
{
  "name": "dna_sequence",
  "dtype": "string",
  "generator": {
    "method": "alphabet",
    "chars": "ACGT",
    "weights": [0.29, 0.21, 0.21, 0.29],  // Optional: per-character weights (e.g., GC content)
    "length": {
      "distribution": "normal",
      "mean": 150,
      "std": 20,
      "min": 50,                        // Hard floor after sampling
      "max": 500                        // Hard ceiling after sampling
    },
    "unique": false
  }
}
```

Covers: DNA sequences (ACGT), protein sequences (20 amino acid alphabet), arbitrary fixed-alphabet strings.

**Character weights** allow tuning composition (e.g., GC content for DNA). When omitted, characters are drawn uniformly.

### 11.3 Template Generator `[Stretch]`

Generate strings from a composable template with typed, independently generated parts. This is the mechanism for compound structures where flat regex cannot enforce semantic relationships between parts.

```jsonc
{
  "name": "genomic_locus",
  "dtype": "string",
  "generator": {
    "method": "template",
    "template": "{chrom}:{start}-{end}",
    "parts": {
      "chrom": {
        "type": "choice",
        "values": ["chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr7",
                   "chr8", "chr9", "chr10", "chr11", "chr12", "chr13", "chr14",
                   "chr15", "chr16", "chr17", "chr18", "chr19", "chr20",
                   "chr21", "chr22", "chrX", "chrY"],
        "weights": null                 // Optional: per-value weights
      },
      "start": {
        "type": "integer_range",
        "min": 100000,
        "max": 250000000
      },
      "end": {
        "type": "relative",            // Value derived from another part
        "base": "start",
        "offset_min": 100,
        "offset_max": 50000
      }
    },
    "unique": false
  }
}
```

**Part types:**

| Type | Description | Parameters |
|---|---|---|
| `choice` | Random selection from a list | `values`, optional `weights` |
| `regex` | Generate from a regex pattern | `pattern` |
| `integer_range` | Random integer in range | `min`, `max` |
| `float_range` | Random float in range | `min`, `max`, optional `precision` |
| `relative` | Derived from another part | `base`, `offset_min`, `offset_max` |
| `sequential` | Incrementing counter | `start`, `zero_pad` |
| `alphabet` | Random string from charset | `chars`, `length` |

The `relative` type is the key enabler for intra-value constraints (e.g., `end > start`). Parts are generated left-to-right, so any part can reference a previously generated part.

**Additional examples:**

```jsonc
// TCGA barcode: TCGA-A1-A0SK-01A-12R-A084-07
{
  "method": "template",
  "template": "TCGA-{tss}-{participant}-{sample}{vial}{portion}{analyte}-{plate}-{center}",
  "parts": {
    "tss":         { "type": "regex", "pattern": "[A-Z][A-Z0-9]" },
    "participant": { "type": "regex", "pattern": "[A-Z0-9]{4}" },
    "sample":      { "type": "choice", "values": ["01","02","06","10","11"] },
    "vial":        { "type": "regex", "pattern": "[A-Z]" },
    "portion":     { "type": "regex", "pattern": "[0-9]{2}" },
    "analyte":     { "type": "choice", "values": ["R","D","T","W","G","X"] },
    "plate":       { "type": "regex", "pattern": "[A-Z][A-Z0-9]{3}" },
    "center":      { "type": "regex", "pattern": "[0-9]{2}" }
  }
}

// S3 URI: s3://bucket-name/sample_001/reads.fastq.gz
{
  "method": "template",
  "template": "s3://{bucket}/{sample_id}/{filename}.fastq.gz",
  "parts": {
    "bucket":    { "type": "choice", "values": ["raw-reads-prod", "raw-reads-staging"] },
    "sample_id": { "type": "template", "template": "sample_{n}", "parts": {
      "n": { "type": "sequential", "start": 1, "zero_pad": 3 }
    }},
    "filename":  { "type": "choice", "values": ["reads_R1", "reads_R2"] }
  }
}
```

### 11.4 Grammar Generator `[Stretch]`

For highly complex or recursive structures, a simplified BNF-like grammar can define the generation rules.

```jsonc
{
  "name": "complex_annotation",
  "dtype": "string",
  "generator": {
    "method": "grammar",
    "rules": {
      "start":  ["{db}:{accession}.{version}"],
      "db":     ["UniProt", "RefSeq", "GenBank"],
      "accession": ["{letter}{digits}"],
      "letter": ["P", "Q", "O", "A", "N", "X"],
      "digits": ["{d}{d}{d}{d}{d}{d}"],
      "d":      ["0","1","2","3","4","5","6","7","8","9"],
      "version": ["1","2","3","4","5"]
    },
    "unique": true
  }
}
```

Each key maps to a list of alternatives (chosen uniformly or with weights). Placeholders in `{braces}` reference other rules. This is more readable than complex regexes for nested structures and supports recursive patterns if needed.

### 11.5 Example-Based Pattern Inference `[Stretch]`

During profiling (`por profile`), the tool can auto-detect structured string patterns from real data. The profiler should:

1. **Detect structure:** For each string column with moderate cardinality (not free-text, not categorical), sample values and check for common structural signals: fixed prefixes/suffixes, consistent delimiters, fixed-length segments, numeric vs alpha segments.

2. **Infer a regex or template:** Use alignment of sampled values to extract the most specific pattern that covers 95%+ of observed values. Store the inferred pattern in the profile.

3. **Allow override:** The inferred pattern is a starting point. The user can replace it with a hand-written regex, template, or grammar in the profile.

**Auto-detection heuristics:**

- All values share a common prefix of 2+ chars -> likely accession (e.g., "ENSG", "GO:", "NM_")
- Values contain consistent delimiter patterns (`:`, `-`, `.`, `/`) -> likely compound structure
- Fixed-length numeric segments -> likely zero-padded identifiers
- Character set is restricted (e.g., only ACGT) -> likely biological sequence
- High entropy with consistent length -> likely random identifier or hash

**Profile output example (auto-detected):**

```jsonc
{
  "name": "gene_id",
  "dtype": "string",
  "role": "identifier",
  "stats": {
    "null_fraction": 0.0,
    "mean_length": 15,
    "pattern_coverage": 0.98           // Fraction of values matching inferred pattern
  },
  "generator": {
    "method": "regex",
    "pattern": "ENSG[0-9]{11}",        // Auto-inferred
    "inferred": true,                   // Flag indicating this was auto-detected
    "unique": true
  }
}
```

The `inferred: true` flag signals to the user that this pattern was auto-detected and may benefit from manual review.

---

## 12. Cross-Column Constraints `[Stretch]`

Relationships between columns that must hold in the generated data. Defined in the profile under top-level `"constraints"`.

### 11.1 Conditional Nulls

A column's nullness depends on another column's value.

```jsonc
{
  "column": "dosage",
  "condition": "treatment == 'placebo'",
  "action": "set_null"                // set_null | set_zero | set_value
}
```

**Enforcement:** After initial generation, evaluate the condition and apply the action. Runs before missingness injection to avoid conflicts.

### 11.2 Temporal Ordering

Date/timestamp columns that must respect a chronological relationship.

```jsonc
{
  "before": "admission_date",
  "after": "discharge_date"
}
```

**Enforcement:** Generate both columns independently, then swap values in any row where the constraint is violated. If swapping causes a cascade, regenerate the offending row.

### 11.3 Arithmetic Invariants

Derived columns that are deterministic functions of other columns.

```jsonc
{
  "derived": "bmi",
  "expression": "weight / (height / 100) ** 2",
  "tolerance": 0.01
}
```

**Enforcement:** Generate the input columns normally, then compute the derived column from the expression. The derived column's `distribution` in the profile is informational only (used for validation, not generation).

### 11.4 Composite Uniqueness

Combinations of columns that must be unique per row.

```jsonc
{
  "columns": ["patient_id", "visit_number"],
  "unique": true
}
```

**Enforcement:** Generate both columns independently, detect duplicate combinations, regenerate one column for colliding rows until unique.

---

## 13. Additional Generation Modes `[Stretch]`

### 12.1 Holdout Splitting

```bash
por generate profile.json --split 0.8 -o train.parquet
# Outputs: train.parquet (80%) and test.parquet (20%)
```

Both splits are drawn from the same profile. The split is deterministic given the seed. Useful for developing train/test pipelines against synthetic data.

### 12.2 Augmentation Mode

```bash
por generate profile.json --augment existing_data.parquet --rows 500 -o augmented.parquet
```

Appends N synthetic rows to an existing (real or synthetic) dataset. The synthetic rows are generated from the profile; the existing rows are passed through unchanged. Output contains both.

### 12.3 Column-Level Generation

```bash
por generate profile.json --add-columns new_columns.json --base existing.parquet -o extended.parquet
```

Adds new synthetic columns to an existing table. The `new_columns.json` defines the columns to add (same schema as profile column definitions). Existing columns are preserved.

### 12.4 Multiple Variants

```bash
por generate profile.json -o replicas/ --variants 5 --seed 42
# Produces replica_001.parquet through replica_005.parquet
# Seeds: 42, 43, 44, 45, 46
```

---

## 14. Data Structure Awareness `[Stretch]`

Beyond flat tables, the tool should understand common data structures in computational biology and the sciences.

### 13.1 Longitudinal / Panel Data

Datasets where the same subject appears across multiple timepoints.

**Profile additions** (in `structure.longitudinal`):

- `subject_column`: the column identifying subjects
- `time_column`: the column representing timepoints
- `expected_timepoints_per_subject`: distribution of visit counts

**Generation logic:**

1. Generate unique subject IDs first
2. For each subject, draw a number of timepoints from the specified distribution
3. Generate time values that are monotonically increasing per subject
4. Generate feature values with optional within-subject autocorrelation

### 13.2 Hierarchical / Nested Data

Data with grouping levels (patients within hospitals within regions).

**Profile additions** (in `structure.hierarchical`):

- `levels`: ordered list of grouping columns from coarsest to finest
- `counts_per_level`: distribution of group sizes at each level

**Generation logic:**

1. Generate the coarsest level first (e.g., 5 regions)
2. For each group, generate sub-groups (e.g., 3 hospitals per region)
3. Continue nesting down to the finest level
4. Feature values can have level-specific random effects (intercept shifts per group)

### 13.3 Sparse Matrices

Datasets dominated by zeros (single-cell RNA-seq, sparse proteomics).

**Profile additions** (in `structure.sparse`):

- `zero_fraction`: observed fraction of zeros
- `zero_inflation_method`: model for the zero-generating process

**Generation logic:**

1. Draw a binary mask: each cell is zero with probability `zero_fraction`
2. For non-zero cells, draw from the fitted distribution of non-zero values
3. Optionally use a zero-inflated negative binomial or similar model that jointly models zeros and counts

### 13.4 Wide/Long Format Conversion

```bash
por profile wide_data.csv --pivot-to-long --id-columns "patient_id,visit" --value-name "expression"
```

The profiler can optionally pivot wide-format data to long-format before profiling (useful when columns represent repeated measurements). The profile stores the pivot metadata so `por generate` can output in either format.

---

## 15. Profile Lifecycle Operations `[Stretch]`

### 14.1 Profile Diffing

```bash
por profile-diff cohort_a.profile.json cohort_b.profile.json
```

Compares two profiles and reports:

- Columns present in one but not the other
- Distribution parameter differences per shared column
- Correlation structure differences
- Missingness pattern differences

Useful for understanding how datasets differ across sites, cohorts, or time periods without accessing the raw data.

### 14.2 Profile Merging

```bash
por profile-merge site_a.profile.json site_b.profile.json -o combined.profile.json --weights 0.6 0.4
```

Produces a single profile representing the weighted combination of multiple source profiles:

- Weighted average of means, standard deviations, and other summary statistics
- Union of categorical value sets with reweighted proportions
- Pooled correlation matrices
- Combined row count (sum or weighted)

Enables multi-site synthetic data generation without centralizing real data.

### 14.3 Profile Redaction

```bash
por profile-redact sensitive.profile.json -o safe.profile.json \
  --drop-columns "ssn,full_name" \
  --drop-stats "percentiles" \
  --round-to 2
```

Strips potentially identifying information from a profile before sharing:

- Remove specified columns entirely
- Remove granular statistics (exact percentiles, min/max)
- Round remaining statistics to reduce precision
- Output is still a valid profile that can drive generation (with reduced fidelity)

---

## 16. Privacy and Safety Features

### 15.1 Nearest-Neighbor Distance Check `[Core]`

See Section 7.3.

### 15.2 Quasi-Identifier Detection `[Stretch]`

During validation, analyze combinations of categorical and low-cardinality columns to detect potential quasi-identifiers (column combinations that could uniquely or nearly-uniquely identify individuals).

Detection logic:

- For all combinations of 2-4 categorical/low-cardinality columns, compute the minimum group size
- Warn if any combination has groups of size < k (default k = 5)
- Report the specific column combinations and their minimum group sizes

### 15.3 k-Anonymity Check `[Stretch]`

Verify that the generated replica satisfies k-anonymity for a specified set of quasi-identifier columns:

- For each unique combination of quasi-identifier values, count the number of rows
- Fail if any group has fewer than k rows
- Configurable k (default: 5)

### 15.4 Differential Privacy Mode `[Stretch]`

Optional formal privacy guarantees applied during the **profiling** step, so that the profile itself is provably insensitive to any individual record.

**CLI surface:**

```bash
por profile data.csv --dp --epsilon 1.0 --delta 1e-5
```

**How it works:**

DP noise is injected at profile extraction time, not at generation time. This is the critical design choice: if the profile is differentially private, everything downstream (generation, sharing the profile) inherits the guarantee automatically. The generation step remains deterministic and noise-free (aside from the existing `noise_level` perturbation, which serves a different purpose).

**What gets noised during profiling:**

- **Counts and frequencies:** Laplace mechanism on value counts, null counts, row counts
- **Summary statistics:** Gaussian mechanism on means, standard deviations, percentiles
- **Histograms / KDE:** DP histogram via Laplace noise on bin counts before KDE fitting
- **Correlation matrices:** Wishart mechanism or analyze-then-Gaussian on the covariance matrix
- **Min/max values:** Clamped to user-specified bounds before profiling (required in DP mode to bound sensitivity)

**Per-column requirements in DP mode:**

Columns must have explicit `constraints.min` and `constraints.max` values to bound sensitivity. If not provided, the tool should either prompt the user or estimate bounds from the data with a portion of the privacy budget.

**Privacy budget accounting:**

The total (epsilon, delta) budget is split across all profiled statistics using sequential composition. The user controls the total budget; the tool allocates it across operations. A future refinement could expose per-statistic budget allocation for advanced users.

**Fidelity tradeoff:**

DP noise degrades profile accuracy, especially for small datasets (under ~500 rows) or high-cardinality categoricals. The tool should report an estimated fidelity impact after profiling in DP mode, so users can make an informed epsilon choice.

**Suggested dependency:** `opendp` or `diffprivlib` (IBM). Both are mature, well-documented, and support the mechanisms listed above.

---

## 17. Signal Injection `[Stretch]`

Allow users to inject known synthetic signals for testing analysis pipelines:

```jsonc
{
  "injections": [
    {
      "column": "gene_expr_TP53",
      "type": "group_difference",
      "group_column": "condition",
      "groups": ["stage_2"],
      "effect_size": 1.5,
      "effect_type": "cohens_d"         // cohens_d | log2fc | absolute_shift
    },
    {
      "column": "survival_months",
      "type": "correlation",
      "with_column": "gene_expr_TP53",
      "target_r": -0.4
    }
  ]
}
```

This enables ground-truth validation: you know the signal is there, so your analysis pipeline should detect it. Injections are applied after standard generation but before noise, so the signal-to-noise ratio is controllable.

---

## 18. Linked Tables `[Stretch]`

Support for multiple tables sharing referential keys (e.g., a clinical table and an expression matrix linked by `sample_id`).

Design sketch:

- A "project profile" JSON wraps multiple table profiles with a `links` section specifying foreign-key relationships
- Generation proceeds in dependency order: parent table first, then child tables sample from the generated parent IDs
- A `por profile-multi` subcommand accepts multiple input files and a link specification

---

## 19. LLM-Assisted Schema Generation `[Stretch]`

```bash
por schema-from-description "A clinical trial dataset with 500 patients,
  3 treatment arms, baseline vitals, and 6-month follow-up outcomes"
# Outputs a starter profile JSON
```

This would call an LLM API to generate a plausible profile JSON from a natural language description.

---

## 20. Streaming / Chunked Generation `[Stretch]`

For datasets too large to fit in memory:

- Generate in chunks of configurable size (via `--chunk-size`)
- Write incrementally to Parquet (row groups) or CSV (append mode)
- Correlation structure applied within chunks (approximate)
- Structural constraints (longitudinal, hierarchical) may require a planning pass before chunked generation

---

## 21. Project Structure

```
proof-of-replica/
├── pyproject.toml
├── README.md
├── LICENSE
├── profile.schema.json           # Formal JSON Schema for profile validation + editor support
├── src/
│   └── proof_of_replica/
│       ├── __init__.py
│       ├── cli.py                # CLI entrypoint and subcommand definitions
│       ├── profiler.py           # Type inference, distribution fitting, profile extraction
│       ├── scaffold.py           # Header-only profile scaffolding and template generation
│       ├── generator.py          # Synthetic data generation engine
│       ├── structured.py         # Structured string generators (regex, alphabet, template, grammar)
│       ├── validator.py          # Fidelity checks and reporting
│       ├── differ.py             # Original vs replica comparison
│       ├── reporter.py           # HTML/PDF fidelity report generation
│       ├── noise.py              # Noise model and perturbation logic
│       ├── privacy.py            # Nearest-neighbor checks, quasi-ID detection, k-anonymity
│       ├── dp.py                 # Differential privacy mechanisms [Stretch]
│       ├── types.py              # Type detection and inference logic
│       ├── distributions.py      # Distribution fitting and sampling
│       ├── correlations.py       # Copula / Cholesky correlation induction
│       ├── constraints.py        # Cross-column constraint enforcement [Stretch]
│       ├── structures.py         # Longitudinal, hierarchical, sparse generation [Stretch]
│       ├── hooks.py              # Post-generation hook loading and execution [Stretch]
│       ├── profile_ops.py        # Profile diff, merge, redact operations [Stretch]
│       ├── injections.py         # Signal injection logic [Stretch]
│       ├── schema.py             # Profile JSON schema, validation, I/O
│       └── io.py                 # File format readers/writers (CSV, TSV, Parquet)
└── tests/
    ├── test_profiler.py
    ├── test_generator.py
    ├── test_validator.py
    ├── test_reporter.py
    ├── test_constraints.py
    ├── test_structures.py
    ├── test_privacy.py
    ├── test_injections.py
    ├── test_roundtrip.py         # Profile → generate → validate end-to-end
    └── fixtures/
        ├── small_clinical.csv
        ├── longitudinal_panel.csv
        ├── sparse_counts.csv
        └── expected_profiles/
```

---

## 22. Testing Strategy

- **Unit tests** for each module: type inference, distribution fitting, each generator method, noise application, validation checks
- **Roundtrip tests:** profile real fixture data, generate replica, validate against profile, assert all checks pass
- **Determinism tests:** same seed + profile must produce bit-identical output across runs
- **Edge cases:** all-null columns, single-row datasets, single-column datasets, columns with zero variance, extremely high cardinality categoricals
- **Privacy tests:** verify no row in the replica exactly matches any row in the original
- **Constraint tests:** verify all cross-column constraints hold in generated output
- **Structure tests:** verify longitudinal data has correct subject-timepoint structure, hierarchical data has correct nesting, sparse data has correct zero fraction
- **Report tests:** verify HTML report generates without errors and contains expected sections
- **Hook tests:** verify custom post-processing hooks are loaded and applied correctly
- **Injection tests:** verify injected signals are detectable with appropriate statistical tests
- **Structured string tests:** verify regex generates valid matches, alphabet respects charset and length distribution, template enforces part constraints (e.g., end > start), uniqueness holds when requested, and auto-inference recovers known patterns from sample data

---

## 23. Open Questions

These should be resolved during implementation:

1. **Copula choice:** Gaussian copula is the default for correlation induction, but vine copulas handle tail dependencies better. Worth the complexity?

2. **KDE bandwidth selection:** Silverman's rule vs. cross-validated bandwidth for `empirical_kde`. Performance vs. fidelity tradeoff at scale.

3. **Mixed-type correlation:** Pearson only works for numeric pairs. Cramér's V for categorical-categorical, point-biserial for categorical-continuous? Or just Spearman across the board?

4. **Free-text generation:** Placeholder is the safe default. Markov chains trained on character or word n-grams could produce more realistic text but risk leaking real patterns. Worth the risk?

5. **Profile versioning:** Should the profile schema have a formal version field with migration logic, or is semver on the tool version sufficient?

6. **DP budget allocation strategy:** Even split across all statistics is simple but wasteful (some stats matter more than others). Allow user-specified per-statistic weights? Or an adaptive strategy that allocates more budget to statistics with higher sensitivity?

7. **DP library choice:** `opendp` has a stronger formal verification story (proof-carrying code), while `diffprivlib` has a more familiar scikit-learn-style API. Which aligns better with the project's priorities?

8. **Constraint expression language:** The `"condition"` field in conditional nulls and the `"expression"` field in arithmetic invariants need a safe expression evaluator. Use a restricted subset of Python (via `ast.literal_eval` or `numexpr`), or define a custom DSL?

9. **Hierarchical random effects:** When generating hierarchical data, how should level-specific effects be parameterized? Fixed intercept shifts per group, or full random-effects covariance?

10. **Report library:** matplotlib is more portable but plotly produces more interactive HTML reports. Which is preferred? Or support both via a flag?

11. **Sparse storage format:** Should sparse replicas be output in a sparse-aware format (e.g., scipy sparse matrix serialized to npz, or AnnData .h5ad) in addition to standard Parquet/CSV?

12. **Regex generation library:** `rstr` is simple but limited; `exrex` enumerates all matches (memory issues for broad patterns); `regen` is more modern. Which balances correctness, performance, and pattern coverage?

13. **Pattern inference algorithm:** Aligning string samples to extract a regex is a research problem. Use a heuristic prefix/delimiter/segment approach, or a more formal algorithm like LNRE (Large Number of Rare Events) sequence analysis? The heuristic is simpler but may miss complex patterns.

14. **Template part ordering:** The current design generates parts left-to-right so later parts can reference earlier ones. Should circular or mutual dependencies be explicitly forbidden at schema validation time, or should the tool attempt topological sorting?
