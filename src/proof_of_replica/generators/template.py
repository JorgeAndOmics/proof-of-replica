"""Template string generator — composable {part} placeholders."""

import random
from typing import Any

import numpy as np
from rstr.xeger import Xeger

from proof_of_replica.core.column_schema import GeneratorConfig
from proof_of_replica.exceptions import GenerationError


def generate_template_strings(
    config: GeneratorConfig,
    n: int,
    rng: np.random.Generator,
) -> list[str]:
    """Generate strings from a composable template.

    Parts are resolved left-to-right so that later parts can reference
    earlier ones via the ``relative`` part type.

    Args:
        config: GeneratorConfig with template_str and parts.
        n: Number of strings to generate.
        rng: Seeded RNG.

    Returns:
        List of generated strings.
    """
    if config.template_str is None or config.parts is None:
        msg = "Template generator requires 'template_str' and 'parts'"
        raise GenerationError(msg)

    # Shared state for sequential counters (persists across rows)
    seq_counters: dict[str, int] = {}

    if not config.unique:
        return [
            _render_one(config.template_str, config.parts, rng, seq_counters)
            for _ in range(n)
        ]

    # Unique mode: retry on collision
    seen: set[str] = set()
    values: list[str] = []
    max_attempts = n * 10

    for _ in range(max_attempts):
        val = _render_one(config.template_str, config.parts, rng, seq_counters)
        if val not in seen:
            seen.add(val)
            values.append(val)
            if len(values) == n:
                return values

    msg = f"Could not generate {n} unique template values after {max_attempts} attempts"
    raise GenerationError(msg)


def _render_one(
    template_str: str,
    parts: dict[str, dict[str, Any]],
    rng: np.random.Generator,
    seq_counters: dict[str, int],
) -> str:
    """Render a single template string by resolving all parts."""
    resolved: dict[str, str] = {}

    for part_name, part_def in parts.items():
        resolved[part_name] = _resolve_part(
            part_name, part_def, resolved, rng, seq_counters
        )

    # Substitute {part_name} placeholders
    result = template_str
    for part_name, value in resolved.items():
        result = result.replace(f"{{{part_name}}}", value)

    return result


def _resolve_part(
    name: str,
    part_def: dict[str, Any],
    resolved: dict[str, str],
    rng: np.random.Generator,
    seq_counters: dict[str, int],
) -> str:
    """Resolve a single template part based on its type."""
    part_type = str(part_def.get("type", ""))

    # Simple part types (no extra context needed)
    simple_resolvers = {
        "choice": lambda: _resolve_choice(part_def, rng),
        "regex": lambda: _resolve_regex(part_def, rng),
        "integer_range": lambda: _resolve_integer_range(part_def, rng),
        "float_range": lambda: _resolve_float_range(part_def, rng),
        "alphabet": lambda: _resolve_alphabet(part_def, rng),
        "relative": lambda: _resolve_relative(part_def, resolved, rng),
        "sequential": lambda: _resolve_sequential(name, part_def, seq_counters),
        "template": lambda: _resolve_nested_template(part_def, rng, seq_counters),
    }

    resolver = simple_resolvers.get(part_type)
    if resolver is not None:
        return resolver()

    msg = f"Unknown template part type: {part_type}"
    raise GenerationError(msg)


def _resolve_choice(part_def: dict[str, Any], rng: np.random.Generator) -> str:
    """Choose from a list of values with optional weights."""
    values = list(part_def.get("values", []))
    if not values:
        msg = "Template choice part requires 'values'"
        raise GenerationError(msg)
    weights = part_def.get("weights")
    if weights is not None:
        w = np.array(weights, dtype=np.float64)
        w = w / w.sum()
        return str(rng.choice(values, p=w))
    return str(rng.choice(values))


def _resolve_regex(part_def: dict[str, Any], rng: np.random.Generator) -> str:
    """Generate from a regex pattern."""
    pattern = str(part_def.get("pattern", ""))
    py_rng = random.Random(int(rng.integers(2**63)))  # noqa: S311
    xeger = Xeger()
    xeger._random = py_rng
    return xeger.xeger(pattern)


def _resolve_integer_range(part_def: dict[str, Any], rng: np.random.Generator) -> str:
    """Generate a random integer in range."""
    lo = int(part_def.get("min", 0))
    hi = int(part_def.get("max", 100))
    return str(int(rng.integers(lo, hi + 1)))


def _resolve_float_range(part_def: dict[str, Any], rng: np.random.Generator) -> str:
    """Generate a random float in range."""
    lo = float(part_def.get("min", 0.0))
    hi = float(part_def.get("max", 1.0))
    precision = part_def.get("precision")
    val = rng.uniform(lo, hi)
    if precision is not None:
        return f"{val:.{int(precision)}f}"
    return str(val)


def _resolve_relative(
    part_def: dict[str, Any], resolved: dict[str, str], rng: np.random.Generator
) -> str:
    """Generate a value relative to a previously resolved part."""
    base_name = str(part_def.get("base", ""))
    if base_name not in resolved:
        msg = f"Relative part references unknown base '{base_name}'"
        raise GenerationError(msg)
    base_val = int(resolved[base_name])
    offset_min = int(part_def.get("offset_min", 0))
    offset_max = int(part_def.get("offset_max", 100))
    offset = int(rng.integers(offset_min, offset_max + 1))
    return str(base_val + offset)


def _resolve_sequential(
    name: str, part_def: dict[str, Any], seq_counters: dict[str, int]
) -> str:
    """Generate a sequential counter value."""
    start = int(part_def.get("start", 1))
    zero_pad = int(part_def.get("zero_pad", 0))

    if name not in seq_counters:
        seq_counters[name] = start
    else:
        seq_counters[name] += 1

    val = seq_counters[name]
    return f"{val:0{zero_pad}d}" if zero_pad > 0 else str(val)


def _resolve_alphabet(part_def: dict[str, Any], rng: np.random.Generator) -> str:
    """Generate random chars from a character set."""
    chars = str(part_def.get("chars", "ACGT"))
    length = int(part_def.get("length", 10))
    indices = rng.integers(0, len(chars), size=length)
    return "".join(chars[i] for i in indices)


def _resolve_nested_template(
    part_def: dict[str, Any],
    rng: np.random.Generator,
    seq_counters: dict[str, int],
) -> str:
    """Resolve a nested template part."""
    template_str = str(part_def.get("template", ""))
    parts = part_def.get("parts", {})
    if not isinstance(parts, dict):
        msg = "Nested template requires 'parts' dict"
        raise GenerationError(msg)
    return _render_one(template_str, parts, rng, seq_counters)
