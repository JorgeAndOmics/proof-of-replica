"""Grammar string generator — BNF-like recursive rule expansion."""

import re

import numpy as np

from proof_of_replica.core.column_schema import GeneratorConfig
from proof_of_replica.exceptions import GenerationError

_PLACEHOLDER_RE = re.compile(r"\{(\w+)\}")
_MAX_DEPTH = 50


def generate_grammar_strings(
    config: GeneratorConfig,
    n: int,
    rng: np.random.Generator,
) -> list[str]:
    """Generate strings by expanding grammar rules.

    Starts from the ``start`` rule and recursively expands ``{placeholder}``
    references to other rules. Each rule maps to a list of alternatives,
    chosen uniformly at random.

    Args:
        config: GeneratorConfig with rules dict.
        n: Number of strings to generate.
        rng: Seeded RNG.

    Returns:
        List of generated strings.
    """
    if config.rules is None:
        msg = "Grammar generator requires 'rules'"
        raise GenerationError(msg)

    if "start" not in config.rules:
        msg = "Grammar rules must contain a 'start' rule"
        raise GenerationError(msg)

    if not config.unique:
        return [_expand_from_start(config.rules, rng) for _ in range(n)]

    seen: set[str] = set()
    values: list[str] = []
    max_attempts = n * 10

    for _ in range(max_attempts):
        val = _expand_from_start(config.rules, rng)
        if val not in seen:
            seen.add(val)
            values.append(val)
            if len(values) == n:
                return values

    msg = f"Could not generate {n} unique grammar values after {max_attempts} attempts"
    raise GenerationError(msg)


def _expand_from_start(rules: dict[str, list[str]], rng: np.random.Generator) -> str:
    """Expand from the start rule."""
    return _expand_rule(rules, "start", rng, depth=0)


def _expand_rule(
    rules: dict[str, list[str]],
    rule_name: str,
    rng: np.random.Generator,
    depth: int,
) -> str:
    """Recursively expand a grammar rule."""
    if depth > _MAX_DEPTH:
        msg = f"Grammar expansion exceeded max depth ({_MAX_DEPTH})"
        raise GenerationError(msg)

    alternatives = rules.get(rule_name)
    if alternatives is None:
        return f"{{{rule_name}}}"

    chosen = str(rng.choice(alternatives))

    def _replace_match(match: re.Match[str]) -> str:
        ref = match.group(1)
        return _expand_rule(rules, ref, rng, depth + 1)

    return _PLACEHOLDER_RE.sub(_replace_match, chosen)
