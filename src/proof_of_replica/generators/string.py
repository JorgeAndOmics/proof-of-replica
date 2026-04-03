"""String column generators (regex, alphabet, placeholder, lorem)."""

import random

import numpy as np
import polars as pl
from rstr.xeger import Xeger

from proof_of_replica.core.column_schema import ColumnDefinition, GeneratorConfig
from proof_of_replica.core.enums import GeneratorMethod
from proof_of_replica.exceptions import GenerationError

_LOREM_WORDS = [
    "lorem",
    "ipsum",
    "dolor",
    "sit",
    "amet",
    "consectetur",
    "adipiscing",
    "elit",
    "sed",
    "do",
    "eiusmod",
    "tempor",
    "incididunt",
    "ut",
    "labore",
    "et",
    "dolore",
    "magna",
    "aliqua",
    "enim",
    "ad",
    "minim",
    "veniam",
    "quis",
    "nostrud",
    "exercitation",
    "ullamco",
    "laboris",
    "nisi",
    "aliquip",
    "ex",
    "ea",
    "commodo",
    "consequat",
    "duis",
    "aute",
    "irure",
    "in",
    "reprehenderit",
    "voluptate",
    "velit",
    "esse",
    "cillum",
    "fugiat",
    "nulla",
    "pariatur",
    "excepteur",
    "sint",
    "occaecat",
    "cupidatat",
    "non",
    "proident",
    "sunt",
    "culpa",
    "qui",
    "officia",
    "deserunt",
    "mollit",
    "anim",
    "id",
    "est",
    "laborum",
]


def generate_string(
    col: ColumnDefinition,
    n: int,
    rng: np.random.Generator,
) -> pl.Series:
    """Generate a string column using the configured generator method.

    Args:
        col: Column definition with GeneratorConfig.
        n: Number of rows.
        rng: Seeded RNG.

    Returns:
        Polars Series of Utf8 dtype.

    Raises:
        GenerationError: If generator config is missing or method unsupported.
    """
    if col.generator is None:
        msg = f"Column '{col.name}': string generator requires a generator config"
        raise GenerationError(msg)

    method = col.generator.method

    if method == GeneratorMethod.REGEX:
        values = _generate_regex(col.generator, n, rng)
    elif method == GeneratorMethod.ALPHABET:
        values = _generate_alphabet(col.generator, n, rng)
    elif method == GeneratorMethod.PLACEHOLDER:
        value = col.generator.placeholder_value or "[REDACTED]"
        values = [value] * n
    elif method == GeneratorMethod.LOREM:
        mean_length = None
        if col.stats is not None and hasattr(col.stats, "mean_length"):
            mean_length = getattr(col.stats, "mean_length", None)
        values = _generate_lorem(n, mean_length, rng)
    else:  # pragma: no cover — GeneratorMethod enum prevents unknown methods
        msg = f"Column '{col.name}': unsupported string method '{method}'"
        raise GenerationError(msg)

    return pl.Series(name=col.name, values=values, dtype=pl.Utf8)


def _generate_regex(
    config: GeneratorConfig, n: int, rng: np.random.Generator
) -> list[str]:
    """Generate strings matching a regex pattern via rstr."""
    if config.pattern is None:  # pragma: no cover — validated by GeneratorConfig
        msg = "Regex generator requires a 'pattern'"
        raise GenerationError(msg)

    # Create a seeded Xeger instance for deterministic regex generation
    py_rng = random.Random(int(rng.integers(2**63)))  # noqa: S311
    xeger = Xeger()
    xeger._random = py_rng

    if not config.unique:
        return [xeger.xeger(config.pattern) for _ in range(n)]

    # Unique mode: retry on collision
    seen: set[str] = set()
    values: list[str] = []
    max_attempts = n * 10

    for _ in range(max_attempts):
        val = xeger.xeger(config.pattern)
        if val not in seen:
            seen.add(val)
            values.append(val)
            if len(values) == n:
                return values

    msg = (
        f"Could not generate {n} unique values from pattern "
        f"'{config.pattern}' after {max_attempts} attempts"
    )
    raise GenerationError(msg)


def _generate_alphabet(
    config: GeneratorConfig, n: int, rng: np.random.Generator
) -> list[str]:
    """Generate random strings from a fixed character set."""
    if config.chars is None:  # pragma: no cover — validated by GeneratorConfig
        msg = "Alphabet generator requires 'chars'"
        raise GenerationError(msg)

    chars = list(config.chars)
    weights: np.ndarray | None = None
    if config.weights is not None:
        weights = np.array(config.weights, dtype=np.float64)
        weights = weights / weights.sum()

    values: list[str] = []
    for _ in range(n):
        length = _sample_length(config, rng)
        indices = rng.choice(len(chars), size=length, p=weights)
        values.append("".join(chars[i] for i in indices))

    return values


def _sample_length(config: GeneratorConfig, rng: np.random.Generator) -> int:
    """Sample a string length from the length distribution."""
    if config.length is None:
        return 10  # Default length

    ld = config.length
    mean = ld.mean if ld.mean is not None else 10.0
    std = ld.std if ld.std is not None else 1.0

    raw = rng.normal(loc=mean, scale=std)

    if ld.min is not None:
        raw = max(raw, ld.min)
    if ld.max is not None:
        raw = min(raw, ld.max)

    return max(1, round(raw))


def _generate_lorem(
    n: int, mean_length: float | None, rng: np.random.Generator
) -> list[str]:
    """Generate lorem ipsum placeholder text."""
    target_len = int(mean_length) if mean_length is not None else 40

    values: list[str] = []
    for _ in range(n):
        word_count = max(1, target_len // 6)
        indices = rng.integers(0, len(_LOREM_WORDS), size=word_count)
        text = " ".join(_LOREM_WORDS[i] for i in indices)
        values.append(text[:target_len])

    return values
