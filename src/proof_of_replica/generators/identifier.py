"""Identifier column generator (sequential, uuid, prefix_increment)."""

import uuid

import numpy as np
import polars as pl

from proof_of_replica.core.column_schema import ColumnDefinition
from proof_of_replica.core.enums import GeneratorMethod
from proof_of_replica.exceptions import GenerationError


def generate_identifier(
    col: ColumnDefinition,
    n: int,
    rng: np.random.Generator,
) -> pl.Series:
    """Generate an identifier column.

    Args:
        col: Column definition with a GeneratorConfig specifying the method.
        n: Number of rows.
        rng: Seeded RNG (used for uuid generation).

    Returns:
        Polars Series of Utf8 dtype with unique identifiers.

    Raises:
        GenerationError: If generator config is missing or method is unsupported.
    """
    if col.generator is None:
        msg = f"Column '{col.name}': identifier generator requires a generator config"
        raise GenerationError(msg)

    method = col.generator.method

    if method in (GeneratorMethod.SEQUENTIAL, GeneratorMethod.PREFIX_INCREMENT):
        return _generate_sequential(col, n)

    if method == GeneratorMethod.UUID:
        return _generate_uuid(col, n, rng)

    msg = f"Column '{col.name}': unsupported identifier method '{method}'"
    raise GenerationError(msg)


def _generate_sequential(col: ColumnDefinition, n: int) -> pl.Series:
    """Generate sequential identifiers with optional prefix and zero-padding."""
    assert col.generator is not None
    prefix = col.generator.prefix or ""
    zero_pad = col.generator.zero_pad or 0

    if zero_pad > 0:
        values = [f"{prefix}{i:0{zero_pad}d}" for i in range(1, n + 1)]
    else:
        values = [f"{prefix}{i}" for i in range(1, n + 1)]

    return pl.Series(name=col.name, values=values, dtype=pl.Utf8)


def _generate_uuid(
    col: ColumnDefinition, n: int, rng: np.random.Generator
) -> pl.Series:
    """Generate deterministic UUIDs from RNG bytes."""
    values = []
    for _ in range(n):
        random_bytes = rng.bytes(16)
        u = uuid.UUID(bytes=random_bytes, version=4)
        values.append(str(u))

    return pl.Series(name=col.name, values=values, dtype=pl.Utf8)
