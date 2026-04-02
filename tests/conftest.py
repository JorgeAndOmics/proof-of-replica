"""Shared test fixtures for proof_of_replica."""

import numpy as np
import pytest

from proof_of_replica.core.column_schema import ColumnDefinition


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded RNG for deterministic tests."""
    return np.random.default_rng(42)


@pytest.fixture
def boolean_col() -> ColumnDefinition:
    return ColumnDefinition(
        name="is_control",
        dtype="boolean",
        stats={"true_fraction": 0.4},
    )


@pytest.fixture
def categorical_col() -> ColumnDefinition:
    return ColumnDefinition(
        name="color",
        dtype="categorical",
        stats={
            "cardinality": 3,
            "value_counts": {"red": 0.5, "green": 0.3, "blue": 0.2},
        },
    )


@pytest.fixture
def numeric_col() -> ColumnDefinition:
    return ColumnDefinition(
        name="age",
        dtype="float64",
        stats={"mean": 50.0, "std": 10.0, "min": 18.0, "max": 90.0},
    )


@pytest.fixture
def int_col() -> ColumnDefinition:
    return ColumnDefinition(
        name="count",
        dtype="int64",
        stats={"mean": 20.0, "std": 5.0},
    )
