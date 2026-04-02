"""Tests for string column generators."""

import re

import numpy as np
import pytest

from proof_of_replica.core.column_schema import ColumnDefinition
from proof_of_replica.exceptions import GenerationError
from proof_of_replica.generators.string import generate_string


class TestRegexGenerator:
    def test_matches_pattern(self, rng):
        col = ColumnDefinition(
            name="gene_id",
            dtype="string",
            generator={"method": "regex", "pattern": r"ENSG\d{11}"},
        )
        result = generate_string(col, 100, rng)
        pattern = re.compile(r"^ENSG\d{11}$")
        for val in result.to_list():
            assert pattern.match(val), f"'{val}' doesn't match pattern"

    def test_unique_mode(self, rng):
        col = ColumnDefinition(
            name="id",
            dtype="string",
            generator={"method": "regex", "pattern": r"ID_\d{6}", "unique": True},
        )
        result = generate_string(col, 100, rng)
        assert result.n_unique() == 100

    def test_deterministic(self):
        col = ColumnDefinition(
            name="x",
            dtype="string",
            generator={"method": "regex", "pattern": r"[A-Z]{3}\d{3}"},
        )
        r1 = generate_string(col, 20, np.random.default_rng(42))
        r2 = generate_string(col, 20, np.random.default_rng(42))
        assert r1.to_list() == r2.to_list()


class TestAlphabetGenerator:
    def test_correct_charset(self, rng):
        col = ColumnDefinition(
            name="seq",
            dtype="string",
            generator={
                "method": "alphabet",
                "chars": "ACGT",
                "length": {"mean": 20, "std": 2, "min": 10},
            },
        )
        result = generate_string(col, 50, rng)
        for val in result.to_list():
            assert set(val) <= {"A", "C", "G", "T"}

    def test_with_weights(self, rng):
        col = ColumnDefinition(
            name="seq",
            dtype="string",
            generator={
                "method": "alphabet",
                "chars": "AB",
                "weights": [0.9, 0.1],
                "length": {"mean": 1000, "std": 0, "min": 1000, "max": 1000},
            },
        )
        result = generate_string(col, 1, rng)
        val = result.to_list()[0]
        a_frac = val.count("A") / len(val)
        assert abs(a_frac - 0.9) < 0.05

    def test_length_distribution(self, rng):
        col = ColumnDefinition(
            name="seq",
            dtype="string",
            generator={
                "method": "alphabet",
                "chars": "XY",
                "length": {"mean": 50, "std": 5, "min": 30, "max": 70},
            },
        )
        result = generate_string(col, 200, rng)
        lengths = [len(v) for v in result.to_list()]
        assert all(30 <= ln <= 70 for ln in lengths)

    def test_default_length(self, rng):
        col = ColumnDefinition(
            name="seq",
            dtype="string",
            generator={"method": "alphabet", "chars": "AB"},
        )
        result = generate_string(col, 5, rng)
        assert all(len(v) == 10 for v in result.to_list())


class TestPlaceholderGenerator:
    def test_constant_value(self, rng):
        col = ColumnDefinition(
            name="notes",
            dtype="string",
            generator={"method": "placeholder", "placeholder_value": "[REDACTED]"},
        )
        result = generate_string(col, 50, rng)
        assert all(v == "[REDACTED]" for v in result.to_list())

    def test_default_placeholder(self, rng):
        col = ColumnDefinition(
            name="notes",
            dtype="string",
            generator={"method": "placeholder"},
        )
        result = generate_string(col, 10, rng)
        assert all(v == "[REDACTED]" for v in result.to_list())


class TestLoremGenerator:
    def test_generates_text(self, rng):
        col = ColumnDefinition(
            name="desc",
            dtype="string",
            stats={"mean_length": 50.0},
            generator={"method": "lorem"},
        )
        result = generate_string(col, 20, rng)
        assert all(len(v) > 0 for v in result.to_list())

    def test_respects_mean_length(self, rng):
        col = ColumnDefinition(
            name="desc",
            dtype="string",
            stats={"mean_length": 30.0},
            generator={"method": "lorem"},
        )
        result = generate_string(col, 100, rng)
        avg_len = sum(len(v) for v in result.to_list()) / len(result)
        assert avg_len <= 35


class TestStringErrors:
    def test_error_on_missing_generator(self, rng):
        col = ColumnDefinition(name="x", dtype="string")
        with pytest.raises(GenerationError, match="generator config"):
            generate_string(col, 10, rng)
