"""Tests for identifier column generator."""

import re

import numpy as np
import pytest

from proof_of_replica.core.column_schema import ColumnDefinition
from proof_of_replica.exceptions import GenerationError
from proof_of_replica.generators.identifier import generate_identifier


class TestGenerateSequential:
    def test_basic_sequential(self, rng):
        col = ColumnDefinition(
            name="id",
            dtype="string",
            role="identifier",
            generator={"method": "sequential", "prefix": "S_", "zero_pad": 4},
        )
        result = generate_identifier(col, 5, rng)
        assert result.to_list() == ["S_0001", "S_0002", "S_0003", "S_0004", "S_0005"]

    def test_no_prefix(self, rng):
        col = ColumnDefinition(
            name="id",
            dtype="string",
            role="identifier",
            generator={"method": "sequential"},
        )
        result = generate_identifier(col, 3, rng)
        assert result.to_list() == ["1", "2", "3"]

    def test_prefix_increment_alias(self, rng):
        col = ColumnDefinition(
            name="id",
            dtype="string",
            role="identifier",
            generator={"method": "prefix_increment", "prefix": "ID_"},
        )
        result = generate_identifier(col, 3, rng)
        assert result.to_list() == ["ID_1", "ID_2", "ID_3"]


class TestGenerateUuid:
    def test_returns_uuid_strings(self, rng):
        col = ColumnDefinition(
            name="uid",
            dtype="string",
            role="identifier",
            generator={"method": "uuid"},
        )
        result = generate_identifier(col, 10, rng)
        assert len(result) == 10
        uuid_pattern = re.compile(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[0-9a-f]{4}-[0-9a-f]{12}$"
        )
        for val in result.to_list():
            assert uuid_pattern.match(val)

    def test_deterministic(self):
        col = ColumnDefinition(
            name="uid",
            dtype="string",
            role="identifier",
            generator={"method": "uuid"},
        )
        r1 = generate_identifier(col, 10, np.random.default_rng(42))
        r2 = generate_identifier(col, 10, np.random.default_rng(42))
        assert r1.to_list() == r2.to_list()

    def test_all_unique(self, rng):
        col = ColumnDefinition(
            name="uid",
            dtype="string",
            role="identifier",
            generator={"method": "uuid"},
        )
        result = generate_identifier(col, 1000, rng)
        assert result.n_unique() == 1000


class TestIdentifierErrors:
    def test_error_on_missing_generator(self, rng):
        col = ColumnDefinition(name="id", dtype="string", role="identifier")
        with pytest.raises(GenerationError, match="generator config"):
            generate_identifier(col, 10, rng)

    def test_error_on_unsupported_method(self, rng):
        col = ColumnDefinition(
            name="id",
            dtype="string",
            role="identifier",
            generator={"method": "regex", "pattern": "X\\d+"},
        )
        with pytest.raises(GenerationError, match="unsupported"):
            generate_identifier(col, 10, rng)
