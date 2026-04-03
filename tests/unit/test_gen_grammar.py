"""Tests for grammar string generator."""

import numpy as np
import pytest

from proof_of_replica.core.column_schema import ColumnDefinition, GeneratorConfig
from proof_of_replica.exceptions import GenerationError
from proof_of_replica.generators.grammar import generate_grammar_strings
from proof_of_replica.generators.string import generate_string


class TestGrammarBasic:
    def test_protein_identifier(self):
        rng = np.random.default_rng(42)
        col = ColumnDefinition(
            name="pid",
            dtype="string",
            generator={
                "method": "grammar",
                "rules": {
                    "start": ["{db}:{accession}"],
                    "db": ["SP", "TR"],
                    "accession": ["{letter}{digits}"],
                    "letter": ["P", "Q", "O"],
                    "digits": ["{d}{d}{d}{d}"],
                    "d": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
                },
            },
        )
        result = generate_string(col, 10, rng)
        assert len(result) == 10
        for val in result.to_list():
            assert ":" in val
            db, acc = val.split(":")
            assert db in ("SP", "TR")
            assert len(acc) == 5  # 1 letter + 4 digits

    def test_simple_choice(self):
        rng = np.random.default_rng(42)
        col = ColumnDefinition(
            name="x",
            dtype="string",
            generator={
                "method": "grammar",
                "rules": {
                    "start": ["hello", "world"],
                },
            },
        )
        result = generate_string(col, 100, rng)
        unique = set(result.to_list())
        assert unique <= {"hello", "world"}


class TestGrammarUnique:
    def test_unique_mode(self):
        rng = np.random.default_rng(42)
        col = ColumnDefinition(
            name="x",
            dtype="string",
            generator={
                "method": "grammar",
                "unique": True,
                "rules": {
                    "start": ["{a}{b}"],
                    "a": ["A", "B", "C", "D", "E"],
                    "b": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"],
                },
            },
        )
        result = generate_string(col, 20, rng)
        assert result.n_unique() == 20


class TestGrammarDeterminism:
    def test_deterministic(self):
        col = ColumnDefinition(
            name="x",
            dtype="string",
            generator={
                "method": "grammar",
                "rules": {"start": ["{a}"], "a": ["X", "Y", "Z"]},
            },
        )
        r1 = generate_string(col, 10, np.random.default_rng(42))
        r2 = generate_string(col, 10, np.random.default_rng(42))
        assert r1.to_list() == r2.to_list()


class TestGrammarEdgeCases:
    def test_missing_start_rule(self):
        config = GeneratorConfig(
            method="grammar",
            rules={"a": ["X"]},  # No "start" rule
        )
        with pytest.raises(GenerationError, match="'start' rule"):
            generate_grammar_strings(config, 1, np.random.default_rng(42))

    def test_unknown_rule_left_as_placeholder(self):
        """Unknown rule references are left as literal {name}."""
        rng = np.random.default_rng(42)
        col = ColumnDefinition(
            name="x",
            dtype="string",
            generator={
                "method": "grammar",
                "rules": {"start": ["{unknown}"]},
            },
        )
        result = generate_string(col, 1, rng)
        assert result.to_list()[0] == "{unknown}"

    def test_deep_recursion_raises(self):
        """Deeply recursive grammar hits depth limit."""

        config = GeneratorConfig(
            method="grammar",
            rules={"start": ["{start}"]},  # Infinite recursion
        )
        with pytest.raises(GenerationError, match="max depth"):
            generate_grammar_strings(config, 1, np.random.default_rng(42))
