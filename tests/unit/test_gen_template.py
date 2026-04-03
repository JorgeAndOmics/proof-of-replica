"""Tests for template string generator."""

import re

import numpy as np

from proof_of_replica.core.column_schema import ColumnDefinition
from proof_of_replica.generators.string import generate_string


class TestTemplateBasic:
    def test_simple_template(self):
        rng = np.random.default_rng(42)
        col = ColumnDefinition(
            name="x",
            dtype="string",
            generator={
                "method": "template",
                "template_str": "{a}-{b}",
                "parts": {
                    "a": {"type": "choice", "values": ["X", "Y"]},
                    "b": {"type": "integer_range", "min": 1, "max": 100},
                },
            },
        )
        result = generate_string(col, 10, rng)
        assert len(result) == 10
        for val in result.to_list():
            assert "-" in val
            prefix = val.split("-")[0]
            assert prefix in ("X", "Y")

    def test_genomic_locus(self):
        rng = np.random.default_rng(42)
        col = ColumnDefinition(
            name="locus",
            dtype="string",
            generator={
                "method": "template",
                "template_str": "{chrom}:{start}-{end}",
                "parts": {
                    "chrom": {"type": "choice", "values": ["chr1", "chr2", "chr3"]},
                    "start": {"type": "integer_range", "min": 1000, "max": 100000},
                    "end": {
                        "type": "relative",
                        "base": "start",
                        "offset_min": 100,
                        "offset_max": 5000,
                    },
                },
            },
        )
        result = generate_string(col, 20, rng)
        for val in result.to_list():
            # Should match pattern like chr1:12345-17345
            assert re.match(r"chr\d+:\d+-\d+", val)
            parts = val.split(":")
            start, end = parts[1].split("-")
            assert int(end) > int(start)

    def test_regex_parts(self):
        rng = np.random.default_rng(42)
        col = ColumnDefinition(
            name="id",
            dtype="string",
            generator={
                "method": "template",
                "template_str": "TCGA-{tss}-{part}",
                "parts": {
                    "tss": {"type": "regex", "pattern": "[A-Z]{2}"},
                    "part": {"type": "regex", "pattern": "[A-Z0-9]{4}"},
                },
            },
        )
        result = generate_string(col, 5, rng)
        for val in result.to_list():
            assert val.startswith("TCGA-")

    def test_float_range_with_precision(self):
        rng = np.random.default_rng(42)
        col = ColumnDefinition(
            name="x",
            dtype="string",
            generator={
                "method": "template",
                "template_str": "val={v}",
                "parts": {
                    "v": {
                        "type": "float_range",
                        "min": 0.0,
                        "max": 1.0,
                        "precision": 3,
                    },
                },
            },
        )
        result = generate_string(col, 5, rng)
        for val in result.to_list():
            assert val.startswith("val=")
            float_part = val.split("=")[1]
            assert len(float_part.split(".")[1]) == 3


class TestTemplateSequential:
    def test_sequential_counter(self):
        rng = np.random.default_rng(42)
        col = ColumnDefinition(
            name="x",
            dtype="string",
            generator={
                "method": "template",
                "template_str": "sample_{n}",
                "parts": {
                    "n": {"type": "sequential", "start": 1, "zero_pad": 3},
                },
            },
        )
        result = generate_string(col, 5, rng)
        assert result.to_list() == [
            "sample_001",
            "sample_002",
            "sample_003",
            "sample_004",
            "sample_005",
        ]


class TestTemplateNested:
    def test_nested_template(self):
        rng = np.random.default_rng(42)
        col = ColumnDefinition(
            name="uri",
            dtype="string",
            generator={
                "method": "template",
                "template_str": "s3://{bucket}/{sample}",
                "parts": {
                    "bucket": {"type": "choice", "values": ["prod", "staging"]},
                    "sample": {
                        "type": "template",
                        "template": "sample_{n}",
                        "parts": {
                            "n": {"type": "sequential", "start": 1, "zero_pad": 3}
                        },
                    },
                },
            },
        )
        result = generate_string(col, 3, rng)
        for val in result.to_list():
            assert val.startswith("s3://")
            assert "sample_" in val


class TestTemplateAlphabet:
    def test_alphabet_part(self):
        rng = np.random.default_rng(42)
        col = ColumnDefinition(
            name="x",
            dtype="string",
            generator={
                "method": "template",
                "template_str": "SEQ:{seq}",
                "parts": {
                    "seq": {"type": "alphabet", "chars": "ACGT", "length": 20},
                },
            },
        )
        result = generate_string(col, 5, rng)
        for val in result.to_list():
            assert val.startswith("SEQ:")
            seq = val.split(":")[1]
            assert len(seq) == 20
            assert set(seq) <= {"A", "C", "G", "T"}


class TestTemplateUnique:
    def test_unique_mode(self):
        rng = np.random.default_rng(42)
        col = ColumnDefinition(
            name="x",
            dtype="string",
            generator={
                "method": "template",
                "template_str": "{a}-{b}",
                "unique": True,
                "parts": {
                    "a": {"type": "choice", "values": ["A", "B", "C"]},
                    "b": {"type": "integer_range", "min": 1, "max": 1000},
                },
            },
        )
        result = generate_string(col, 50, rng)
        assert result.n_unique() == 50


class TestTemplateDeterminism:
    def test_deterministic(self):
        col = ColumnDefinition(
            name="x",
            dtype="string",
            generator={
                "method": "template",
                "template_str": "{a}:{b}",
                "parts": {
                    "a": {"type": "choice", "values": ["X", "Y"]},
                    "b": {"type": "integer_range", "min": 1, "max": 100},
                },
            },
        )
        r1 = generate_string(col, 10, np.random.default_rng(42))
        r2 = generate_string(col, 10, np.random.default_rng(42))
        assert r1.to_list() == r2.to_list()
