"""Tests for the HTML reporting engine."""

import numpy as np
import polars as pl

from proof_of_replica.core.column_schema import ColumnDefinition
from proof_of_replica.core.schema import CorrelationConfig, Profile
from proof_of_replica.engines.reporting import (
    generate_fidelity_report,
    generate_validation_report,
)
from proof_of_replica.engines.validation import validate


def _make_profile(**kwargs: object) -> Profile:
    defaults = {"version": "0.2.0", "seed": 42}
    defaults.update(kwargs)  # type: ignore[arg-type]
    return Profile(**defaults)  # type: ignore[arg-type]


class TestGenerateFidelityReport:
    def test_basic_html_output(self):
        profile = _make_profile(
            row_count=50,
            columns=[
                ColumnDefinition(
                    name="x", dtype="float64", stats={"mean": 0, "std": 1}
                ),
            ],
        )
        rng = np.random.default_rng(42)
        df = pl.DataFrame({"x": rng.normal(0, 1, 50).tolist()})
        html = generate_fidelity_report(df, profile)

        assert "<!DOCTYPE html>" in html
        assert "Fidelity Report" in html
        assert "data:image/png;base64," in html

    def test_with_original(self):
        profile = _make_profile(
            columns=[
                ColumnDefinition(
                    name="x", dtype="float64", stats={"mean": 0, "std": 1}
                ),
            ],
        )
        rng = np.random.default_rng(42)
        replica = pl.DataFrame({"x": rng.normal(0, 1, 50).tolist()})
        original = pl.DataFrame({"x": rng.normal(0, 1, 50).tolist()})
        html = generate_fidelity_report(replica, profile, original=original)

        assert "Original" in html or "data:image/png;base64," in html

    def test_categorical_column(self):
        profile = _make_profile(
            columns=[
                ColumnDefinition(
                    name="c",
                    dtype="categorical",
                    stats={"cardinality": 2, "value_counts": {"a": 0.5, "b": 0.5}},
                ),
            ],
        )
        df = pl.DataFrame({"c": ["a", "b"] * 25})
        html = generate_fidelity_report(df, profile)
        assert "data:image/png;base64," in html

    def test_with_correlations(self):
        rng = np.random.default_rng(42)
        profile = _make_profile(
            columns=[
                ColumnDefinition(
                    name="x", dtype="float64", stats={"mean": 0, "std": 1}
                ),
                ColumnDefinition(
                    name="y", dtype="float64", stats={"mean": 0, "std": 1}
                ),
            ],
            correlations=CorrelationConfig(
                columns=["x", "y"], matrix=[[1.0, 0.5], [0.5, 1.0]]
            ),
        )
        df = pl.DataFrame(
            {
                "x": rng.normal(0, 1, 100).tolist(),
                "y": rng.normal(0, 1, 100).tolist(),
            }
        )
        html = generate_fidelity_report(df, profile)
        assert "Correlation" in html

    def test_boolean_column_no_plot(self):
        profile = _make_profile(
            columns=[
                ColumnDefinition(
                    name="b", dtype="boolean", stats={"true_fraction": 0.5}
                ),
            ],
        )
        df = pl.DataFrame({"b": [True, False] * 25})
        html = generate_fidelity_report(df, profile)
        assert "<!DOCTYPE html>" in html


class TestGenerateValidationReport:
    def test_basic(self):
        profile = _make_profile(
            columns=[
                ColumnDefinition(
                    name="x", dtype="float64", stats={"mean": 0, "std": 1}
                ),
            ],
        )
        df = pl.DataFrame({"x": [0.1, -0.2, 0.3]})
        result = validate(df, profile)
        html = generate_validation_report(result)
        assert "<!DOCTYPE html>" in html
        assert "Fidelity Summary" in html
