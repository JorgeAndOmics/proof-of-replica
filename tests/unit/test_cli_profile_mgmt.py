"""Tests for profile management CLI commands."""

import json
from pathlib import Path

import polars as pl
import pytest
from click.testing import CliRunner

from proof_of_replica.cli.main import app
from proof_of_replica.core.column_schema import ColumnDefinition
from proof_of_replica.core.schema import Profile, save_profile


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _save(p: Profile, path: Path) -> Path:
    save_profile(p, path)
    return path


# ── profile-diff ─────────────────────────────────────────────


class TestProfileDiff:
    def test_identical_profiles(self, runner, tmp_path):
        p = Profile(
            version="0.2.0",
            row_count=100,
            columns=[
                ColumnDefinition(name="x", dtype="float64", stats={"mean": 0, "std": 1})
            ],
        )
        a = _save(p, tmp_path / "a.json")
        b = _save(p, tmp_path / "b.json")
        result = runner.invoke(app, ["profile-diff", str(a), str(b)])
        assert result.exit_code == 0
        assert "identical" in result.output

    def test_differing_stats(self, runner, tmp_path):
        pa = Profile(
            version="0.2.0",
            row_count=100,
            columns=[
                ColumnDefinition(name="x", dtype="float64", stats={"mean": 0, "std": 1})
            ],
        )
        pb = Profile(
            version="0.2.0",
            row_count=200,
            columns=[
                ColumnDefinition(
                    name="x", dtype="float64", stats={"mean": 10, "std": 5}
                )
            ],
        )
        a = _save(pa, tmp_path / "a.json")
        b = _save(pb, tmp_path / "b.json")
        result = runner.invoke(app, ["profile-diff", str(a), str(b)])
        assert result.exit_code == 0
        assert "mean" in result.output

    def test_missing_column(self, runner, tmp_path):
        pa = Profile(
            version="0.2.0",
            columns=[
                ColumnDefinition(
                    name="x", dtype="float64", stats={"mean": 0, "std": 1}
                ),
                ColumnDefinition(
                    name="y", dtype="float64", stats={"mean": 0, "std": 1}
                ),
            ],
        )
        pb = Profile(
            version="0.2.0",
            columns=[
                ColumnDefinition(name="x", dtype="float64", stats={"mean": 0, "std": 1})
            ],
        )
        a = _save(pa, tmp_path / "a.json")
        b = _save(pb, tmp_path / "b.json")
        result = runner.invoke(app, ["profile-diff", str(a), str(b)])
        assert result.exit_code == 0
        assert "only_in_a" in result.output or "only in A" in result.output

    def test_json_format(self, runner, tmp_path):
        p = Profile(
            version="0.2.0",
            columns=[
                ColumnDefinition(name="x", dtype="float64", stats={"mean": 0, "std": 1})
            ],
        )
        a = _save(p, tmp_path / "a.json")
        b = _save(p, tmp_path / "b.json")
        result = runner.invoke(
            app, ["profile-diff", str(a), str(b), "--format", "json"]
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "columns" in data

    def test_report_to_file(self, runner, tmp_path):
        p = Profile(
            version="0.2.0",
            columns=[
                ColumnDefinition(name="x", dtype="float64", stats={"mean": 0, "std": 1})
            ],
        )
        a = _save(p, tmp_path / "a.json")
        b = _save(p, tmp_path / "b.json")
        report = tmp_path / "diff.txt"
        result = runner.invoke(
            app, ["profile-diff", str(a), str(b), "--report", str(report)]
        )
        assert result.exit_code == 0
        assert report.exists()


# ── profile-merge ────────────────────────────────────────────


class TestProfileMerge:
    def test_merge_two_profiles(self, runner, tmp_path):
        pa = Profile(
            version="0.2.0",
            row_count=100,
            columns=[
                ColumnDefinition(
                    name="x", dtype="float64", stats={"mean": 10, "std": 2}
                )
            ],
        )
        pb = Profile(
            version="0.2.0",
            row_count=200,
            columns=[
                ColumnDefinition(
                    name="x", dtype="float64", stats={"mean": 20, "std": 4}
                )
            ],
        )
        a = _save(pa, tmp_path / "a.json")
        b = _save(pb, tmp_path / "b.json")
        out = tmp_path / "merged.json"
        result = runner.invoke(app, ["profile-merge", str(a), str(b), "-o", str(out)])
        assert result.exit_code == 0, result.output
        assert out.exists()
        merged = json.loads(out.read_text())
        # Mean should be average of 10 and 20
        assert 14 < merged["columns"][0]["stats"]["mean"] < 16

    def test_merge_with_weights(self, runner, tmp_path):
        pa = Profile(
            version="0.2.0",
            row_count=100,
            columns=[
                ColumnDefinition(name="x", dtype="float64", stats={"mean": 0, "std": 1})
            ],
        )
        pb = Profile(
            version="0.2.0",
            row_count=100,
            columns=[
                ColumnDefinition(
                    name="x", dtype="float64", stats={"mean": 100, "std": 10}
                )
            ],
        )
        a = _save(pa, tmp_path / "a.json")
        b = _save(pb, tmp_path / "b.json")
        out = tmp_path / "merged.json"
        result = runner.invoke(
            app,
            [
                "profile-merge",
                str(a),
                str(b),
                "-o",
                str(out),
                "--weights",
                "0.9",
                "--weights",
                "0.1",
            ],
        )
        assert result.exit_code == 0
        merged = json.loads(out.read_text())
        # Heavily weighted toward profile A (mean=0)
        assert merged["columns"][0]["stats"]["mean"] < 20

    def test_merge_categorical_union(self, runner, tmp_path):
        pa = Profile(
            version="0.2.0",
            columns=[
                ColumnDefinition(
                    name="c",
                    dtype="categorical",
                    stats={"value_counts": {"a": 0.5, "b": 0.5}},
                )
            ],
        )
        pb = Profile(
            version="0.2.0",
            columns=[
                ColumnDefinition(
                    name="c",
                    dtype="categorical",
                    stats={"value_counts": {"b": 0.5, "c": 0.5}},
                )
            ],
        )
        a = _save(pa, tmp_path / "a.json")
        b = _save(pb, tmp_path / "b.json")
        out = tmp_path / "merged.json"
        result = runner.invoke(app, ["profile-merge", str(a), str(b), "-o", str(out)])
        assert result.exit_code == 0
        merged = json.loads(out.read_text())
        vc = merged["columns"][0]["stats"]["value_counts"]
        assert "a" in vc
        assert "b" in vc
        assert "c" in vc

    def test_error_single_profile(self, runner, tmp_path):
        p = Profile(
            version="0.2.0",
            columns=[
                ColumnDefinition(name="x", dtype="float64", stats={"mean": 0, "std": 1})
            ],
        )
        a = _save(p, tmp_path / "a.json")
        result = runner.invoke(
            app, ["profile-merge", str(a), "-o", str(tmp_path / "out.json")]
        )
        assert result.exit_code != 0


# ── profile-redact ───────────────────────────────────────────


class TestProfileRedact:
    def test_drop_column(self, runner, tmp_path):
        p = Profile(
            version="0.2.0",
            columns=[
                ColumnDefinition(
                    name="x", dtype="float64", stats={"mean": 0, "std": 1}
                ),
                ColumnDefinition(
                    name="secret", dtype="float64", stats={"mean": 99, "std": 1}
                ),
            ],
        )
        src = _save(p, tmp_path / "p.json")
        out = tmp_path / "redacted.json"
        result = runner.invoke(
            app,
            [
                "profile-redact",
                str(src),
                "-o",
                str(out),
                "--drop-columns",
                "secret",
            ],
        )
        assert result.exit_code == 0
        data = json.loads(out.read_text())
        names = [c["name"] for c in data["columns"]]
        assert "secret" not in names
        assert "x" in names

    def test_drop_stats(self, runner, tmp_path):
        p = Profile(
            version="0.2.0",
            columns=[
                ColumnDefinition(
                    name="x",
                    dtype="float64",
                    stats={
                        "mean": 50,
                        "std": 10,
                        "min": 0,
                        "max": 100,
                        "percentiles": {"25": 40, "75": 60},
                    },
                ),
            ],
        )
        src = _save(p, tmp_path / "p.json")
        out = tmp_path / "redacted.json"
        result = runner.invoke(
            app,
            [
                "profile-redact",
                str(src),
                "-o",
                str(out),
                "--drop-stats",
                "percentiles",
                "--drop-stats",
                "min",
                "--drop-stats",
                "max",
            ],
        )
        assert result.exit_code == 0
        data = json.loads(out.read_text())
        stats = data["columns"][0]["stats"]
        assert "percentiles" not in stats
        assert "min" not in stats
        assert "mean" in stats

    def test_rounding(self, runner, tmp_path):
        p = Profile(
            version="0.2.0",
            columns=[
                ColumnDefinition(
                    name="x",
                    dtype="float64",
                    stats={"mean": 3.14159265, "std": 1.23456789},
                ),
            ],
        )
        src = _save(p, tmp_path / "p.json")
        out = tmp_path / "redacted.json"
        result = runner.invoke(
            app,
            ["profile-redact", str(src), "-o", str(out), "--round-to", "3"],
        )
        assert result.exit_code == 0
        data = json.loads(out.read_text())
        mean = data["columns"][0]["stats"]["mean"]
        assert mean == 3.14


# ── convert ──────────────────────────────────────────────────


class TestConvert:
    def test_csv_to_parquet(self, runner, tmp_path):
        df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        csv_path = tmp_path / "data.csv"
        df.write_csv(csv_path)

        out = tmp_path / "data.parquet"
        result = runner.invoke(app, ["convert", str(csv_path), "-o", str(out)])
        assert result.exit_code == 0
        assert out.exists()
        loaded = pl.read_parquet(out)
        assert loaded.shape == (3, 2)

    def test_parquet_to_tsv(self, runner, tmp_path):
        df = pl.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
        pq = tmp_path / "data.parquet"
        df.write_parquet(pq)

        out = tmp_path / "data.tsv"
        result = runner.invoke(app, ["convert", str(pq), "-o", str(out)])
        assert result.exit_code == 0
        assert out.exists()
        assert "\t" in out.read_text()
