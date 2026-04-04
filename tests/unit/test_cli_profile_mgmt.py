"""Tests for profile management CLI commands."""

import json
from pathlib import Path

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
