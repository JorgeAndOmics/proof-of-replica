"""Tests for post-generation hooks."""

from pathlib import Path

import polars as pl
import pytest
from click.testing import CliRunner

from proof_of_replica.cli.main import app
from proof_of_replica.core.column_schema import ColumnDefinition
from proof_of_replica.core.schema import Profile, save_profile
from proof_of_replica.engines.hooks import load_and_run_hook
from proof_of_replica.exceptions import GenerationError


class TestLoadAndRunHook:
    def test_valid_hook(self, tmp_path: Path):
        hook = tmp_path / "hook.py"
        hook.write_text(
            "import polars as pl\n"
            "def post_generate(df: pl.DataFrame) -> pl.DataFrame:\n"
            "    return df.head(5)\n"
        )
        df = pl.DataFrame({"x": list(range(20))})
        result = load_and_run_hook(hook, df)
        assert len(result) == 5

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(GenerationError, match="not found"):
            load_and_run_hook(tmp_path / "nonexistent.py", pl.DataFrame())

    def test_no_post_generate_raises(self, tmp_path: Path):
        hook = tmp_path / "hook.py"
        hook.write_text("x = 1\n")
        with pytest.raises(GenerationError, match="no 'post_generate'"):
            load_and_run_hook(hook, pl.DataFrame())

    def test_hook_returns_wrong_type_raises(self, tmp_path: Path):
        hook = tmp_path / "hook.py"
        hook.write_text("def post_generate(df):\n    return 'not a df'\n")
        with pytest.raises(GenerationError, match="must return a DataFrame"):
            load_and_run_hook(hook, pl.DataFrame({"x": [1]}))

    def test_hook_exception_wrapped(self, tmp_path: Path):
        hook = tmp_path / "hook.py"
        hook.write_text("def post_generate(df):\n    raise ValueError('boom')\n")
        with pytest.raises(GenerationError, match="Hook execution failed"):
            load_and_run_hook(hook, pl.DataFrame({"x": [1]}))

    def test_syntax_error_in_hook(self, tmp_path: Path):
        hook = tmp_path / "hook.py"
        hook.write_text("def post_generate(df):\n    return df[\n")
        with pytest.raises(GenerationError, match="Failed to load"):
            load_and_run_hook(hook, pl.DataFrame({"x": [1]}))


class TestHookCLI:
    def test_generate_with_hook(self, tmp_path: Path):
        runner = CliRunner()

        hook = tmp_path / "hook.py"
        hook.write_text(
            "import polars as pl\n"
            "def post_generate(df: pl.DataFrame) -> pl.DataFrame:\n"
            "    return df.head(5)\n"
        )

        p = Profile(
            version="0.2.0",
            seed=42,
            row_count=50,
            columns=[
                ColumnDefinition(
                    name="x", dtype="float64", stats={"mean": 0, "std": 1}
                ),
            ],
        )
        profile_path = tmp_path / "p.json"
        save_profile(p, profile_path)

        output = tmp_path / "out.parquet"
        result = runner.invoke(
            app,
            [
                "generate",
                str(profile_path),
                "-o",
                str(output),
                "--hook",
                str(hook),
                "--validate",
                "off",
            ],
        )
        assert result.exit_code == 0
        df = pl.read_parquet(output)
        assert len(df) == 5
