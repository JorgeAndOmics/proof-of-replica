"""Tests for file I/O utilities."""

from pathlib import Path

import polars as pl
import pytest

from proof_of_replica.exceptions import FileIOError
from proof_of_replica.utils.io import detect_separator, read_dataframe, write_dataframe


@pytest.fixture
def sample_df() -> pl.DataFrame:
    return pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})


class TestDetectSeparator:
    def test_detects_comma(self, tmp_path: Path):
        path = tmp_path / "data.csv"
        path.write_text("a,b,c\n1,2,3\n")
        assert detect_separator(path) == ","

    def test_detects_tab(self, tmp_path: Path):
        path = tmp_path / "data.tsv"
        path.write_text("a\tb\tc\n1\t2\t3\n")
        assert detect_separator(path) == "\t"

    def test_prefers_tab_when_tied(self, tmp_path: Path):
        path = tmp_path / "data.txt"
        path.write_text("a,b\tc\n")
        assert detect_separator(path) == "\t"

    def test_error_on_no_separators(self, tmp_path: Path):
        path = tmp_path / "data.txt"
        path.write_text("abcdef\n")
        with pytest.raises(FileIOError, match="Cannot detect"):
            detect_separator(path)

    def test_error_on_missing_file(self, tmp_path: Path):
        with pytest.raises(FileIOError, match="Cannot read"):
            detect_separator(tmp_path / "missing.csv")


class TestReadDataframe:
    def test_read_csv(self, tmp_path: Path, sample_df: pl.DataFrame):
        path = tmp_path / "data.csv"
        sample_df.write_csv(path)
        result = read_dataframe(path)
        assert result.shape == (3, 2)

    def test_read_tsv(self, tmp_path: Path, sample_df: pl.DataFrame):
        path = tmp_path / "data.tsv"
        sample_df.write_csv(path, separator="\t")
        result = read_dataframe(path)
        assert result.shape == (3, 2)

    def test_read_parquet(self, tmp_path: Path, sample_df: pl.DataFrame):
        path = tmp_path / "data.parquet"
        sample_df.write_parquet(path)
        result = read_dataframe(path)
        assert result.shape == (3, 2)

    def test_read_csv_with_n_rows(self, tmp_path: Path, sample_df: pl.DataFrame):
        path = tmp_path / "data.csv"
        sample_df.write_csv(path)
        result = read_dataframe(path, n_rows=2)
        assert len(result) == 2

    def test_read_parquet_with_n_rows(self, tmp_path: Path, sample_df: pl.DataFrame):
        path = tmp_path / "data.parquet"
        sample_df.write_parquet(path)
        result = read_dataframe(path, n_rows=1)
        assert len(result) == 1

    def test_read_csv_custom_separator(self, tmp_path: Path, sample_df: pl.DataFrame):
        path = tmp_path / "data.csv"
        sample_df.write_csv(path, separator="|")
        result = read_dataframe(path, separator="|")
        assert result.shape == (3, 2)

    def test_error_unsupported_format(self, tmp_path: Path):
        path = tmp_path / "data.xlsx"
        path.write_text("dummy")
        with pytest.raises(FileIOError, match="Unsupported"):
            read_dataframe(path)

    def test_error_missing_file(self, tmp_path: Path):
        with pytest.raises(FileIOError, match="Failed to read"):
            read_dataframe(tmp_path / "missing.csv")


class TestWriteDataframe:
    def test_write_csv(self, tmp_path: Path, sample_df: pl.DataFrame):
        path = tmp_path / "out.csv"
        write_dataframe(sample_df, path)
        result = read_dataframe(path)
        assert result.shape == (3, 2)

    def test_write_tsv(self, tmp_path: Path, sample_df: pl.DataFrame):
        path = tmp_path / "out.tsv"
        write_dataframe(sample_df, path)
        result = read_dataframe(path)
        assert result.shape == (3, 2)

    def test_write_parquet(self, tmp_path: Path, sample_df: pl.DataFrame):
        path = tmp_path / "out.parquet"
        write_dataframe(sample_df, path)
        result = read_dataframe(path)
        assert result.shape == (3, 2)

    def test_round_trip_csv(self, tmp_path: Path, sample_df: pl.DataFrame):
        path = tmp_path / "rt.csv"
        write_dataframe(sample_df, path)
        result = read_dataframe(path)
        assert result["a"].to_list() == [1, 2, 3]
        assert result["b"].to_list() == ["x", "y", "z"]

    def test_error_unsupported_format(self, tmp_path: Path, sample_df: pl.DataFrame):
        with pytest.raises(FileIOError, match="Unsupported"):
            write_dataframe(sample_df, tmp_path / "out.xlsx")
