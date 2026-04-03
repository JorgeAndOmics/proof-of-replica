"""File I/O utilities for reading and writing tabular data."""

from pathlib import Path

import polars as pl

from proof_of_replica.exceptions import FileIOError

_SEPARATOR_MAP = {
    ".csv": ",",
    ".tsv": "\t",
}


def detect_separator(path: Path) -> str:
    """Auto-detect column separator by sniffing the first line.

    Args:
        path: Path to a delimited text file.

    Returns:
        The detected separator character.

    Raises:
        FileIOError: If no separator can be detected.
    """
    try:
        first_line = path.read_text(encoding="utf-8").split("\n", maxsplit=1)[0]
    except OSError as exc:
        msg = f"Cannot read file for separator detection: {path}"
        raise FileIOError(msg) from exc

    tab_count = first_line.count("\t")
    comma_count = first_line.count(",")

    if tab_count > 0 and tab_count >= comma_count:
        return "\t"
    if comma_count > 0:
        return ","

    msg = f"Cannot detect separator in {path} (no tabs or commas found)"
    raise FileIOError(msg)


def read_dataframe(
    path: Path,
    *,
    separator: str | None = None,
    n_rows: int | None = None,
) -> pl.DataFrame:
    """Read a tabular file into a Polars DataFrame.

    Args:
        path: Path to a .csv, .tsv, or .parquet file.
        separator: Column separator for CSV/TSV. Auto-detected if None.
        n_rows: Maximum number of rows to read. None reads all.

    Returns:
        A Polars DataFrame.

    Raises:
        FileIOError: If the file cannot be read or format is unsupported.
    """
    suffix = path.suffix.lower()

    try:
        if suffix == ".parquet":
            df = pl.read_parquet(path)
            if n_rows is not None:
                df = df.head(n_rows)
            return df

        if suffix in (".csv", ".tsv"):
            sep = separator or _SEPARATOR_MAP.get(suffix) or detect_separator(path)
            return pl.read_csv(path, separator=sep, n_rows=n_rows)

        msg = f"Unsupported file format: {suffix}"
        raise FileIOError(msg)
    except FileIOError:
        raise
    except Exception as exc:
        msg = f"Failed to read {path}: {exc}"
        raise FileIOError(msg) from exc


def write_dataframe(
    df: pl.DataFrame,
    path: Path,
    *,
    separator: str | None = None,
) -> None:
    """Write a Polars DataFrame to a tabular file.

    Args:
        df: DataFrame to write.
        path: Output file path (.csv, .tsv, or .parquet).
        separator: Column separator for CSV/TSV output.

    Raises:
        FileIOError: If the file cannot be written or format is unsupported.
    """
    suffix = path.suffix.lower()

    try:
        if suffix == ".parquet":
            df.write_parquet(path)
        elif suffix in (".csv", ".tsv"):
            sep = separator or _SEPARATOR_MAP.get(suffix, ",")
            df.write_csv(path, separator=sep)
        else:
            msg = f"Unsupported output format: {suffix}"
            raise FileIOError(msg)
    except FileIOError:
        raise
    except Exception as exc:
        msg = f"Failed to write {path}: {exc}"
        raise FileIOError(msg) from exc
