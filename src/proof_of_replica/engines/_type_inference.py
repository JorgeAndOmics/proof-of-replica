"""Type inference for profiling — classifies column dtypes."""

import polars as pl

from proof_of_replica.core.enums import ColumnDtype
from proof_of_replica.core.schema import ProfilerConfig

_BOOLEAN_VALUES = {
    frozenset({"0", "1"}),
    frozenset({"true", "false"}),
    frozenset({"yes", "no"}),
    frozenset({"t", "f"}),
    frozenset({"y", "n"}),
}


def infer_dtype(
    series: pl.Series,
    n_rows: int,
    config: ProfilerConfig,
) -> ColumnDtype:
    """Infer the semantic dtype of a column.

    Attempts detection in priority order: boolean → integer → float →
    date → categorical → identifier → string.

    Args:
        series: Column data.
        n_rows: Total row count in the dataset (used for cardinality thresholds).
        config: Profiler configuration with thresholds.

    Returns:
        The inferred ColumnDtype.
    """
    classifiers = [
        _is_boolean,
        _is_integer,
        _is_float,
        _is_date,
        _is_categorical,
        _is_identifier,
    ]
    for classifier in classifiers:
        result = classifier(series, n_rows, config)
        if result is not None:
            return result

    return ColumnDtype.STRING


def _is_boolean(
    series: pl.Series, _n_rows: int, _config: ProfilerConfig
) -> ColumnDtype | None:
    """Check if column has exactly 2 unique non-null values interpretable as bool."""
    non_null = series.drop_nulls()
    unique = non_null.unique()

    if len(unique) != 2:
        return None

    # Check native boolean dtype
    if series.dtype == pl.Boolean:
        return ColumnDtype.BOOLEAN

    # Check numeric 0/1
    vals = set(unique.to_list())
    if series.dtype.is_integer() and vals == {0, 1}:
        return ColumnDtype.BOOLEAN

    # Check string representations
    str_vals = {str(v).lower() for v in vals}
    if str_vals in _BOOLEAN_VALUES:
        return ColumnDtype.BOOLEAN  # type: ignore[unreachable]

    return None


def _is_integer(
    series: pl.Series, _n_rows: int, _config: ProfilerConfig
) -> ColumnDtype | None:
    """Check if column contains integer values."""
    if series.dtype in (
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
    ):
        return ColumnDtype.INT64

    if series.dtype == pl.Utf8:
        non_null = series.drop_nulls()
        if len(non_null) == 0:
            return None
        try:
            casted = non_null.cast(pl.Int64, strict=False)
            success_rate = 1.0 - (casted.null_count() / len(non_null))
            if success_rate > 0.95:
                return ColumnDtype.INT64
        except Exception:
            return None

    return None


def _is_float(
    series: pl.Series, _n_rows: int, _config: ProfilerConfig
) -> ColumnDtype | None:
    """Check if column contains float values."""
    if series.dtype in (pl.Float32, pl.Float64):
        return ColumnDtype.FLOAT64

    if series.dtype == pl.Utf8:
        non_null = series.drop_nulls()
        if len(non_null) == 0:
            return None
        try:
            casted = non_null.cast(pl.Float64, strict=False)
            success_rate = 1.0 - (casted.null_count() / len(non_null))
            if success_rate > 0.95:
                return ColumnDtype.FLOAT64
        except Exception:
            return None

    return None


def _is_date(
    series: pl.Series, _n_rows: int, _config: ProfilerConfig
) -> ColumnDtype | None:
    """Check if column contains date values."""
    if series.dtype in (pl.Date, pl.Datetime):
        return ColumnDtype.DATE

    if series.dtype != pl.Utf8:
        return None

    non_null = series.drop_nulls()
    if len(non_null) == 0:
        return None

    # Try ISO date parsing
    try:
        parsed = non_null.str.to_date(format="%Y-%m-%d", strict=False)
        success_rate = 1.0 - (parsed.null_count() / len(non_null))
    except Exception:
        return None

    return ColumnDtype.DATE if success_rate > 0.90 else None


def _is_categorical(
    series: pl.Series, n_rows: int, config: ProfilerConfig
) -> ColumnDtype | None:
    """Check if string column has low cardinality."""
    if series.dtype != pl.Utf8:
        return None

    cardinality = series.drop_nulls().n_unique()
    max_card = min(
        config.categorical_max_cardinality,
        int(config.categorical_max_fraction * n_rows),
    )

    if cardinality <= max_card and cardinality < n_rows:
        return ColumnDtype.CATEGORICAL

    return None


def _is_identifier(
    series: pl.Series, n_rows: int, _config: ProfilerConfig
) -> ColumnDtype | None:
    """Check if string column has all unique values (identifier)."""
    if series.dtype != pl.Utf8:
        return None

    non_null = series.drop_nulls()
    if len(non_null) == n_rows and non_null.n_unique() == n_rows:
        return ColumnDtype.STRING  # Identified as string with identifier role

    return None
