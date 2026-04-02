"""StrEnum definitions for the profile JSON schema."""

from enum import StrEnum


class ColumnDtype(StrEnum):
    """Data types supported for profile columns."""

    STRING = "string"
    FLOAT64 = "float64"
    INT64 = "int64"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    DATE = "date"


class ColumnRole(StrEnum):
    """Semantic roles a column can play in the dataset."""

    IDENTIFIER = "identifier"
    FEATURE = "feature"
    GROUP = "group"
    OUTCOME = "outcome"
    INDEX = "index"


class DistributionFamily(StrEnum):
    """Parametric distribution families for column generation."""

    NORMAL = "normal"
    LOGNORMAL = "lognormal"
    UNIFORM = "uniform"
    BETA = "beta"
    GAMMA = "gamma"
    EXPONENTIAL = "exponential"
    EMPIRICAL_KDE = "empirical_kde"
    EMPIRICAL_HISTOGRAM = "empirical_histogram"


class GeneratorMethod(StrEnum):
    """Methods for generating column values."""

    SEQUENTIAL = "sequential"
    UUID = "uuid"
    PREFIX_INCREMENT = "prefix_increment"
    REGEX = "regex"
    ALPHABET = "alphabet"
    WEIGHTED_CHOICE = "weighted_choice"
    PLACEHOLDER = "placeholder"
    LOREM = "lorem"


class CorrelationMethod(StrEnum):
    """Correlation computation methods."""

    PEARSON = "pearson"
    SPEARMAN = "spearman"
    KENDALL = "kendall"


class MissingnessPattern(StrEnum):
    """Patterns describing the mechanism of missing data."""

    MCAR = "MCAR"
    MAR = "MAR"
    MNAR = "MNAR"
    OBSERVED = "observed"
