"""Post-generation hook loading and execution."""

import importlib.util
import logging
from pathlib import Path

import polars as pl

from proof_of_replica.exceptions import GenerationError

logger = logging.getLogger(__name__)


def load_and_run_hook(hook_path: Path, df: pl.DataFrame) -> pl.DataFrame:
    """Load a Python file and run its post_generate function.

    The hook file must define a function:
        def post_generate(df: pl.DataFrame) -> pl.DataFrame

    Args:
        hook_path: Path to the Python hook file.
        df: Generated DataFrame to transform.

    Returns:
        Transformed DataFrame.

    Raises:
        GenerationError: If the hook file cannot be loaded or executed.
    """
    if not hook_path.exists():
        msg = f"Hook file not found: {hook_path}"
        raise GenerationError(msg)

    try:
        spec = importlib.util.spec_from_file_location("hook_module", hook_path)
        if spec is None or spec.loader is None:
            msg = f"Cannot load hook module from: {hook_path}"
            raise GenerationError(msg)

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except GenerationError:
        raise
    except Exception as exc:
        msg = f"Failed to load hook file {hook_path}: {exc}"
        raise GenerationError(msg) from exc

    if not hasattr(module, "post_generate"):
        msg = f"Hook file {hook_path} has no 'post_generate' function"
        raise GenerationError(msg)

    try:
        result = module.post_generate(df)
    except Exception as exc:
        msg = f"Hook execution failed: {exc}"
        raise GenerationError(msg) from exc

    if not isinstance(result, pl.DataFrame):
        msg = f"Hook must return a DataFrame, got {type(result).__name__}"
        raise GenerationError(msg)

    return result
