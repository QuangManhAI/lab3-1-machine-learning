"""General helpers for the K-Means image segmentation exercise."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def setup_seed(seed: int = 42) -> None:
    """Set deterministic seeds for Python and NumPy."""
    random.seed(seed)
    np.random.seed(seed)


def save_table(df: pd.DataFrame, path: str | Path) -> Path:
    """Save a DataFrame as CSV, creating the parent directory first."""
    output_path = Path(path)
    ensure_dir(output_path.parent)
    df.to_csv(output_path, index=False)
    return output_path


def save_figure(fig, path: str | Path, dpi: int = 160) -> Path:
    """Save a matplotlib figure with tight layout."""
    output_path = Path(path)
    ensure_dir(output_path.parent)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    return output_path


def report_status(message: str) -> None:
    """Print a simple status line for notebook readability."""
    print(f"[lab3_1] {message}")
