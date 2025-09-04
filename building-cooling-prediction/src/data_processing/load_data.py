import pandas as pd
from pathlib import Path


def _project_root() -> Path:
    """Return the repository root (two levels up from this file)."""
    return Path(__file__).resolve().parents[2]


def resolve_path(file_path: str) -> Path:
    """Resolve a path relative to the repo root when not absolute.

    This makes notebook execution location-independent (e.g., running from
    ``notebooks/`` still finds files under ``data/`` at the project root).
    """
    p = Path(file_path)
    if p.is_absolute():
        return p
    return _project_root() / p

def load_csv_data(file_path: str) -> pd.DataFrame:
    """
    Loads data from a CSV file.

    Args:
        file_path: The path to the CSV file.

    Returns:
        A pandas DataFrame containing the data from the CSV file.
    """
    path = resolve_path(file_path)
    return pd.read_csv(path)

def save_csv_data(df: pd.DataFrame, file_path: str):
    """
    Saves a DataFrame to a CSV file.

    Args:
        df: The DataFrame to save.
        file_path: The path to save the CSV file to.
    """
    path = resolve_path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
