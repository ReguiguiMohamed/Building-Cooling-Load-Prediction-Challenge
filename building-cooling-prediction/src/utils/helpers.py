import pandas as pd
import yaml
from pathlib import Path
from src.feature_engineering.time_features import create_time_features
from src.feature_engineering.lag_features import create_lag_features
from src.feature_engineering.weather_features import create_weather_features
from src.feature_engineering.technical_features import create_technical_features
from src.data_processing.load_data import load_csv_data, resolve_path
from src.data_processing.external_data import load_weather_data, merge_with_weather

def create_features(
    data_path: str,
    timestamp_col: str,
    cols_to_lag: list[str],
    window_sizes: list[int],
    weather_paths: list[str] | None = None,
    config_path: str | None = None,
) -> pd.DataFrame:
    """
    Master function to create all features.

    Args:
        data_path: Path to the raw data.
        timestamp_col: Name of the timestamp column.
        cols_to_lag: List of columns to create lag features for.
        window_sizes: List of window sizes for rolling features.

    Returns:
        DataFrame with all features.
    """
    # Load data
    df = load_csv_data(data_path)

    # If provided timestamp_col is missing, try common alternatives
    if timestamp_col not in df.columns:
        for candidate in (timestamp_col, 'prediction_time', 'record_timestamp', 'timestamp', 'datetime'):
            if candidate in df.columns:
                timestamp_col = candidate
                break
        else:
            raise KeyError(
                f"Timestamp column '{timestamp_col}' not found in data and no common alternatives detected."
            )

    # Create time features
    df = create_time_features(df, timestamp_col)

    # Load and merge weather data
    if weather_paths is None:
        # Load from config.yaml if not explicitly provided
        cfg_file = resolve_path(config_path or "config.yaml")
        if not Path(cfg_file).exists():
            raise FileNotFoundError(f"Config file not found at {cfg_file}")
        with open(cfg_file, "r") as f:
            cfg = yaml.safe_load(f)
        try:
            weather_paths = [
                cfg["data"]["raw"]["weather_2023"],
                cfg["data"]["raw"]["weather_2024_jan"],
            ]
        except KeyError as e:
            raise KeyError(
                "Missing weather paths in config.yaml under data.raw.weather_2023/weather_2024_jan"
            ) from e

    weather_df = load_weather_data(*weather_paths)
    df = merge_with_weather(df, weather_df, timestamp_col)

    # Create technical features
    df = create_technical_features(df)

    # Create lag features
    df = create_lag_features(df, cols_to_lag, window_sizes)

    # Create weather features
    df = create_weather_features(df)

    return df


def create_test_features(config_path: str | None = None) -> pd.DataFrame:
    """Convenience wrapper to generate features for the test set.

    - Reads paths and settings from config.yaml
    - Forces timestamp column to 'prediction_time' (as in test.csv)
    - Uses config-driven weather paths
    """
    cfg_file = resolve_path(config_path or "config.yaml")
    with open(cfg_file, "r") as f:
        cfg = yaml.safe_load(f)

    return create_features(
        data_path=cfg["data"]["raw"]["test"],
        timestamp_col="prediction_time",
        cols_to_lag=cfg["feature_engineering"]["cols_to_lag"],
        window_sizes=cfg["feature_engineering"]["window_sizes"],
        weather_paths=[
            cfg["data"]["raw"]["weather_2023"],
            cfg["data"]["raw"]["weather_2024_jan"],
        ],
        config_path=str(cfg_file),
    )


def save_test_features(df: pd.DataFrame, config_path: str | None = None) -> str:
    """Save provided test features to the configured path, returning the resolved path."""
    from src.data_processing.load_data import save_csv_data, resolve_path

    cfg_file = resolve_path(config_path or "config.yaml")
    with open(cfg_file, "r") as f:
        cfg = yaml.safe_load(f)
    out = cfg["data"]["processed"]["features_test"]
    save_csv_data(df, out)
    return str(resolve_path(out))
