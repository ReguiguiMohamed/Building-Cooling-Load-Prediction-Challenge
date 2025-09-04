import pandas as pd
from src.feature_engineering.time_features import create_time_features
from src.feature_engineering.lag_features import create_lag_features
from src.feature_engineering.weather_features import create_weather_features
from src.feature_engineering.technical_features import create_technical_features
from src.data_processing.load_data import load_csv_data
from src.data_processing.external_data import load_weather_data, merge_with_weather

def create_features(data_path: str, timestamp_col: str, cols_to_lag: list[str], window_sizes: list[int]) -> pd.DataFrame:
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

    # Create time features
    df = create_time_features(df, timestamp_col)

    # Load and merge weather data
    weather_df = load_weather_data(
        "data/raw/external/hk_weather_2023.csv",
        "data/raw/external/hk_weather_2024_jan.csv",
    )
    df = merge_with_weather(df, weather_df, timestamp_col)

    # Create technical features
    df = create_technical_features(df)

    # Create lag features
    df = create_lag_features(df, cols_to_lag, window_sizes)

    # Create weather features
    df = create_weather_features(df)

    return df

