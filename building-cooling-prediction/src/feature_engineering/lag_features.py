import pandas as pd

def create_lag_features(df: pd.DataFrame, cols_to_lag: list[str], window_sizes: list[int]) -> pd.DataFrame:
    """
    Creates lag and rolling window features for specified columns.

    Args:
        df: DataFrame with time-series data.
        cols_to_lag: List of column names to create lag features for.
        window_sizes: List of window sizes for rolling features.

    Returns:
        DataFrame with new lag and rolling window features.
    """
    for col in cols_to_lag:
        if col not in df.columns:
            # Skip missing columns to be robust across train/test
            continue
        for window in window_sizes:
            # Lag features
            df[f'{col}_lag_{window}'] = df[col].shift(window)

            # Rolling mean features
            df[f'{col}_rolling_mean_{window}'] = df[col].shift(1).rolling(window=window).mean()

    return df
