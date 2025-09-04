import pandas as pd

def create_time_features(df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
    """
    Creates time-based features from a timestamp column.

    Args:
        df: DataFrame with a timestamp column.
        timestamp_col: The name of the timestamp column.

    Returns:
        DataFrame with new time-based features.
    """
    # Pick a valid timestamp column if the provided one is missing
    if timestamp_col not in df.columns:
        for candidate in (timestamp_col, 'prediction_time', 'record_timestamp', 'timestamp', 'datetime'):
            if candidate in df.columns:
                timestamp_col = candidate
                break
        else:
            raise KeyError(f"Timestamp column not found. Tried: {timestamp_col}, prediction_time, record_timestamp, timestamp, datetime")

    df[timestamp_col] = pd.to_datetime(df[timestamp_col], dayfirst=True)
    df['hour'] = df[timestamp_col].dt.hour
    df['dayofweek'] = df[timestamp_col].dt.dayofweek
    df['dayofyear'] = df[timestamp_col].dt.dayofyear
    df['month'] = df[timestamp_col].dt.month
    df['year'] = df[timestamp_col].dt.year
    df['weekofyear'] = df[timestamp_col].dt.isocalendar().week.astype(int)

    return df
