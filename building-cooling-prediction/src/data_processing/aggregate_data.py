import pandas as pd

def aggregate_to_hourly(df: pd.DataFrame, timestamp_col: str = 'timestamp') -> pd.DataFrame:
    """
    Aggregates 15-minute data to hourly data.

    Args:
        df: DataFrame with 15-minute data.
        timestamp_col: The name of the timestamp column.

    Returns:
        DataFrame with hourly aggregated data.
    """
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df.set_index(timestamp_col, inplace=True)
    hourly_df = df.resample('H').mean()
    return hourly_df.reset_index()
