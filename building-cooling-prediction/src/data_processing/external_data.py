import pandas as pd
from pathlib import Path
from .load_data import load_csv_data, resolve_path


def load_weather_data(*paths: str) -> pd.DataFrame:
    """Load and concatenate weather data files.

    Parameters
    ----------
    paths:
        One or more file paths pointing to weather CSV files. Each file
        must contain at least a ``date`` column.

    Returns
    -------
    pd.DataFrame
        Concatenated weather data sorted by date.
    """
    dataframes = []
    for p in paths:
        path = resolve_path(str(p))
        if not path.exists():
            raise FileNotFoundError(f"Weather file not found: {path}")
        dataframes.append(load_csv_data(str(path)))

    if not dataframes:
        raise ValueError("No weather data provided")

    weather_df = pd.concat(dataframes, ignore_index=True)
    weather_df['date'] = pd.to_datetime(weather_df['date'])
    weather_df = weather_df.sort_values('date')
    return weather_df


def merge_with_weather(
    df: pd.DataFrame,
    weather_df: pd.DataFrame,
    timestamp_col: str,
) -> pd.DataFrame:
    """Merge building data with external weather data on the date field.

    Parameters
    ----------
    df:
        Building data containing ``timestamp_col``.
    weather_df:
        DataFrame returned by :func:`load_weather_data`.
    timestamp_col:
        Name of the timestamp column in ``df``.

    Returns
    -------
    pd.DataFrame
        ``df`` with weather columns joined on matching dates.
    """
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    merged = df.merge(
        weather_df,
        left_on=df[timestamp_col].dt.date,
        right_on=weather_df['date'].dt.date,
        how='left',
    )
    return merged.drop(columns=['date'])

