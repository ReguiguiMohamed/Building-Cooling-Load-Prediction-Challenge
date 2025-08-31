import pandas as pd

def create_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates new features by interacting weather data with time features.

    Args:
        df: DataFrame with weather and time features.

    Returns:
        DataFrame with new interaction features.
    """
    if 'temperature_celsius' in df.columns and 'hour' in df.columns:
        df['temp_x_hour'] = df['temperature_celsius'] * df['hour']

    if 'humidity_percent' in df.columns and 'hour' in df.columns:
        df['humidity_x_hour'] = df['humidity_percent'] * df['hour']

    return df
