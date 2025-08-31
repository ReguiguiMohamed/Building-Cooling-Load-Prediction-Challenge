import pandas as pd

def create_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers domain-specific technical features.

    Args:
        df: DataFrame with building data.

    Returns:
        DataFrame with new technical features.
    """
    for i in range(1, 4):  # For chillers 1, 2, and 3
        chiller_id = f"CHR-0{i}"
        supply_temp_col = f"{chiller_id}-CHWSWT"
        return_temp_col = f"{chiller_id}-CHWRWT"
        delta_t_col = f"{chiller_id}-delta_t"

        if supply_temp_col in df.columns and return_temp_col in df.columns:
            df[delta_t_col] = df[return_temp_col] - df[supply_temp_col]

    return df
