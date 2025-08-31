import pandas as pd
from src.utils import constants

def calculate_chiller_cooling_load(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the cooling load for each chiller and the total cooling load.

    Args:
        df: DataFrame with building data.

    Returns:
        DataFrame with calculated cooling loads for each chiller and the total cooling load.
    """
    for i in range(1, 4):  # For chillers 1, 2, and 3
        chiller_id = f"CHR-0{i}"
        flow_rate_col = f"{chiller_id}-CHWFWR"
        supply_temp_col = f"{chiller_id}-CHWSWT"
        return_temp_col = f"{chiller_id}-CHWRWT"
        cooling_load_col = f"{chiller_id}-CL"

        if all(col in df.columns for col in [flow_rate_col, supply_temp_col, return_temp_col]):
            delta_t = df[return_temp_col] - df[supply_temp_col]
            # Cooling Load (kW) = 4.19 * FR * ΔT / 3600
            df[cooling_load_col] = (constants.CP_WATER * df[flow_rate_col] * delta_t) / 3600

    # Calculate Total Cooling Load
    chiller_load_cols = [f"CHR-0{i}-CL" for i in range(1, 4) if f"CHR-0{i}-CL" in df.columns]
    if chiller_load_cols:
        df['Total_Cooling_Load'] = df[chiller_load_cols].sum(axis=1)

    return df