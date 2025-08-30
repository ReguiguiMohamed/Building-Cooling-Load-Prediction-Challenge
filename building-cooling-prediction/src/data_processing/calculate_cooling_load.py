import pandas as pd
from src.utils import constants

def calculate_chiller_cooling_load(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the cooling load for each chiller.

    Args:
        df: DataFrame with building data, including chiller flow rates and temperature differences.
            Assumes columns like 'chiller_i_flow_rate' and 'chiller_i_return_temp' and 'chiller_i_supply_temp'
            for each chiller i.

    Returns:
        DataFrame with calculated cooling loads for each chiller.
    """
    # This is a placeholder implementation.
    # The actual implementation will depend on the column names in Building_X.csv
    # For now, we assume the following columns exist:
    # - chiller_1_flow_rate, chiller_1_supply_temp, chiller_1_return_temp
    # - chiller_2_flow_rate, chiller_2_supply_temp, chiller_2_return_temp
    # ... and so on.

    for i in range(1, 6): # Assuming 5 chillers
        flow_rate_col = f'chiller_{i}_flow_rate'
        supply_temp_col = f'chiller_{i}_supply_temp'
        return_temp_col = f'chiller_{i}_return_temp'
        cooling_load_col = f'chiller_{i}_cooling_load_kw'

        if all(col in df.columns for col in [flow_rate_col, supply_temp_col, return_temp_col]):
            # Cooling Load (kW) = Flow Rate (m³/h) * Specific Heat (kJ/kg°C) * Density (kg/m³) * Temp Diff (°C) / 3600 (s/h)
            # Assuming density of water is 1000 kg/m³
            # Cooling Load (kW) = Flow Rate (m³/h) * 4.186 * 1000 * (Return Temp - Supply Temp) / 3600
            df[cooling_load_col] = (df[flow_rate_col] * constants.CP_WATER * 1000 *
                                    (df[return_temp_col] - df[supply_temp_col])) / 3600

    return df
