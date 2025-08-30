import pandas as pd

def load_csv_data(file_path: str) -> pd.DataFrame:
    """
    Loads data from a CSV file.

    Args:
        file_path: The path to the CSV file.

    Returns:
        A pandas DataFrame containing the data from the CSV file.
    """
    return pd.read_csv(file_path)

def save_csv_data(df: pd.DataFrame, file_path: str):
    """
    Saves a DataFrame to a CSV file.

    Args:
        df: The DataFrame to save.
        file_path: The path to save the CSV file to.
    """
    df.to_csv(file_path, index=False)
