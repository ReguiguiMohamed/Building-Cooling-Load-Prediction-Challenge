import yaml
from src.data_processing.load_data import load_csv_data, save_csv_data
from src.data_processing.calculate_cooling_load import calculate_chiller_cooling_load
from src.data_processing.aggregate_data import aggregate_to_hourly

def main():
    """
    Main function to run the building cooling load prediction pipeline.
    """
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    print(f"Starting project: {config['project_name']}")

    # Step 2: Data Loading
    try:
        building_df = load_csv_data(config['data']['raw']['building_x'])
        print("Building_X.csv loaded successfully.")
    except FileNotFoundError:
        print("Building_X.csv not found. Please add it to the data/raw directory.")
        return

    # Step 3: Calculate Chiller Cooling Loads
    chiller_loads_df = calculate_chiller_cooling_load(building_df)
    save_csv_data(chiller_loads_df, config['data']['processed']['chiller_loads'])
    print(f"Chiller cooling loads calculated and saved to {config['data']['processed']['chiller_loads']}")

    # Step 4: Data Aggregation
    hourly_df = aggregate_to_hourly(chiller_loads_df)
    save_csv_data(hourly_df, config['data']['processed']['hourly_training_data'])
    print(f"Data aggregated to hourly and saved to {config['data']['processed']['hourly_training_data']}")


if __name__ == "__main__":
    main()