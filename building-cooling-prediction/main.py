import yaml
import os
from src.data_processing.load_data import load_csv_data, save_csv_data
from src.data_processing.calculate_cooling_load import calculate_chiller_cooling_load
from src.data_processing.aggregate_data import aggregate_to_hourly
from src.utils.helpers import create_features

def main():
    """
    Main function to run the building cooling load prediction pipeline.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config.yaml")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    print(f"Starting project: {config['project_name']}")

    # Construct absolute paths from config
    for key, path in config['data']['raw'].items():
        config['data']['raw'][key] = os.path.join(script_dir, path)
    for key, path in config['data']['processed'].items():
        config['data']['processed'][key] = os.path.join(script_dir, path)

    # Step 2: Data Loading
    try:
        building_df = load_csv_data(config['data']['raw']['building_x'])
        print("Building_X.csv loaded successfully.")
    except FileNotFoundError:
        print(f"Building_X.csv not found at {config['data']['raw']['building_x']}. Please add it to the data/raw directory.")
        return

    # Step 3: Calculate Chiller Cooling Loads
    chiller_loads_df = calculate_chiller_cooling_load(building_df)
    save_csv_data(chiller_loads_df, config['data']['processed']['chiller_loads'])
    print(f"Chiller cooling loads calculated and saved to {config['data']['processed']['chiller_loads']}")

    # Step 4: Data Aggregation
    hourly_df = aggregate_to_hourly(chiller_loads_df, timestamp_col='record_timestamp')
    save_csv_data(hourly_df, config['data']['processed']['hourly_training_data'])
    print(f"Data aggregated to hourly and saved to {config['data']['processed']['hourly_training_data']}")

    # Step 5: Feature Engineering
    print("Starting feature engineering...")
    feature_eng_config = config['feature_engineering']
    features_df = create_features(
        data_path=config['data']['processed']['hourly_training_data'],
        timestamp_col=feature_eng_config['timestamp_col'],
        cols_to_lag=feature_eng_config['cols_to_lag'],
        window_sizes=feature_eng_config['window_sizes']
    )
    save_csv_data(features_df, config['data']['processed']['features_train'])
    print(f"Features created and saved to {config['data']['processed']['features_train']}")


if __name__ == "__main__":
    main()