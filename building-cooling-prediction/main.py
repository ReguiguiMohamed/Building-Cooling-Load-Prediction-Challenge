import os
import yaml
import joblib
import pandas as pd
from src.data_processing.load_data import load_csv_data, save_csv_data
from src.data_processing.calculate_cooling_load import calculate_chiller_cooling_load
from src.data_processing.aggregate_data import aggregate_to_hourly
from src.utils.helpers import create_features
from src.evaluation.validation import simple_train_test_split
from src.evaluation.metrics import nrmse
from src.models.ensemble import mean_ensemble, save_ensemble

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

    # Step 18: Feature engineering for test data
    test_raw = config['data']['raw'].get('test')
    if os.path.exists(test_raw):
        test_features = create_features(
            data_path=test_raw,
            timestamp_col=feature_eng_config['timestamp_col'],
            cols_to_lag=feature_eng_config['cols_to_lag'],
            window_sizes=feature_eng_config['window_sizes']
        )
        save_csv_data(test_features, config['data']['processed']['features_test'])
        print(f"Test features saved to {config['data']['processed']['features_test']}")
    else:
        print("Test data not found; skipping test feature engineering.")

    # Step 19: Train simple model and evaluate
    from sklearn.ensemble import RandomForestRegressor

    train_df = features_df.select_dtypes(include=[float, int]).dropna()
    y = train_df['Total_Cooling_Load'].values
    X = train_df.drop(columns=['Total_Cooling_Load']).values
    X_tr, X_va, y_tr, y_va = simple_train_test_split(X, y, test_size=0.2)

    model = RandomForestRegressor(random_state=42)
    model.fit(X_tr, y_tr)
    preds = model.predict(X_va)
    print(f"Validation NRMSE: {nrmse(y_va, preds):.4f}")

    model_dir = config['models']['trained_models']
    os.makedirs(model_dir, exist_ok=True)
    best_model_path = os.path.join(model_dir, 'best_model.pkl')
    joblib.dump(model, best_model_path)

    # Step 20: Create simple ensemble (single model for placeholder)
    ensemble = mean_ensemble([model])
    ensemble_path = os.path.join(model_dir, 'ensemble_model.pkl')
    save_ensemble(ensemble, ensemble_path)

    # Step 21: Generate predictions on test data if available
    if 'test_features' in locals():
        test_X = test_features.select_dtypes(include=[float, int]).drop(columns=['Total_Cooling_Load'], errors='ignore')
        test_preds = model.predict(test_X)
        submissions_dir = config['data']['submissions']
        os.makedirs(submissions_dir, exist_ok=True)
        save_csv_data(pd.DataFrame({'prediction': test_preds}), os.path.join(submissions_dir, 'submission_final.csv'))
        ens_preds = ensemble.predict(test_X)
        save_csv_data(pd.DataFrame({'prediction': ens_preds}), os.path.join(submissions_dir, 'submission_ensemble.csv'))
        print(f"Predictions saved to {submissions_dir}")


if __name__ == "__main__":
    main()
