# Building Cooling Load Prediction

This project provides an end-to-end pipeline for predicting building cooling load using historical sensor readings and weather data. The goal is to create a model that can accurately forecast cooling load, which can help in optimizing energy consumption.

The project is organized as a set of scripts and utilities under `src/` with accompanying Jupyter notebooks showing exploratory analysis and modeling steps.

## Project Pipeline

The project is divided into the following phases:

1.  **Data Ingestion and Processing**: Loading raw data and processing it into a usable format.
2.  **Feature Engineering**: Creating new features from the existing data to improve model performance. This includes time-based features, lag features, and weather-based features.
3.  **Model Training and Evaluation**: Training several models (LightGBM, XGBoost, LSTM) and evaluating their performance using NRMSE.
4.  **Ensemble Modeling**: Combining the predictions of the best models to create a more robust ensemble model.
5.  **Final Predictions and Submission**: Generating predictions on the test set and creating a submission file.

## Model Performance

The models were evaluated on a test set of 1752 samples. The performance of the models is summarized below:

| Model    | NRMSE    | Rank |
|----------|----------|------|
| LightGBM | 0.02315  | 1    |
| XGBoost  | 0.04026  | 2    |
| LSTM     | 11.05651 | 3    |

The LightGBM model demonstrated the best performance with the lowest NRMSE. The LSTM model performed poorly, indicating potential issues with its architecture or training process. An ensemble model combining the strengths of LightGBM and XGBoost was also developed.

## Usage

1.  Ensure the required dependencies are installed (see `requirements.txt` or `environment.yml`).
2.  Place the raw data files in the paths described in `config.yaml`.
3.  Run the pipeline:

    ```bash
    python main.py
    ```

    This performs feature engineering, trains models, evaluates them using NRMSE, and writes submission files to `data/submissions/`.

## Repository Structure

-   `src/` – source code for data processing, feature engineering and models.
-   `notebooks/` – notebooks documenting each phase of the project.
-   `reports/` – markdown reports and generated figures.
-   `data/` – raw, processed, and submission data.
-   `models/` – trained models and model configurations.

See `reports/final_report.md` for a high-level project summary.