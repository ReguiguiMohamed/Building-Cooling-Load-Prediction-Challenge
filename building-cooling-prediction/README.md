  
# Building Cooling Load Prediction

End-to-end pipeline for predicting building cooling load from historical
sensor readings and weather data. The project is organised as a set of
scripts and utilities under ``src/`` with accompanying Jupyter notebooks
showing exploratory analysis and modelling steps.

## Usage

1. Ensure the required dependencies are installed (see
   ``requirements.txt`` or ``environment.yml``).
2. Place the raw data files in the paths described in ``config.yaml``.
3. Run the pipeline:

   ```bash
   python main.py
   ```

   This performs feature engineering, trains models, evaluates using
   NRMSE and writes submission files to ``data/submissions/``.

## Repository Structure

- ``src/`` – source code for data processing, feature engineering and
  models.
- ``notebooks/`` – notebooks documenting each phase of the project.
- ``reports/`` – markdown reports and generated figures.

See ``reports/final_report.md`` for a high-level project summary.