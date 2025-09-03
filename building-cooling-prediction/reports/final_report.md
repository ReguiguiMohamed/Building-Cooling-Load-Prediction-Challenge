  
# Final Report

This project implements an end-to-end pipeline for predicting building
cooling load. Data ingestion, feature engineering, model training,
evaluation and submission file generation are automated through
``main.py``. Key components include:

- **Metrics** – RMSE and NRMSE with NaN-safe handling.
- **Validation** – helper utilities for standard and time-series splits.
- **Visualisations** – prediction, residual and metric comparison plots.
- **Ensemble** – simple averaging ensemble for combining model outputs.

Refer to the notebooks in ``notebooks/`` for example usage and the
``README.md`` for instructions on running the full pipeline.