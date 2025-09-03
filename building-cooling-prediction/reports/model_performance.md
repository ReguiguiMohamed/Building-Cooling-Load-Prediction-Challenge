# Model Performance

This report details the performance of various models on the cooling load prediction task.

| Model               | NRMSE (Cross-Validation) | Notes                                  |
|---------------------|--------------------------|----------------------------------------|
| **Baseline**        |                          | Mean, Last Value, and Linear Regression. |
| **XGBoost**         |                          | Tuned hyperparameters.                 |
| **LightGBM**        |                          | Tuned hyperparameters.                 |
| **LSTM**            |                          | Deep learning approach.                |
| **Ensemble**        |                          | Combination of the best models.        |

*Figures are saved in `reports/figures/` when generated.*