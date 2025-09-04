# Model Performance Evaluation

Evaluation performed on 1752 test samples.
Metric: Normalized Root Mean Square Error (NRMSE)

## Results

| Model | NRMSE | Rank |
|-------|-------|------|
| Lightgbm | 0.02315 | 1 |
| Xgboost | 0.04026 | 2 |
| Lstm | 11.05651 | 3 |

## Notes

- Lower NRMSE values indicate better performance
- Test set size: 1752 samples
- Train set size: 7008 samples
- Target variable range: [nan, nan]

## Visualization

Generated plots available in `reports/figures/`:
- `xgboost_predictions.png` - Predictions vs Actual
- `xgboost_residuals.png` - Residual Analysis
- `lightgbm_predictions.png` - Predictions vs Actual
- `lightgbm_residuals.png` - Residual Analysis
- `lstm_predictions.png` - Predictions vs Actual
- `lstm_residuals.png` - Residual Analysis