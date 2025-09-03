import numpy as np
from sklearn.metrics import mean_squared_error


def rmse(y_true, y_pred):
    """Root mean squared error.

    Ignores positions where either `y_true` or `y_pred` is NaN to avoid
    scikit-learn raising a ValueError on NaN inputs.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if not np.any(mask):
        return np.nan
    return np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))


def nrmse(y_true, y_pred):
    """Normalized root mean squared error."""
    denominator = np.nanmax(y_true) - np.nanmin(y_true)
    if denominator == 0:
        return np.nan
    return rmse(y_true, y_pred) / denominator
