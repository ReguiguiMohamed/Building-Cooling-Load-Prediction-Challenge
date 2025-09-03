"""Evaluation metrics used throughout the project."""

import numpy as np
from sklearn.metrics import mean_squared_error

__all__ = ["rmse", "nrmse"]


def rmse(y_true, y_pred):
    """Root mean squared error.

    Positions where either ``y_true`` or ``y_pred`` are ``NaN`` are ignored to
    match scikit-learn's behaviour on non-finite values.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if not np.any(mask):
        return np.nan
    return np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))


def nrmse(y_true, y_pred):
    """Normalized RMSE.

    The RMSE is normalised by the range (max - min) of ``y_true``. If the
    range is zero the function returns ``NaN`` to avoid division by zero.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.nanptp(y_true)  # equivalent to max - min while ignoring NaNs
    if denom == 0:
        return np.nan
    return rmse(y_true, y_pred) / denom
