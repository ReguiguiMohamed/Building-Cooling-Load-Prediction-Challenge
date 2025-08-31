import numpy as np
from sklearn.metrics import mean_squared_error


def rmse(y_true, y_pred):
    """Root mean squared error."""
    return mean_squared_error(y_true, y_pred, squared=False)


def nrmse(y_true, y_pred):
    """Normalized root mean squared error."""
    denominator = np.max(y_true) - np.min(y_true)
    if denominator == 0:
        return np.nan
    return rmse(y_true, y_pred) / denominator
