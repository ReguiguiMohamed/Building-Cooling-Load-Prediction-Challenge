import numpy as np
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression


@dataclass
class MeanBaseline:
    """Predicts the mean of the training targets."""
    mean_value_: float = 0.0

    def fit(self, y):
        self.mean_value_ = float(np.nanmean(y))
        return self

    def predict(self, X):
        return np.full(shape=len(X), fill_value=self.mean_value_)


@dataclass
class LastValueBaseline:
    """Predicts using the last observed target value."""
    last_value_: float = 0.0

    def fit(self, y):
        self.last_value_ = float(np.nan_to_num(y[-1]))
        return self

    def predict(self, X):
        return np.full(shape=len(X), fill_value=self.last_value_)


class LinearRegressionBaseline:
    """Wrapper around scikit-learn's LinearRegression."""

    def __init__(self):
        self.model = LinearRegression()

    def fit(self, X, y):
        # Ensure numpy arrays
        X_arr = np.asarray(X)
        y_arr = np.asarray(y).ravel()

        # Handle 1D feature input
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        # Build mask of valid rows: drop NaNs in y and, if numeric, in X
        valid_mask = ~np.isnan(y_arr)
        # If X is numeric, also drop rows with NaNs in X to avoid downstream errors
        if np.issubdtype(X_arr.dtype, np.number):
            valid_mask &= ~np.any(np.isnan(X_arr), axis=1)

        if not np.any(valid_mask):
            raise ValueError("No valid samples after dropping NaNs in X or y.")

        self.model.fit(X_arr[valid_mask], y_arr[valid_mask])
        return self

    def predict(self, X):
        X_arr = np.asarray(X)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        return self.model.predict(X_arr)
