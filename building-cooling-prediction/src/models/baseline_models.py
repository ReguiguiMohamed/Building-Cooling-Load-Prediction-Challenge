import numpy as np
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression


@dataclass
class MeanBaseline:
    """Predicts the mean of the training targets."""
    mean_value_: float = 0.0

    def fit(self, y):
        self.mean_value_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(shape=len(X), fill_value=self.mean_value_)


@dataclass
class LastValueBaseline:
    """Predicts using the last observed target value."""
    last_value_: float = 0.0

    def fit(self, y):
        self.last_value_ = float(y[-1])
        return self

    def predict(self, X):
        return np.full(shape=len(X), fill_value=self.last_value_)


class LinearRegressionBaseline:
    """Wrapper around scikit-learn's LinearRegression."""

    def __init__(self):
        self.model = LinearRegression()

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)
