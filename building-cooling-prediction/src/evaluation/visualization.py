"""Visualisation helpers for model evaluation."""

from pathlib import Path

import matplotlib.pyplot as plt


def _prepare_path(save_path: str) -> None:
    """Ensure the directory for ``save_path`` exists."""

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)


def plot_predictions(y_true, y_pred, save_path: str):
    """Line plot comparing actual and predicted values."""

    _prepare_path(save_path)
    plt.figure()
    plt.plot(y_true, label="Actual")
    plt.plot(y_pred, label="Predicted")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_residuals(y_true, y_pred, save_path: str):
    """Plot the residuals (``y_true - y_pred``)."""

    residuals = y_true - y_pred
    _prepare_path(save_path)
    plt.figure()
    plt.plot(residuals)
    plt.title("Residuals")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_nrmse_comparison(results: dict, save_path: str):
    """Bar chart comparing NRMSE values across models.

    Parameters
    ----------
    results:
        Mapping of model name to NRMSE value.
    save_path:
        File location where the plot will be saved.
    """

    _prepare_path(save_path)
    plt.figure()
    models = list(results.keys())
    scores = [results[m] for m in models]
    plt.bar(models, scores)
    plt.ylabel("NRMSE")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_feature_importance(importances, feature_names, save_path: str):
    """Horizontal bar chart of feature importances."""

    _prepare_path(save_path)
    plt.figure()
    order = range(len(feature_names))
    plt.barh(order, importances)
    plt.yticks(order, feature_names)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
