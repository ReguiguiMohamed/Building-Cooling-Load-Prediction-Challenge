"""Simple ensemble models used for combining predictions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import joblib
import numpy as np


@dataclass
class AverageEnsemble:
    """Average the predictions from several fitted models."""

    models: Sequence

    def predict(self, X):
        preds = [model.predict(X) for model in self.models]
        return np.mean(preds, axis=0)


def save_ensemble(model: AverageEnsemble, path: str) -> None:
    """Serialise an ensemble model using :mod:`joblib`."""

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_ensemble(path: str) -> AverageEnsemble:
    """Load a previously saved ensemble model."""

    return joblib.load(path)


def mean_ensemble(models: Iterable, save_path: str | None = None) -> AverageEnsemble:
    """Create an :class:`AverageEnsemble` from ``models``.

    Parameters
    ----------
    models:
        Iterable of already fitted estimators implementing ``predict``.
    save_path:
        Optional path to serialise the resulting ensemble.
    """

    ensemble = AverageEnsemble(list(models))
    if save_path:
        save_ensemble(ensemble, save_path)
    return ensemble

