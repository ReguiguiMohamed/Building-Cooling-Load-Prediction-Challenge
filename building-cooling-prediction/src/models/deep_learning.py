import json
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    from tensorflow.keras.callbacks import EarlyStopping
except ImportError:  # pragma: no cover - optional dependency
    Sequential = None

CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / 'models' / 'model_configs'
MODEL_DIR = Path(__file__).resolve().parent.parent.parent / 'models' / 'trained_models'


def save_config(config: Dict[str, Any]):
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    path = CONFIG_DIR / 'lstm_config.json'
    with open(path, 'w') as f:
        json.dump(config, f, indent=4)


def build_lstm_model(input_shape, units: int = 50, output_dim: int = 1):
    if Sequential is None:
        raise ImportError('TensorFlow is not installed')
    model = Sequential([
        LSTM(units, input_shape=input_shape),
        Dense(output_dim)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


def train_lstm(X_train, y_train, config: Optional[Dict[str, Any]] = None):
    if config is None:
        config = {"units": 50, "epochs": 10, "batch_size": 32}
    save_config(config)
    model = build_lstm_model(input_shape=X_train.shape[1:], units=config["units"])
    callbacks = [EarlyStopping(patience=3, restore_best_weights=True)]
    model.fit(
        X_train,
        y_train,
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        callbacks=callbacks,
        verbose=0,
    )
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_DIR / 'lstm_model.h5')
    return model
