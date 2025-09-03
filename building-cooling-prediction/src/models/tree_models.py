import json
import joblib
from pathlib import Path
from typing import Any, Dict

try:
    import xgboost as xgb
except ImportError:  # pragma: no cover - optional dependency
    xgb = None

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover - optional dependency
    lgb = None

CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / 'models' / 'model_configs'
MODEL_DIR = Path(__file__).resolve().parent.parent.parent / 'models' / 'trained_models'


def load_config(name: str) -> Dict[str, Any]:
    path = CONFIG_DIR / f"{name}_config.json"
    with open(path) as f:
        return json.load(f)


def save_config(name: str, config: Dict[str, Any]):
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    path = CONFIG_DIR / f"{name}_config.json"
    with open(path, 'w') as f:
        json.dump(config, f, indent=4)


def train_xgboost(X_train, y_train, **override_params):
    if xgb is None:
        raise ImportError('xgboost is not installed')
    config = load_config('xgboost')
    config.update(override_params)
    model = xgb.XGBRegressor(**config)
    model.fit(X_train, y_train)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_DIR / 'xgboost_model.pkl')
    return model


def train_lightgbm(X_train, y_train, **override_params):
    if lgb is None:
        raise ImportError('lightgbm is not installed')
    config = load_config('lightgbm')
    config.update(override_params)
    model = lgb.LGBMRegressor(**config)
    model.fit(X_train, y_train)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_DIR / 'lightgbm_model.pkl')
    return model
