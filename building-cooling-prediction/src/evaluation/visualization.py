from pathlib import Path
import matplotlib.pyplot as plt


def plot_predictions(y_true, y_pred, save_path: str):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(y_true, label='Actual')
    plt.plot(y_pred, label='Predicted')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_residuals(y_true, y_pred, save_path: str):
    residuals = y_true - y_pred
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(residuals)
    plt.title('Residuals')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
