import json
from pathlib import Path
from typing import Any, Dict, Literal

import numpy as np
from sklearn.metrics import roc_curve

MODEL_DIR = Path(__file__).parent.parent / "pretrained"


class ONNXModel:
    """Wrapper for an ONNX model to provide a more familiar interface."""

    def __init__(self, path):
        import onnxruntime as ort

        self.model = ort.InferenceSession(path, providers=["CPUExecutionProvider"])

    def predict(self, x: np.ndarray, apply_norm: bool = True):
        """If apply_norm is specified, then apply a norm before feeding into model."""
        assert x.ndim == 1
        if apply_norm:
            x /= np.linalg.norm(x)
        if x.dtype != np.float32:
            x = x.astype(np.float32)
        return self.model.run(None, {"input": x[None, :]})[0].squeeze()

    def predict_batch(self, x: np.ndarray, apply_norm: bool = True):
        assert x.ndim == 2
        if apply_norm:
            x /= np.linalg.norm(x, axis=1)[:, None]
        if x.dtype != np.float32:
            x = x.astype(np.float32)
        return self.model.run(None, {"input": x})[0]


def load_proteinclip(
    model_arch: Literal["esm", "t5"], model_size: int | None = None
) -> ONNXModel:
    """Load the ProteinCLIP model for the given protein language model."""
    assert MODEL_DIR.is_dir()
    if model_arch == "esm":
        assert model_size is not None, "ESM model requires a size."
        assert model_size in [
            6,
            12,
            30,
            33,
            36,
        ], f"Invalid ESM model size: {model_size}"
        model_path = MODEL_DIR / f"proteinclip_esm2_{model_size}.onnx"
    elif model_arch == "t5":
        assert model_size is None, "T5 model does not have different sizes."
        model_path = MODEL_DIR / "proteinclip_prott5.onnx"
    else:
        raise ValueError(f"Invalid model architecture: {model_arch}")
    assert model_path.exists(), f"Model path does not exist: {model_path}"
    return ONNXModel(model_path)


def load_model_config(model_config_path: Path | str) -> Dict[str, Any]:
    """Load model configuration."""
    with open(model_config_path) as source:
        model_config = json.load(source)
    return model_config


def load_training_config(training_config_path: Path | str) -> Dict[str, Any]:
    """Load training configuration."""
    with open(training_config_path) as source:
        training_config = json.load(source)
    return training_config


def load_training_and_model_config(model_dir: Path | str) -> Dict[str, Any]:
    """Load model and training configuration."""
    model_config = load_model_config(Path(model_dir) / "model_config.json")

    training_config = load_training_config(Path(model_dir) / "training_config.json")
    return model_config, training_config


def youden_threshold(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Find the optimal probability threshold point for a classification model based on Youden's J statistic.

    Parameters
    ----------
    y_true : array, shape = [n_samples]
        True binary labels.
    y_pred : array, shape = [n_samples]
        Target scores, can either be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions.

    Returns
    -------
    optimal_threshold : float
        Threshold value which maximizes Youden's J statistic.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    j_scores = tpr - fpr
    optimal_threshold = thresholds[np.argmax(j_scores)]

    return optimal_threshold
