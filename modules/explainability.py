"""
Module: Core Explainability
SHAP explainer initialization, computation, and caching.
"""

import joblib
import logging

import numpy as np
import pandas as pd
import shap

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def detect_model_type(model):
    """Detect explainer family from model class."""
    cls = model.__class__.__name__.lower()
    if "logisticregression" in cls or "linear" in cls:
        return "linear"
    if any(x in cls for x in ["xgb", "lgbm", "forest", "tree", "gbm", "boost"]):
        return "tree"
    # Conservative fallback for non-linear models.
    return "tree"


def initialize_shap_explainer(model, X_background, model_type="tree"):
    """Initialize SHAP explainer with background dataset."""
    logging.info(f"Initializing SHAP {model_type} explainer...")

    if model_type == "tree":
        # Keep explanations in native model output ("raw") space.
        explainer = shap.TreeExplainer(model, X_background, model_output="raw")
    elif model_type == "linear":
        explainer = shap.LinearExplainer(model, X_background)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    logging.info("SHAP explainer initialized successfully.")
    return explainer


def compute_shap_values(explainer, X_data, class_index=1):
    """
    Compute SHAP values for dataset or single instance.

    Handles multiple SHAP return formats across versions/explainers:
    - shap.Explanation
    - ndarray [n_samples, n_features]
    - ndarray [n_samples, n_features, n_classes]
    - list[class] of ndarray
    """
    logging.info(f"Computing SHAP values for {len(X_data)} samples...")

    X_array = X_data.values if hasattr(X_data, "values") else X_data
    raw = explainer.shap_values(X_array)
    values = raw.values if isinstance(raw, shap.Explanation) else raw

    if isinstance(values, list):
        if len(values) <= class_index:
            raise ValueError(f"SHAP class index {class_index} out of range.")
        shap_values = values[class_index]
    elif isinstance(values, np.ndarray) and values.ndim == 3:
        if values.shape[-1] <= class_index:
            raise ValueError(f"SHAP class axis missing class index {class_index}.")
        shap_values = values[:, :, class_index]
    else:
        shap_values = values

    if not isinstance(shap_values, np.ndarray) or shap_values.ndim != 2:
        shape = getattr(shap_values, "shape", type(shap_values))
        raise ValueError(f"Unexpected SHAP output shape: {shape}")

    logging.info("SHAP values computed successfully.")
    return shap_values.astype(np.float32, copy=False)


def extract_base_value(explainer, class_index=1):
    """Extract scalar expected value for selected class."""
    base = explainer.expected_value
    if isinstance(base, (list, tuple)):
        if len(base) <= class_index:
            raise ValueError(f"Expected value class index {class_index} out of range.")
        return float(base[class_index])
    if isinstance(base, np.ndarray):
        if base.ndim == 0:
            return float(base.item())
        flat = base.reshape(-1)
        if len(flat) == 1:
            return float(flat[0])
        if len(flat) <= class_index:
            raise ValueError(f"Expected value class index {class_index} out of range.")
        return float(flat[class_index])
    return float(base)


def save_explainer(explainer, filepath):
    """Save SHAP explainer to disk for reuse."""
    joblib.dump(explainer, filepath)
    logging.info(f"SHAP explainer saved to {filepath}")


def load_explainer(filepath):
    """Load cached SHAP explainer from disk."""
    explainer = joblib.load(filepath)
    logging.info(f"SHAP explainer loaded from {filepath}")
    return explainer


def validate_features(X_data, expected_features):
    """Validate that input data contains expected features in exact order."""
    if isinstance(X_data, pd.DataFrame):
        actual_features = X_data.columns.tolist()
        if actual_features != expected_features:
            missing = set(expected_features) - set(actual_features)
            extra = set(actual_features) - set(expected_features)
            error_msg = []
            if missing:
                error_msg.append(f"Missing features: {missing}")
            if extra:
                error_msg.append(f"Extra features: {extra}")
            raise ValueError(" | ".join(error_msg))
    logging.info("Feature validation passed.")


def verify_efficiency_axiom(shap_values, predictions, base_value, tolerance=1e-3):
    """
    Verify SHAP efficiency axiom: sum(SHAP values) ~= prediction - base_value.

    Inputs must be in the same output space (e.g., both log-odds or both probability).
    """
    shap_sum = shap_values.sum(axis=1).astype(np.float64)
    expected = predictions - base_value
    diff = np.abs(shap_sum - expected)

    if np.all(diff < tolerance):
        logging.info(f"[OK] Efficiency axiom verified (max diff: {diff.max():.6f})")
        return True

    logging.warning(f"[WARN] Efficiency axiom violated (max diff: {diff.max():.6f})")
    return False
