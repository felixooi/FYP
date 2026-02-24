"""
Inference-safe end-to-end pipeline.
Loads trained artifacts and predicts attrition risk for new datasets.
"""

import json
import os
from typing import Tuple

import joblib
import numpy as np
import pandas as pd

from modules.data_ingestion import load_data
from modules.data_cleaning import clean_data
from modules.feature_engineering import (
    apply_workload_features,
    encode_categorical_features,
    align_encoded_columns,
)


def _load_artifacts(
    model_path: str = "models/best_model_tuned.pkl",
    metadata_path: str = "models/tuning_metadata.json",
    selected_features_path: str = "data/selected_features.json",
    scaler_path: str = "data/scaler.pkl",
    fe_params_path: str = "models/feature_engineering_params.json",
) -> Tuple[object, float, list, object, dict]:
    """Load model and preprocessing artifacts for inference."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing model artifact: {model_path}")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Missing metadata artifact: {metadata_path}")
    if not os.path.exists(selected_features_path):
        raise FileNotFoundError(f"Missing selected features artifact: {selected_features_path}")

    model = joblib.load(model_path)
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    with open(selected_features_path, "r") as f:
        selected_features = json.load(f)

    threshold = float(metadata.get("threshold", 0.5))
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    fe_params = None
    if os.path.exists(fe_params_path):
        with open(fe_params_path, "r") as f:
            fe_params = json.load(f)

    return model, threshold, selected_features, scaler, fe_params


def _is_preprocessed(df: pd.DataFrame, selected_features: list) -> bool:
    """Check if input already matches model feature schema."""
    return set(selected_features).issubset(df.columns)


def _preprocess_for_inference(
    df_raw: pd.DataFrame,
    selected_features: list,
    scaler=None,
    fe_params=None,
) -> Tuple[pd.DataFrame, str]:
    """
    Convert raw dataset to model-ready feature matrix.
    Supports raw HR-style input and already-preprocessed input.
    """
    if _is_preprocessed(df_raw, selected_features):
        X = df_raw.copy()
        fe_param_source = "not_applicable_preprocessed_input"
    else:
        cleaned = clean_data(df_raw.copy())
        if isinstance(cleaned, tuple):
            df_clean, _ = cleaned
        else:
            df_clean = cleaned

        if fe_params is not None:
            df_eng = apply_workload_features(df_clean, fe_params)
            fe_param_source = "saved_training_params"
        else:
            # Fallback: computes params from incoming batch if training params are unavailable.
            df_eng = apply_workload_features(df_clean, params=None)
            fe_param_source = "fallback_from_incoming_data"

        df_enc = encode_categorical_features(df_eng)
        template = pd.DataFrame(columns=selected_features + ["Resigned"])
        if "Resigned" not in df_enc.columns:
            df_enc["Resigned"] = 0
        df_enc = align_encoded_columns(template, df_enc, target_col="Resigned")
        X = df_enc.drop(columns=["Resigned"])

    for col in selected_features:
        if col not in X.columns:
            X[col] = 0
    X = X[selected_features].copy()

    if scaler is not None:
        X[selected_features] = scaler.transform(X[selected_features])

    return X, fe_param_source


def _predict_with_model(model, X: pd.DataFrame) -> np.ndarray:
    """Run predict_proba across sklearn-like and TabNet-like interfaces."""
    try:
        return model.predict_proba(X)[:, 1]
    except Exception:
        return model.predict_proba(X.values)[:, 1]


def run_inference_pipeline(
    input_file: str,
    output_dir: str = "pipeline_output",
    model_path: str = "models/best_model_tuned.pkl",
    metadata_path: str = "models/tuning_metadata.json",
    selected_features_path: str = "data/selected_features.json",
    scaler_path: str = "data/scaler.pkl",
    fe_params_path: str = "models/feature_engineering_params.json",
) -> dict:
    """
    Run inference pipeline for a new dataset and save prediction outputs.
    """
    os.makedirs(output_dir, exist_ok=True)

    model, threshold, selected_features, scaler, fe_params = _load_artifacts(
        model_path=model_path,
        metadata_path=metadata_path,
        selected_features_path=selected_features_path,
        scaler_path=scaler_path,
        fe_params_path=fe_params_path,
    )

    df_input = load_data(filepath=input_file)
    X, fe_param_source = _preprocess_for_inference(
        df_raw=df_input,
        selected_features=selected_features,
        scaler=scaler,
        fe_params=fe_params,
    )

    y_prob = _predict_with_model(model, X)
    y_pred = (y_prob >= threshold).astype(int)
    risk_level = np.where(y_prob < 0.30, "LOW", np.where(y_prob < 0.70, "MEDIUM", "HIGH"))

    output = df_input.copy()
    output["Attrition_Probability"] = y_prob
    output["Predicted_Resigned"] = y_pred
    output["Risk_Level"] = risk_level

    predictions_path = os.path.join(output_dir, "inference_predictions.csv")
    summary_path = os.path.join(output_dir, "inference_summary.json")
    output.to_csv(predictions_path, index=False)

    summary = {
        "input_rows": int(len(df_input)),
        "model_path": model_path,
        "threshold": float(threshold),
        "predicted_resigned_count": int((y_pred == 1).sum()),
        "predicted_stayed_count": int((y_pred == 0).sum()),
        "mean_attrition_probability": float(np.mean(y_prob)),
        "used_feature_count": int(len(selected_features)),
        "feature_engineering_param_source": fe_param_source,
        "predictions_path": predictions_path,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    print("Inference complete")
    print(f"Saved: {predictions_path}")
    print(f"Saved: {summary_path}")

    return summary


def main():
    # Example usage
    run_inference_pipeline(
        input_file="data/test_data.csv",
        output_dir="pipeline_output",
    )


if __name__ == "__main__":
    main()
