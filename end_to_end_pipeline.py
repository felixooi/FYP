"""
Inference-safe end-to-end pipeline.
Loads trained artifacts and predicts attrition risk for new datasets.
"""

import json
import os
import hashlib
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
from modules.explainability import (
    detect_model_type,
    initialize_shap_explainer,
    compute_shap_values,
    extract_base_value,
    load_explainer,
    save_explainer,
)
from modules.explanation_analysis import (
    compute_global_feature_importance,
    extract_local_explanation,
    create_contribution_table,
)
from modules.explanation_generator import generate_explanation


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


def _risk_level(probability: np.ndarray, threshold: float) -> np.ndarray:
    """
    Risk bands aligned with model decision threshold:
    - LOW: below threshold
    - MEDIUM: [threshold, threshold + 0.2)
    - HIGH: >= threshold + 0.2
    """
    return np.where(
        probability < threshold,
        "LOW",
        np.where(probability < (threshold + 0.2), "MEDIUM", "HIGH"),
    )


def _hash_file(path: str) -> str:
    """Compute MD5 hash for artifact cache validation."""
    md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            md5.update(chunk)
    return md5.hexdigest()


def _hash_dataframe(df: pd.DataFrame) -> str:
    """Compute stable hash for dataframe content."""
    return hashlib.md5(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()


def _load_or_build_explainer(
    model,
    model_path: str,
    selected_features: list,
    explainer_cache_dir: str = "outputs/explainability/explainers",
    train_data_path: str = "data/train_data.csv",
):
    """Load cached SHAP explainer if valid, otherwise build from training background."""
    os.makedirs(explainer_cache_dir, exist_ok=True)
    explainer_path = os.path.join(explainer_cache_dir, "shap_explainer.pkl")
    meta_path = os.path.join(explainer_cache_dir, "shap_explainer_meta.json")

    train_df = pd.read_csv(train_data_path)
    if "Resigned" in train_df.columns:
        train_df = train_df.drop(columns=["Resigned"])
    X_background = train_df[selected_features].sample(n=min(100, len(train_df)), random_state=42)

    model_type = detect_model_type(model)
    expected_meta = {
        "model_hash": _hash_file(model_path),
        "model_type": model_type,
        "feature_names": selected_features,
        "background_hash": _hash_dataframe(X_background),
    }

    if os.path.exists(explainer_path) and os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            cached_meta = json.load(f)
        if all(cached_meta.get(k) == v for k, v in expected_meta.items()):
            return load_explainer(explainer_path)

    explainer = initialize_shap_explainer(model, X_background, model_type=model_type)
    save_explainer(explainer, explainer_path)
    with open(meta_path, "w") as f:
        json.dump(expected_meta, f, indent=2)
    return explainer


def _generate_xai_outputs(
    model,
    model_path: str,
    X: pd.DataFrame,
    y_prob: np.ndarray,
    threshold: float,
    selected_features: list,
    output_dir: str,
    raw_df: pd.DataFrame,
    local_count: int = 5,
    global_max_samples: int = 5000,
    local_indices=None,
) -> dict:
    """
    Generate compact global and local SHAP outputs for dashboard consumption.
    """
    xai_dir = os.path.join(output_dir, "xai")
    global_dir = os.path.join(xai_dir, "global")
    local_dir = os.path.join(xai_dir, "local")
    os.makedirs(global_dir, exist_ok=True)
    os.makedirs(local_dir, exist_ok=True)

    explainer = _load_or_build_explainer(model, model_path, selected_features)
    base_value = extract_base_value(explainer, class_index=1)

    # Global explanation on a bounded sample for latency control
    if len(X) > global_max_samples:
        global_idx = np.random.RandomState(42).choice(len(X), size=global_max_samples, replace=False)
        X_global = X.iloc[global_idx]
    else:
        global_idx = np.arange(len(X))
        X_global = X

    shap_global = compute_shap_values(explainer, X_global)
    importance_df = compute_global_feature_importance(shap_global, selected_features)

    global_importance_csv = os.path.join(global_dir, "feature_importance.csv")
    global_importance_json = os.path.join(global_dir, "feature_importance.json")
    importance_df.to_csv(global_importance_csv, index=False)
    with open(global_importance_json, "w") as f:
        json.dump(importance_df.to_dict("records"), f, indent=2)

    # Local explanations for requested or top-risk employees
    if local_indices is None:
        sorted_idx = np.argsort(y_prob)[::-1]
        local_indices = sorted_idx[: min(local_count, len(sorted_idx))].tolist()
    else:
        local_indices = [int(i) for i in local_indices if 0 <= int(i) < len(X)]

    X_local = X.iloc[local_indices]
    shap_local = compute_shap_values(explainer, X_local)

    explanation_files = []
    contribution_files = []
    summary_rows = []

    for pos, idx in enumerate(local_indices):
        local_exp = extract_local_explanation(
            shap_values=shap_local,
            base_value=base_value,
            feature_values=X_local.values,
            feature_names=selected_features,
            instance_idx=pos,
        )

        nl_explanation = generate_explanation(
            prediction_proba=float(y_prob[idx]),
            shap_values=shap_local[pos],
            feature_values=X_local.values[pos],
            feature_names=selected_features,
            threshold=threshold,
            base_value=base_value,
        )

        contrib_df = create_contribution_table(
            shap_values=shap_local,
            feature_values=X_local.values,
            feature_names=selected_features,
            instance_idx=pos,
        )
        contrib_path = os.path.join(local_dir, f"employee_{idx}_contributions.csv")
        contrib_df.to_csv(contrib_path, index=False)

        actual_resigned = None
        if "Resigned" in raw_df.columns:
            actual_resigned = int(raw_df.iloc[idx]["Resigned"])

        explanation_obj = {
            "employee_index": int(idx),
            "prediction_probability": float(y_prob[idx]),
            "predicted_resigned": int(y_prob[idx] >= threshold),
            "risk_level": str(_risk_level(np.array([y_prob[idx]]), threshold)[0]),
            "actual_resigned": actual_resigned,
            "threshold": float(threshold),
            "base_value": float(base_value),
            "explanation_space": "model_score",
            "top_risk_increasing_factors": local_exp["top_positive"],
            "top_risk_reducing_factors": local_exp["top_negative"],
            "explanation_text": nl_explanation,
            "contributions_csv": contrib_path,
        }
        exp_path = os.path.join(local_dir, f"employee_{idx}_explanation.json")
        with open(exp_path, "w") as f:
            json.dump(explanation_obj, f, indent=2)

        explanation_files.append(exp_path)
        contribution_files.append(contrib_path)
        summary_rows.append(
            {
                "employee_index": int(idx),
                "prediction_probability": float(y_prob[idx]),
                "risk_level": str(explanation_obj["risk_level"]),
                "actual_resigned": actual_resigned,
                "explanation_json": exp_path,
                "contributions_csv": contrib_path,
            }
        )

    local_summary_path = os.path.join(local_dir, "local_explanations_summary.csv")
    pd.DataFrame(summary_rows).to_csv(local_summary_path, index=False)

    xai_summary = {
        "base_value": float(base_value),
        "global_sample_size": int(len(X_global)),
        "global_feature_importance_csv": global_importance_csv,
        "global_feature_importance_json": global_importance_json,
        "local_explanations_summary_csv": local_summary_path,
        "local_explanation_files": explanation_files,
        "local_contribution_files": contribution_files,
    }
    with open(os.path.join(xai_dir, "xai_summary.json"), "w") as f:
        json.dump(xai_summary, f, indent=2)

    return xai_summary


def run_inference_pipeline(
    input_file: str,
    output_dir: str = "pipeline_output",
    model_path: str = "models/best_model_tuned.pkl",
    metadata_path: str = "models/tuning_metadata.json",
    selected_features_path: str = "data/selected_features.json",
    scaler_path: str = "data/scaler.pkl",
    fe_params_path: str = "models/feature_engineering_params.json",
    run_xai: bool = True,
    xai_local_count: int = 5,
    xai_global_max_samples: int = 5000,
    xai_local_indices=None,
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
    risk_level = _risk_level(y_prob, threshold)

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

    if run_xai:
        xai_summary = _generate_xai_outputs(
            model=model,
            model_path=model_path,
            X=X,
            y_prob=y_prob,
            threshold=threshold,
            selected_features=selected_features,
            output_dir=output_dir,
            raw_df=df_input,
            local_count=xai_local_count,
            global_max_samples=xai_global_max_samples,
            local_indices=xai_local_indices,
        )
        summary["xai"] = xai_summary

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
