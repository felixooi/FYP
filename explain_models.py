"""
Phase B: Explainable AI - Main Orchestration Script
Generates global and local explanations using SHAP for attrition predictions.

Usage:
    python explain_models.py                    # Full pipeline
    python explain_models.py --global-only      # Global explanations only
    python explain_models.py --employee-idx 42  # Explain specific employee
"""

import pandas as pd
import numpy as np
import json
import os
import sys
import argparse
import joblib
import hashlib
from datetime import datetime

# Add modules to path
sys.path.append('modules')
from end_to_end_pipeline import _load_artifacts as _load_inference_artifacts, _preprocess_for_inference
from modules.data_ingestion import load_data

from modules.explainability import (
    initialize_shap_explainer, compute_shap_values,
    save_explainer, load_explainer, validate_features, verify_efficiency_axiom,
    detect_model_type, extract_base_value
)
from modules.explanation_analysis import (
    compute_global_feature_importance, plot_shap_summary,
    plot_feature_importance_bar, plot_dependence,
    extract_local_explanation, plot_waterfall, create_contribution_table
)
from modules.explanation_generator import generate_explanation


def _hash_dataframe(df):
    """Stable hash for cache invalidation checks."""
    return hashlib.md5(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()


def _is_cache_valid(cache_meta_path, expected_meta):
    if not os.path.exists(cache_meta_path):
        return False
    with open(cache_meta_path, "r") as f:
        cache_meta = json.load(f)
    for k, v in expected_meta.items():
        if cache_meta.get(k) != v:
            return False
    return True


def _hash_file(path):
    """Stable hash for file contents."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_artifacts(input_file=None):
    """Load model artifacts and explanation dataset (validation by default, or external input file)."""
    print("Loading artifacts...")

    model_path = 'models/best_model_tuned.pkl'
    model, threshold, feature_names, scaler, fe_params = _load_inference_artifacts(
        model_path=model_path,
        metadata_path='models/tuning_metadata.json',
        selected_features_path='data/selected_features.json',
        scaler_path='data/scaler.pkl',
        fe_params_path='models/feature_engineering_params.json',
    )

    # Load tuning metadata for model name
    with open('models/tuning_metadata.json', 'r') as f:
        tuning_meta = json.load(f)
    model_name = tuning_meta['selected_model']

    # Load train data for SHAP background (methodologically preferred)
    train_data = pd.read_csv('data/train_data.csv')
    X_train = train_data.drop(columns=['Resigned'])

    # Validate and enforce feature order for train
    validate_features(X_train, feature_names)
    X_train = X_train[feature_names]

    # Explanation dataset: external input file or validation split
    dataset_source = "data/val_data.csv"
    preprocessing_mode = "preprocessed_validation_split"
    if input_file:
        dataset_source = input_file
        raw_df = load_data(filepath=input_file)
        y_data = raw_df['Resigned'] if 'Resigned' in raw_df.columns else None
        X_data, fe_source = _preprocess_for_inference(
            df_raw=raw_df,
            selected_features=feature_names,
            scaler=scaler,
            fe_params=fe_params,
        )
        preprocessing_mode = f"inference_preprocessing::{fe_source}"
    else:
        val_data = pd.read_csv('data/val_data.csv')
        y_data = val_data['Resigned']
        X_data = val_data.drop(columns=['Resigned'])
        validate_features(X_data, feature_names)
        X_data = X_data[feature_names]

    print(f"[OK] Model: {model_name}")
    print(f"[OK] Threshold: {threshold}")
    print(f"[OK] Dataset source: {dataset_source}")
    print(f"[OK] Samples: {len(X_data)}")
    print(f"[OK] Features: {len(feature_names)}")
    print(f"[OK] Preprocessing mode: {preprocessing_mode}")

    return model, model_path, X_train, X_data, y_data, feature_names, threshold, model_name, dataset_source, preprocessing_mode


def generate_global_explanations(model, model_path, X_train, X_val, feature_names, output_dir, model_name):
    """Generate global explainability outputs."""
    print("\n" + "="*80)
    print("GLOBAL EXPLAINABILITY")
    print("="*80)
    
    # Initialize SHAP explainer with background dataset from TRAIN split.
    explainer_path = f"{output_dir}/explainers/shap_explainer.pkl"
    explainer_meta_path = f"{output_dir}/explainers/shap_explainer_meta.json"
    shap_values_meta_path = f"{output_dir}/global/shap_values_meta.json"
    model_type = detect_model_type(model)
    X_background = X_train.sample(n=min(100, len(X_train)), random_state=42)
    cache_meta = {
        "model_name": model_name,
        "model_type": model_type,
        "feature_names": feature_names,
        "background_hash": _hash_dataframe(X_background),
        "model_hash": _hash_file(model_path),
    }
    
    if os.path.exists(explainer_path) and _is_cache_valid(explainer_meta_path, cache_meta):
        print("Loading cached SHAP explainer...")
        explainer = load_explainer(explainer_path)
    else:
        print("Initializing SHAP explainer (this may take 30-60 seconds)...")
        explainer = initialize_shap_explainer(model, X_background, model_type=model_type)
        os.makedirs(f"{output_dir}/explainers", exist_ok=True)
        save_explainer(explainer, explainer_path)
        with open(explainer_meta_path, "w") as f:
            json.dump(cache_meta, f, indent=2)
    
    # Compute SHAP values for full validation set
    shap_values_path = f"{output_dir}/global/shap_values.pkl"
    
    shap_cache_meta = {
        "val_hash": _hash_dataframe(X_val),
        "feature_names": feature_names,
        "model_name": model_name,
    }
    if os.path.exists(shap_values_path) and _is_cache_valid(shap_values_meta_path, shap_cache_meta):
        print("Loading cached SHAP values...")
        shap_values = joblib.load(shap_values_path)
    else:
        print("Computing SHAP values (this may take 30-60 seconds)...")
        shap_values = compute_shap_values(explainer, X_val)
        os.makedirs(f"{output_dir}/global", exist_ok=True)
        joblib.dump(shap_values, shap_values_path)
        with open(shap_values_meta_path, "w") as f:
            json.dump(shap_cache_meta, f, indent=2)
        print(f"[OK] SHAP values saved to {shap_values_path}")
    
    # Verify efficiency axiom in model score space (LR: log-odds).
    base_value = extract_base_value(explainer, class_index=1)
    if hasattr(model, "decision_function"):
        predictions_score = model.decision_function(X_val)
        verify_efficiency_axiom(shap_values, predictions_score, base_value)
    else:
        print("[WARN] Skipping efficiency check: model output space is not guaranteed to match SHAP raw space.")
    
    # Compute global feature importance
    importance_df = compute_global_feature_importance(shap_values, feature_names)
    print("\nTop 10 Features by Global Importance:")
    print(importance_df.head(10).to_string(index=False))
    
    # Save feature importance
    importance_path = f"{output_dir}/global/feature_importance.json"
    importance_dict = {
        'top_features': importance_df.head(10).to_dict('records'),
        'all_features': importance_df.to_dict('records')
    }
    with open(importance_path, 'w') as f:
        json.dump(importance_dict, f, indent=2)
    print(f"[OK] Feature importance saved to {importance_path}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # SHAP summary plot (beeswarm)
    plot_shap_summary(
        shap_values, X_val.values, feature_names,
        save_path=f"{output_dir}/global/shap_summary.png"
    )
    
    # Feature importance bar plot (top 10)
    plot_feature_importance_bar(
        importance_df, top_n=10,
        save_path=f"{output_dir}/global/feature_importance_bar.png"
    )
    
    # Optional: Dependence plots for top 3 features
    for i, feature in enumerate(importance_df['feature'].head(3)):
        plot_dependence(
            shap_values, X_val.values, feature_names, feature,
            save_path=f"{output_dir}/global/dependence_{i+1}_{feature}.png"
        )
    
    print("[OK] Global explanations complete")
    
    return explainer, shap_values, base_value, importance_df


def generate_local_explanations(model, explainer, shap_values, X_val, y_val, feature_names, threshold, base_value, output_dir, specific_idx=None):
    """Generate local explainability outputs for sample employees."""
    print("\n" + "="*80)
    print("LOCAL EXPLAINABILITY")
    print("="*80)
    
    os.makedirs(f"{output_dir}/local", exist_ok=True)
    
    # Get predictions
    predictions = model.predict_proba(X_val)[:, 1]
    
    # Select sample employees
    if specific_idx is not None:
        if specific_idx < 0 or specific_idx >= len(X_val):
            raise IndexError(f"employee index {specific_idx} out of range [0, {len(X_val)-1}]")
        sample_indices = [specific_idx]
        print(f"Explaining employee at index {specific_idx}")
    else:
        # High-risk (top 10)
        high_risk_idx = np.argsort(predictions)[-10:][::-1]
        
        # Low-risk (top 10)
        low_risk_idx = np.argsort(predictions)[:10]
        
        # Borderline (10 near threshold)
        diff_from_threshold = np.abs(predictions - threshold)
        borderline_idx = np.argsort(diff_from_threshold)[:10]
        
        sample_indices = np.concatenate([high_risk_idx, low_risk_idx, borderline_idx])
        print(f"Selected {len(sample_indices)} sample employees (10 high-risk, 10 low-risk, 10 borderline)")
    
    # Generate explanations for each sample
    explanations_summary = []
    
    for idx in sample_indices:
        pred_proba = predictions[idx]
        actual = None if y_val is None else int(y_val.iloc[idx])
        
        # Extract local explanation
        local_exp = extract_local_explanation(
            shap_values, base_value, X_val.values, feature_names, idx
        )
        
        # Generate natural language explanation
        nl_explanation = generate_explanation(
            pred_proba, shap_values[idx], X_val.values[idx],
            feature_names, threshold, base_value
        )
        
        # Generate waterfall plot
        plot_waterfall(
            shap_values, base_value, X_val.values, feature_names, idx,
            save_path=f"{output_dir}/local/employee_{idx}_waterfall.png"
        )
        
        # Create contribution table
        contrib_table = create_contribution_table(
            shap_values, X_val.values, feature_names, idx
        )
        contrib_table.to_csv(f"{output_dir}/local/employee_{idx}_contributions.csv", index=False)
        
        # Save JSON explanation
        explanation_json = {
            'employee_index': int(idx),
            'prediction_probability': float(pred_proba),
            'actual_resigned': actual,
            'risk_level': 'HIGH' if pred_proba >= threshold + 0.2 else 'MEDIUM' if pred_proba >= threshold else 'LOW',
            'base_value': float(base_value),
            'explanation_space': 'model_score',
            'threshold': float(threshold),
            'top_risk_increasing_factors': [
                {
                    'feature': item['feature'],
                    'feature_value': float(item['feature_value']),
                    'shap_value': float(item['shap_value']),
                    'impact_score_units': float(item['shap_value'])
                }
                for item in local_exp['top_positive']
            ],
            'top_risk_reducing_factors': [
                {
                    'feature': item['feature'],
                    'feature_value': float(item['feature_value']),
                    'shap_value': float(item['shap_value']),
                    'impact_score_units': float(item['shap_value'])
                }
                for item in local_exp['top_negative']
            ],
            'explanation_text': nl_explanation,
            'visualizations': {
                'waterfall_plot': f"outputs/explainability/local/employee_{idx}_waterfall.png",
                'contributions_table': f"outputs/explainability/local/employee_{idx}_contributions.csv"
            }
        }
        
        with open(f"{output_dir}/local/employee_{idx}_explanation.json", 'w') as f:
            json.dump(explanation_json, f, indent=2)
        
        explanations_summary.append({
            'employee_index': int(idx),
            'prediction_probability': float(pred_proba),
            'actual_resigned': actual,
            'risk_level': explanation_json['risk_level']
        })
        
        if specific_idx is not None:
            print(f"\n{nl_explanation}")
    
    # Save summary
    summary_df = pd.DataFrame(explanations_summary)
    summary_df.to_csv(f"{output_dir}/local/explanations_summary.csv", index=False)
    
    print(f"\n[OK] Generated explanations for {len(sample_indices)} employees")
    print(f"[OK] Local explanations complete")
    
    return explanations_summary


def main():
    """Execute Phase B: Explainable AI pipeline."""
    parser = argparse.ArgumentParser(description='Phase B: Explainable AI')
    parser.add_argument('--global-only', action='store_true', help='Generate only global explanations')
    parser.add_argument('--employee-idx', type=int, help='Explain specific employee by index')
    parser.add_argument('--input-file', type=str, help='Optional external dataset for inference-time explanations')
    args = parser.parse_args()
    
    print("="*80)
    print("PHASE B: EXPLAINABLE AI")
    print("="*80)
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    # Create output directory
    output_dir = 'outputs/explainability'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load artifacts
    model, model_path, X_train, X_val, y_val, feature_names, threshold, model_name, dataset_source, preprocessing_mode = load_artifacts(
        input_file=args.input_file
    )
    
    # Generate global explanations
    explainer, shap_values, base_value, importance_df = generate_global_explanations(
        model, model_path, X_train, X_val, feature_names, output_dir, model_name
    )
    
    # Generate local explanations (unless global-only flag)
    if not args.global_only:
        generate_local_explanations(
            model, explainer, shap_values, X_val, y_val,
            feature_names, threshold, base_value, output_dir,
            specific_idx=args.employee_idx
        )
    
    # Save metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'model_name': model_name,
        'threshold': threshold,
        'dataset_source': dataset_source,
        'preprocessing_mode': preprocessing_mode,
        'n_samples': len(X_val),
        'n_features': len(feature_names),
        'top_3_features': importance_df['feature'].head(3).tolist(),
        'outputs': {
            'global': f"{output_dir}/global/",
            'local': f"{output_dir}/local/",
            'explainer': f"{output_dir}/explainers/"
        }
    }
    
    with open(f"{output_dir}/metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "="*80)
    print("PHASE B COMPLETE")
    print("="*80)
    print(f"[OK] Global explanations: {output_dir}/global/")
    print(f"[OK] Local explanations: {output_dir}/local/")
    print(f"[OK] Cached explainer: {output_dir}/explainers/")
    print(f"[OK] Metadata: {output_dir}/metadata.json")


if __name__ == "__main__":
    main()
