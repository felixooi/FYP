"""
Module: Explanation Analysis
Global and local explainability logic with SHAP visualizations.

Global Explainability: Model-wide feature importance and impact patterns
Local Explainability: Individual prediction breakdowns
"""

import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def compute_global_feature_importance(shap_values, feature_names):
    """
    Compute global feature importance using mean absolute SHAP values.
    
    Args:
        shap_values: SHAP values array (n_samples, n_features)
        feature_names: List of feature names
    
    Returns:
        DataFrame with features ranked by importance
    """
    logging.info("Computing global feature importance...")
    
    if shap_values.ndim != 2:
        raise ValueError(f"Expected 2D shap_values, got shape {shap_values.shape}")
    if shap_values.shape[1] != len(feature_names):
        raise ValueError("Feature alignment error: SHAP column count does not match feature names.")

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': mean_abs_shap
    }).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)
    
    importance_df['rank'] = range(1, len(importance_df) + 1)
    
    logging.info(f"Top 3 features: {importance_df['feature'].head(3).tolist()}")
    return importance_df


def plot_shap_summary(shap_values, X_data, feature_names, save_path):
    """
    Generate SHAP summary plot (beeswarm).
    
    Args:
        shap_values: SHAP values array (n_samples, n_features)
        X_data: Feature values (n_samples, n_features)
        feature_names: List of feature names
        save_path: Path to save plot
    """
    logging.info("Generating SHAP summary plot...")
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_data, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"SHAP summary plot saved to {save_path}")


def plot_feature_importance_bar(importance_df, top_n, save_path):
    """
    Generate bar plot of top N features.
    
    Args:
        importance_df: DataFrame from compute_global_feature_importance
        top_n: Number of top features to plot
        save_path: Path to save plot
    """
    logging.info(f"Generating feature importance bar plot (top {top_n})...")
    
    top_features = importance_df.head(top_n)
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(top_features)), top_features['mean_abs_shap'], color='steelblue')
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Mean |SHAP value| (average impact on model output)', fontsize=11)
    plt.ylabel('Feature', fontsize=11)
    plt.title(f'Top {top_n} Features by Global Importance', fontsize=13, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Feature importance bar plot saved to {save_path}")


def plot_dependence(shap_values, X_data, feature_names, feature_name, save_path):
    """
    Generate SHAP dependence plot for a single feature.
    
    Args:
        shap_values: SHAP values array (n_samples, n_features)
        X_data: Feature values (n_samples, n_features)
        feature_names: List of feature names
        feature_name: Feature to plot
        save_path: Path to save plot
    """
    logging.info(f"Generating dependence plot for {feature_name}...")
    
    feature_idx = feature_names.index(feature_name)
    if np.unique(X_data[:, feature_idx]).shape[0] <= 1:
        logging.warning(f"Skipping dependence plot for {feature_name}: constant feature.")
        return

    plt.figure(figsize=(8, 6))
    shap.dependence_plot(
        feature_name,
        shap_values,
        X_data,
        feature_names=feature_names,
        interaction_index=None,
        show=False
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Dependence plot saved to {save_path}")


def extract_local_explanation(shap_values, base_value, feature_values, feature_names, instance_idx):
    """
    Extract SHAP explanation for a single employee.
    
    Args:
        shap_values: SHAP values array (n_samples, n_features)
        base_value: Base value (expected value)
        feature_values: Feature values array (n_samples, n_features)
        feature_names: List of feature names
        instance_idx: Index of employee to explain
    
    Returns:
        Dictionary with top positive/negative contributors
    """
    logging.info(f"Extracting local explanation for instance {instance_idx}...")
    
    instance_shap = shap_values[instance_idx]
    instance_features = feature_values[instance_idx]
    
    # Create DataFrame for sorting
    contrib_df = pd.DataFrame({
        'feature': feature_names,
        'feature_value': instance_features,
        'shap_value': instance_shap
    })
    
    # Separate positive and negative contributors
    positive = contrib_df[contrib_df['shap_value'] > 0].sort_values('shap_value', ascending=False)
    negative = contrib_df[contrib_df['shap_value'] < 0].sort_values('shap_value', ascending=True)
    
    return {
        'instance_idx': instance_idx,
        'base_value': base_value,
        'prediction_score': float(base_value + instance_shap.sum()),
        'top_positive': positive.head(5).to_dict('records'),
        'top_negative': negative.head(5).to_dict('records')
    }


def plot_waterfall(shap_values, base_value, feature_values, feature_names, instance_idx, save_path):
    """
    Generate SHAP waterfall plot for individual prediction.
    
    Args:
        shap_values: SHAP values array (n_samples, n_features)
        base_value: Base value (expected value)
        feature_values: Feature values array (n_samples, n_features)
        feature_names: List of feature names
        instance_idx: Index of employee to explain
        save_path: Path to save plot
    """
    logging.info(f"Generating waterfall plot for instance {instance_idx}...")
    
    # Create Explanation object for waterfall plot
    explanation = shap.Explanation(
        values=shap_values[instance_idx],
        base_values=base_value,
        data=feature_values[instance_idx],
        feature_names=feature_names
    )
    
    plt.figure(figsize=(10, 8))
    shap.waterfall_plot(explanation, show=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Waterfall plot saved to {save_path}")


def create_contribution_table(shap_values, feature_values, feature_names, instance_idx):
    """
    Create tabular breakdown of feature contributions.
    
    Args:
        shap_values: SHAP values array (n_samples, n_features)
        feature_values: Feature values array (n_samples, n_features)
        feature_names: List of feature names
        instance_idx: Index of employee to explain
    
    Returns:
        DataFrame with feature contributions sorted by absolute SHAP value
    """
    logging.info(f"Creating contribution table for instance {instance_idx}...")
    
    contrib_df = pd.DataFrame({
        'feature': feature_names,
        'feature_value': feature_values[instance_idx],
        'shap_value': shap_values[instance_idx],
        'abs_shap_value': np.abs(shap_values[instance_idx])
    }).sort_values('abs_shap_value', ascending=False).reset_index(drop=True)
    
    contrib_df['rank'] = range(1, len(contrib_df) + 1)
    
    return contrib_df[['rank', 'feature', 'feature_value', 'shap_value']]
