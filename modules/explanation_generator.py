"""
Module: Explanation Generator
Natural language explanation generation for HR stakeholders.

Converts technical SHAP outputs into human-readable, actionable insights.
"""

import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def generate_explanation(prediction_proba, shap_values, feature_values, feature_names, threshold, base_value):
    """
    Generate full natural language explanation for an employee's attrition risk.
    
    Args:
        prediction_proba: Predicted attrition probability (0-1)
        shap_values: SHAP values for this instance (n_features,)
        feature_values: Feature values for this instance (n_features,)
        feature_names: List of feature names
        threshold: Classification threshold
        base_value: Base value (expected value)
    
    Returns:
        String with complete natural language explanation
    """
    logging.info(f"Generating explanation for prediction {prediction_proba:.3f}...")
    
    # Risk level
    risk_level = format_risk_level(prediction_proba, threshold)
    
    # Top contributors
    risk_increasing = describe_top_contributors(shap_values, feature_values, feature_names, top_n=3)
    protective = describe_protective_factors(
        shap_values, feature_values, feature_names, risk_level=risk_level, top_n=2
    )
    
    # Build explanation
    explanation = (
        f"This employee is classified as {risk_level} of attrition "
        f"(probability: {prediction_proba:.1%}).\n\n"
        "Note: feature contributions below are SHAP values in model score space, "
        "not direct probability percentage points.\n\n"
    )
    
    if risk_increasing:
        explanation += risk_increasing + "\n\n"
    
    if protective:
        explanation += protective
    
    return explanation.strip()


def format_risk_level(prediction_proba, threshold):
    """
    Convert prediction probability to risk level category.
    
    Args:
        prediction_proba: Predicted attrition probability (0-1)
        threshold: Classification threshold
    
    Returns:
        String: "HIGH RISK", "MEDIUM RISK", or "LOW RISK"
    """
    if prediction_proba >= threshold + 0.2:
        return "HIGH RISK"
    elif prediction_proba >= threshold:
        return "MEDIUM RISK"
    else:
        return "LOW RISK"


def describe_top_contributors(shap_values, feature_values, feature_names, top_n=3):
    """
    Generate text for top risk-increasing factors.
    
    Args:
        shap_values: SHAP values for this instance (n_features,)
        feature_values: Feature values for this instance (n_features,)
        feature_names: List of feature names
        top_n: Number of top factors to describe
    
    Returns:
        String describing top risk-increasing factors
    """
    # Get positive SHAP values (risk-increasing)
    positive_mask = shap_values > 0
    if not np.any(positive_mask):
        return ""
    
    positive_indices = np.where(positive_mask)[0]
    positive_shap = shap_values[positive_mask]
    sorted_idx = np.argsort(positive_shap)[::-1][:top_n]
    
    top_factors = []
    for idx in sorted_idx:
        feature_idx = positive_indices[idx]
        feature = feature_names[feature_idx]
        value = feature_values[feature_idx]
        shap_val = shap_values[feature_idx]
        # Format feature value
        value_str = f"{value:.2f}" if isinstance(value, (int, float)) else str(value)

        top_factors.append({
            'feature': feature,
            'value': value_str,
            'impact': shap_val
        })
    
    if not top_factors:
        return ""
    
    # Build description
    primary = top_factors[0]
    text = (
        f"The primary contributing factor is {_format_feature_name(primary['feature'])} "
        f"({primary['value']}), with SHAP impact {primary['impact']:.3f} score units."
    )
    
    if len(top_factors) > 1:
        additional = [f"{_format_feature_name(f['feature'])} ({f['value']})" for f in top_factors[1:]]
        if len(additional) == 1:
            text += f" Additionally, {additional[0]} further elevates the risk with SHAP impact {top_factors[1]['impact']:.3f}."
        else:
            text += (
                f" Additionally, {', '.join(additional[:-1])} and {additional[-1]} "
                f"further elevate the risk with SHAP impacts {top_factors[1]['impact']:.3f} "
                f"and {top_factors[2]['impact']:.3f} respectively."
            )
    
    return text


def describe_protective_factors(shap_values, feature_values, feature_names, risk_level, top_n=2):
    """
    Generate text for top risk-reducing factors.
    
    Args:
        shap_values: SHAP values for this instance (n_features,)
        feature_values: Feature values for this instance (n_features,)
        feature_names: List of feature names
        top_n: Number of top factors to describe
    
    Returns:
        String describing top risk-reducing factors
    """
    # Get negative SHAP values (risk-reducing)
    negative_mask = shap_values < 0
    if not np.any(negative_mask):
        return ""
    
    negative_indices = np.where(negative_mask)[0]
    negative_shap = shap_values[negative_mask]
    sorted_idx = np.argsort(negative_shap)[:top_n]  # Most negative first
    
    top_factors = []
    for idx in sorted_idx:
        feature_idx = negative_indices[idx]
        feature = feature_names[feature_idx]
        value = feature_values[feature_idx]
        shap_val = shap_values[feature_idx]
        # Format feature value
        value_str = f"{value:.2f}" if isinstance(value, (int, float)) else str(value)

        top_factors.append({
            'feature': feature,
            'value': value_str,
            'impact': shap_val
        })
    
    if not top_factors:
        return ""
    
    # Build description
    if len(top_factors) == 1:
        text = (
            f"Positive factors such as {_format_feature_name(top_factors[0]['feature'])} "
            f"({top_factors[0]['value']}) help reduce the overall risk "
            f"(SHAP impact {top_factors[0]['impact']:.3f})."
        )
    else:
        factors_str = " and ".join([f"{_format_feature_name(f['feature'])} ({f['value']})" for f in top_factors])
        impacts_str = " and ".join([f"{f['impact']:.3f}" for f in top_factors])
        text = (
            f"Positive factors such as {factors_str} help reduce the overall risk "
            f"(SHAP impacts {impacts_str} respectively)."
        )

    if risk_level in ("MEDIUM RISK", "HIGH RISK"):
        text += " These factors are not enough to move the employee below the current risk threshold."
    
    return text


def quantify_impact(shap_value, base_value):
    """
    Return SHAP impact in model score units.
    
    Args:
        shap_value: SHAP value for a feature
        base_value: Base value (expected value)
    
    Returns:
        Float: Impact in model score units
    """
    return float(shap_value)


def _format_feature_name(feature_name):
    """
    Convert feature name to human-readable format.
    
    Args:
        feature_name: Raw feature name (e.g., "Employee_Satisfaction_Score")
    
    Returns:
        String: Human-readable name (e.g., "employee satisfaction score")
    """
    # Replace underscores with spaces and convert to lowercase
    readable = feature_name.replace('_', ' ').lower()
    
    # Handle specific cases
    if 'ratio' in readable:
        readable = readable.replace('ratio', '(ratio)')
    if 'gap' in readable:
        readable = readable.replace('gap', '(gap)')
    
    return readable
