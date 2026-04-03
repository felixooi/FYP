"""
Module: Department Explanation Generator
Layer 1 of the Department-Level XAI Summary system.

Aggregates per-employee SHAP values by department and generates:
  - Structured dept-level risk statistics (feature importances, risk distribution)
  - A natural-language department risk summary (template-based, no LLM)

Output is persisted to pipeline_output/xai/dept/ and consumed by /api/dept-xai.
"""

import json
import os
from typing import Dict, List, Any

import numpy as np
import pandas as pd


# ── Constants ─────────────────────────────────────────────────────────────────
HIGH_RISK_THRESHOLD = 0.70   # matches _risk_level() in end_to_end_pipeline.py
MED_RISK_THRESHOLD  = 0.40


def _risk_label(prob: float) -> str:
    if prob >= HIGH_RISK_THRESHOLD:
        return "HIGH"
    elif prob >= MED_RISK_THRESHOLD:
        return "MEDIUM"
    return "LOW"


# ── Core aggregation ──────────────────────────────────────────────────────────

def aggregate_dept_shap(
    local_explanation_files: List[str],
    predictions_csv: str,
) -> Dict[str, Any]:
    """
    Aggregate per-employee SHAP values by department.

    Args:
        local_explanation_files: List of paths to employee_{idx}_explanation.json files.
        predictions_csv: Path to inference_predictions.csv (contains Department column).

    Returns:
        Dict mapping department name → aggregated stats dict.
    """
    # Load prediction CSV for dept lookup
    df_preds = pd.read_csv(predictions_csv)
    if "Department" not in df_preds.columns:
        raise ValueError("predictions_csv must contain a 'Department' column.")

    dept_data: Dict[str, List[Dict]] = {}

    for exp_file in local_explanation_files:
        if not os.path.exists(exp_file):
            continue
        try:
            with open(exp_file, "r") as f:
                exp = json.load(f)
        except Exception:
            continue

        emp_idx = exp.get("employee_index")
        if emp_idx is None or emp_idx >= len(df_preds):
            continue

        dept = df_preds.iloc[emp_idx].get("Department", "Unknown")
        if not dept or pd.isna(dept):
            dept = "Unknown"

        if dept not in dept_data:
            dept_data[dept] = []

        dept_data[dept].append({
            "employee_index": emp_idx,
            "prediction_probability": exp.get("prediction_probability", 0.0),
            "risk_level": exp.get("risk_level", "LOW"),
            "top_risk_increasing_factors": exp.get("top_risk_increasing_factors", []),
            "top_risk_reducing_factors": exp.get("top_risk_reducing_factors", []),
        })

    # Build aggregated stats per department
    dept_summaries: Dict[str, Any] = {}

    for dept, records in dept_data.items():
        n = len(records)
        probs = [r["prediction_probability"] for r in records]

        risk_dist = {
            "HIGH":   sum(1 for p in probs if p >= HIGH_RISK_THRESHOLD),
            "MEDIUM": sum(1 for p in probs if MED_RISK_THRESHOLD <= p < HIGH_RISK_THRESHOLD),
            "LOW":    sum(1 for p in probs if p < MED_RISK_THRESHOLD),
        }

        # Aggregate SHAP values across all employees in department
        feature_shap_totals: Dict[str, List[float]] = {}
        feature_occurrence: Dict[str, int] = {}  # how many employees each feature appears in

        for record in records:
            seen_features = set()
            for factor in record["top_risk_increasing_factors"]:
                feat  = factor.get("feature", "")
                shval = float(factor.get("shap_value", factor.get("impact", 0.0)))
                if feat not in feature_shap_totals:
                    feature_shap_totals[feat] = []
                feature_shap_totals[feat].append(shval)
                if feat not in seen_features:
                    feature_occurrence[feat] = feature_occurrence.get(feat, 0) + 1
                    seen_features.add(feat)

        # Rank features by mean SHAP magnitude across employees
        ranked_features = []
        for feat, shap_list in feature_shap_totals.items():
            mean_shap = float(np.mean(shap_list))
            occurrence_pct = round((feature_occurrence.get(feat, 0) / n) * 100, 1)
            ranked_features.append({
                "feature": feat,
                "mean_shap": round(mean_shap, 4),
                "occurrence_count": feature_occurrence.get(feat, 0),
                "occurrence_pct": occurrence_pct,
            })

        ranked_features.sort(key=lambda x: -x["mean_shap"])

        dept_summaries[dept] = {
            "department": dept,
            "total_employees_with_xai": n,
            "avg_attrition_probability": round(float(np.mean(probs)) * 100, 1),
            "risk_distribution": risk_dist,
            "top_risk_drivers": ranked_features[:5],
            "explanation_text": _generate_dept_nl_explanation(dept, n, probs, risk_dist, ranked_features),
        }

    return dept_summaries


# ── Natural-Language Template Generator ───────────────────────────────────────

def _generate_dept_nl_explanation(
    dept: str,
    n: int,
    probs: List[float],
    risk_dist: Dict[str, int],
    ranked_features: List[Dict],
) -> str:
    """
    Generate a structured NL explanation for a department's aggregate attrition risk.
    Template-based (no LLM), mirrors explanation_generator.py style.
    """
    avg_prob = float(np.mean(probs)) * 100
    high_count = risk_dist["HIGH"]
    med_count  = risk_dist["MEDIUM"]
    low_count  = risk_dist["LOW"]

    # Overall risk sentence
    lines = [
        f"The {dept} department has an average attrition risk of {avg_prob:.1f}% "
        f"across {n} analysed employee(s). "
        f"Risk distribution: {high_count} HIGH, {med_count} MEDIUM, {low_count} LOW.",
        "",
        "Note: feature contributions are SHAP values in model score space, "
        "not direct probability percentage points.",
        "",
    ]

    if ranked_features:
        primary = ranked_features[0]
        lines.append(
            f"The primary department-level risk driver is {primary['feature'].replace('_', ' ')} "
            f"(mean SHAP: {primary['mean_shap']:.3f}), present in "
            f"{primary['occurrence_count']} of {n} employee(s) "
            f"({primary['occurrence_pct']}% of analysed staff)."
        )
        if len(ranked_features) >= 2:
            secondary = ranked_features[1]
            lines.append(
                f"Secondary driver: {secondary['feature'].replace('_', ' ')} "
                f"(mean SHAP: {secondary['mean_shap']:.3f}, "
                f"affecting {secondary['occurrence_pct']}% of staff)."
            )
        if len(ranked_features) >= 3:
            others = ranked_features[2:5]
            other_str = ", ".join(
                f"{f['feature'].replace('_', ' ')} ({f['mean_shap']:.3f})" for f in others
            )
            lines.append(f"Additional contributing factors include: {other_str}.")
    else:
        lines.append("No significant shared risk factors were identified across employees in this department.")

    if high_count >= n * 0.5:
        lines.append(
            f"\nThis department is at ELEVATED organisational risk — "
            f"over {round((high_count/n)*100)}% of its workforce is classified HIGH RISK. "
            "Immediate management attention and targeted interventions are recommended."
        )
    elif high_count == 0:
        lines.append(
            f"\nNo employees in this department are currently classified as HIGH RISK."
        )

    return "\n".join(lines)


# ── Pipeline integration entry point ─────────────────────────────────────────

def generate_dept_xai_outputs(
    local_explanation_files: List[str],
    predictions_csv: str,
    output_dir: str,
) -> Dict[str, Any]:
    """
    Aggregate department-level XAI and persist to output_dir/xai/dept/.

    Args:
        local_explanation_files: From _generate_xai_outputs() in end_to_end_pipeline.py.
        predictions_csv: Path to inference_predictions.csv.
        output_dir: Pipeline output root (e.g., 'pipeline_output').

    Returns:
        Dict of dept name → summary stats (also written to JSON files).
    """
    dept_dir = os.path.join(output_dir, "xai", "dept")
    os.makedirs(dept_dir, exist_ok=True)

    dept_summaries = aggregate_dept_shap(local_explanation_files, predictions_csv)

    # Write per-department JSON files
    for dept, summary in dept_summaries.items():
        safe_name = dept.replace(" ", "_").replace("/", "-")
        dept_path = os.path.join(dept_dir, f"dept_{safe_name}_xai.json")
        with open(dept_path, "w") as f:
            json.dump(summary, f, indent=2)
        summary["xai_file"] = dept_path

    # Write combined summary
    combined_path = os.path.join(dept_dir, "dept_xai_summary.json")
    with open(combined_path, "w") as f:
        json.dump(dept_summaries, f, indent=2)

    print(f"Department XAI saved to {dept_dir}")
    return dept_summaries
