"""
Hyperparameter tuning script with stratified CV and precision-constrained thresholding.
"""

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score, classification_report, brier_score_loss
from sklearn.calibration import calibration_curve
from scipy.stats import loguniform, randint, uniform
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import xgboost as xgb
import joblib

from modules.model_selection import find_best_threshold
from modules.feature_engineering import fit_workload_feature_params


def constrained_recall_scorer(estimator, X, y, precision_min=0.50, grid_size=101):
    """
    Maximize recall subject to precision >= precision_min using a threshold sweep.
    Returns the best recall for this fold.
    """
    y_prob = estimator.predict_proba(X)[:, 1]
    thresholds = np.linspace(0, 1, grid_size)
    best_recall = 0.0
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        precision = precision_score(y, y_pred, zero_division=0)
        if precision + 1e-12 < precision_min:
            continue
        recall = recall_score(y, y_pred, zero_division=0)
        if recall > best_recall:
            best_recall = recall
    return best_recall


def load_data(data_dir="data"):
    X_train = pd.read_csv(f"{data_dir}/train_data.csv")
    X_val = pd.read_csv(f"{data_dir}/val_data.csv")
    X_test = pd.read_csv(f"{data_dir}/test_data.csv")

    y_train = X_train["Resigned"]
    y_val = X_val["Resigned"]
    y_test = X_test["Resigned"]

    X_train = X_train.drop(columns=["Resigned"])
    X_val = X_val.drop(columns=["Resigned"])
    X_test = X_test.drop(columns=["Resigned"])

    return X_train, X_val, X_test, y_train, y_val, y_test


def save_feature_engineering_params(
    cleaned_data_path="data/cleaned_data.csv",
    output_path="models/feature_engineering_params.json",
):
    """
    Persist feature-engineering params for inference.
    Uses cleaned_data.csv when available.
    """
    if not os.path.exists(cleaned_data_path):
        return None

    df_clean = pd.read_csv(cleaned_data_path)
    params = fit_workload_feature_params(df_clean)
    with open(output_path, "w") as f:
        json.dump(params, f, indent=4)
    return output_path


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    X_train, X_val, X_test, y_train, y_val, y_test = load_data()

    # Base tuning objective; operating-point sweep is applied after tuning.
    base_precision_min = 0.50
    scorer = lambda est, X, y: constrained_recall_scorer(est, X, y, precision_min=base_precision_min)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    precision_constraints = [0.50, 0.60, 0.70]

    models = {
        "Logistic_Regression": (
            LogisticRegression(max_iter=2000, solver="lbfgs"),
            {
                "C": loguniform(1e-3, 1e2),
                "penalty": ["l2"]
            },
            8
        ),
        "XGBoost": (
            xgb.XGBClassifier(
                n_estimators=500,
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1
            ),
            {
                "max_depth": randint(3, 9),
                "learning_rate": loguniform(0.01, 0.3),
                "subsample": uniform(0.7, 0.3),
                "colsample_bytree": uniform(0.7, 0.3),
                "min_child_weight": randint(1, 8),
                "gamma": uniform(0.0, 0.5),
                "reg_alpha": loguniform(1e-4, 1.0),
                "reg_lambda": loguniform(1e-2, 10.0)
            },
            12
        )
    }

    results = []
    best_models = {}

    for name, (model, param_dist, n_iter) in models.items():
        print(f"\nTuning {name}...")
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring=scorer,
            cv=cv,
            n_jobs=-1,
            verbose=0,
            random_state=42
        )
        search.fit(X_train, y_train)

        best_model = search.best_estimator_
        best_models[name] = best_model

        results.append({
            "Model": name,
            "Best_Params": search.best_params_,
            "CV_Best_Recall": search.best_score_
        })

    tuned_df = pd.DataFrame(results)

    # Precision-constraint sweep on validation + test metrics for comparison
    sweep_rows = []
    for _, base_row in tuned_df.iterrows():
        name = base_row["Model"]
        model = best_models[name]
        y_prob_val = model.predict_proba(X_val)[:, 1]
        y_prob_test = model.predict_proba(X_test)[:, 1]
        for precision_min in precision_constraints:
            tuned = find_best_threshold(model, X_val, y_val, precision_min=precision_min)
            threshold = float(tuned["Threshold"])
            y_pred_test = (y_prob_test >= threshold).astype(int)
            sweep_rows.append({
                "Model": name,
                "Precision_Constraint": precision_min,
                "Threshold": threshold,
                "Val_Precision": float(tuned["Precision"]),
                "Val_Recall": float(tuned["Recall"]),
                "Val_F1": float(tuned["F1_Score"]),
                "Val_PR_AUC": float(tuned["PR_AUC"]),
                "Test_Precision": float(precision_score(y_test, y_pred_test, zero_division=0)),
                "Test_Recall": float(recall_score(y_test, y_pred_test, zero_division=0)),
                "Test_F1": float(f1_score(y_test, y_pred_test, zero_division=0)),
                "Test_PR_AUC": float(average_precision_score(y_test, y_prob_test)),
                "Note": tuned.get("note", ""),
                "Best_Params": base_row["Best_Params"],
                "CV_Best_Recall": float(base_row["CV_Best_Recall"])
            })

    sweep_df = pd.DataFrame(sweep_rows).sort_values(
        by=["Val_F1", "Val_Recall", "Val_PR_AUC"],
        ascending=False
    ).reset_index(drop=True)
    sweep_df.to_csv("outputs/precision_constraint_sweep.csv", index=False)
    print("Saved: outputs/precision_constraint_sweep.csv")

    # Final selection from validation sweep (no test leakage in selection)
    best_row = sweep_df.iloc[0]
    best_name = best_row["Model"]
    best_model = best_models[best_name]
    selected_constraint = float(best_row["Precision_Constraint"])

    # Save best model and metadata
    joblib.dump(best_model, f"models/best_model_tuned.pkl")
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "selected_model": best_name,
        "selection_criteria": (
            "Validation sweep: maximize Val_F1 across precision constraints "
            f"{precision_constraints}, tie-break by Val_Recall then Val_PR_AUC"
        ),
        "selected_precision_constraint": selected_constraint,
        "threshold": float(best_row["Threshold"]),
        "best_params": best_row["Best_Params"],
        "cv_best_recall": float(best_row["CV_Best_Recall"]),
        "val_metrics": {
            "precision": float(best_row["Val_Precision"]),
            "recall": float(best_row["Val_Recall"]),
            "f1": float(best_row["Val_F1"]),
            "pr_auc": float(best_row["Val_PR_AUC"])
        }
    }
    with open("models/tuning_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    fe_params_output = save_feature_engineering_params()
    if fe_params_output:
        print(f"Saved: {fe_params_output}")
    else:
        print("Warning: feature engineering params not saved (missing data/cleaned_data.csv)")

    tuned_df.to_csv("outputs/tuning_results.csv", index=False)
    print("\nTuning complete. Results saved to outputs/tuning_results.csv")
    print(sweep_df[[
        "Model", "Precision_Constraint", "Val_Precision", "Val_Recall",
        "Val_F1", "Val_PR_AUC", "Threshold"
    ]].to_string(index=False))

    # Final test report at selected threshold
    threshold = float(best_row["Threshold"])
    y_prob_val = best_model.predict_proba(X_val)[:, 1]
    y_prob_test = best_model.predict_proba(X_test)[:, 1]
    y_pred_test = (y_prob_test >= threshold).astype(int)
    y_pred_val = (y_prob_val >= threshold).astype(int)

    report = classification_report(y_test, y_pred_test, digits=4)
    with open("outputs/final_model_test_report.txt", "w") as f:
        f.write(f"Selected model: {best_name}\n")
        f.write(f"Selected precision constraint: >= {selected_constraint:.2f}\n")
        f.write(f"Selected threshold: {threshold:.3f}\n\n")
        f.write(report)

    test_pr_auc = average_precision_score(y_test, y_prob_test)
    val_brier = brier_score_loss(y_val, y_prob_val)
    test_metrics = {
        "Model": best_name,
        "Precision_Constraint": selected_constraint,
        "Threshold": threshold,
        "Precision": precision_score(y_test, y_pred_test, zero_division=0),
        "Recall": recall_score(y_test, y_pred_test, zero_division=0),
        "F1_Score": f1_score(y_test, y_pred_test, zero_division=0),
        "PR_AUC": test_pr_auc,
        "Validation_Brier_Score": val_brier
    }
    pd.DataFrame([test_metrics]).to_csv("outputs/final_model_test_metrics.csv", index=False)

    # Validation calibration plot for probability reliability
    frac_pos, mean_pred = calibration_curve(y_val, y_prob_val, n_bins=10, strategy="quantile")
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    plt.plot(mean_pred, frac_pos, marker="o", linewidth=2, label=f"{best_name}")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Validation Calibration Curve")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig("outputs/final_model_validation_calibration.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Threshold rationale for viva/readability
    threshold_rationale = (
        "Threshold Selection Rationale\n"
        "===========================\n"
        "Business objective prioritizes catching at-risk employees (high recall),\n"
        "while keeping operational load manageable.\n"
        f"Constraint sweep used: {precision_constraints}.\n"
        f"Selected constraint: Precision >= {selected_constraint:.2f}.\n"
        "Interpretation: selected operating point balances false-positive workload for HR\n"
        "and recall of at-risk employees using validation-first model selection.\n"
        "For Precision >= 0.50, at least 1 in 2 flagged employees is expected to truly resign,\n"
        "which bounds false-positive workload for HR while preserving high recall.\n"
    )
    with open("models/threshold_rationale.txt", "w") as f:
        f.write(threshold_rationale)

    # Model card: one-page deployment/evaluation summary
    model_card = (
        "# Final Model Card\n\n"
        f"- Generated: {datetime.now().isoformat()}\n"
        f"- Model: {best_name}\n"
        f"- Model file: `models/best_model_tuned.pkl`\n"
        f"- Selected precision constraint: >= {selected_constraint:.2f}\n"
        f"- Threshold: {threshold:.3f}\n"
        "- Selection rule: validation sweep over precision constraints; maximize Val_F1, "
        "tie-break by Val_Recall then Val_PR_AUC\n\n"
        "## Data Summary\n"
        f"- Train samples: {len(y_train)}\n"
        f"- Validation samples: {len(y_val)}\n"
        f"- Test samples: {len(y_test)}\n"
        f"- Features used: {X_train.shape[1]}\n"
        f"- Class balance (train, positive rate): {float(y_train.mean()):.4f}\n"
        f"- Class balance (validation, positive rate): {float(y_val.mean()):.4f}\n"
        f"- Class balance (test, positive rate): {float(y_test.mean()):.4f}\n\n"
        "## Validation Performance (selected threshold)\n"
        f"- Precision: {precision_score(y_val, y_pred_val, zero_division=0):.4f}\n"
        f"- Recall: {recall_score(y_val, y_pred_val, zero_division=0):.4f}\n"
        f"- F1: {f1_score(y_val, y_pred_val, zero_division=0):.4f}\n"
        f"- PR-AUC: {average_precision_score(y_val, y_prob_val):.4f}\n"
        f"- Brier score: {val_brier:.4f}\n\n"
        "## Test Performance (selected threshold)\n"
        f"- Precision: {test_metrics['Precision']:.4f}\n"
        f"- Recall: {test_metrics['Recall']:.4f}\n"
        f"- F1: {test_metrics['F1_Score']:.4f}\n"
        f"- PR-AUC: {test_metrics['PR_AUC']:.4f}\n\n"
        "## Artifact Index\n"
        "- `outputs/final_model_test_report.txt`\n"
        "- `outputs/final_model_test_metrics.csv`\n"
        "- `outputs/precision_constraint_sweep.csv`\n"
        "- `outputs/final_model_validation_calibration.png`\n"
        "- `models/tuning_metadata.json`\n"
        "- `models/threshold_rationale.txt`\n"
    )
    with open("models/final_model_card.md", "w") as f:
        f.write(model_card)

    # Consolidate artifacts: remove legacy names to avoid confusion
    for legacy_path in [
        "outputs/tuned_best_model_test_report.txt",
        "outputs/tuned_best_model_test_metrics.csv",
        "outputs/best_model_test_report.txt",
        "outputs/best_model_test_metrics.csv",
    ]:
        if os.path.exists(legacy_path):
            os.remove(legacy_path)

    print("Saved: outputs/final_model_test_report.txt")
    print("Saved: outputs/final_model_test_metrics.csv")
    print("Saved: outputs/final_model_validation_calibration.png")
    print("Saved: models/final_model_card.md")
    print("Saved: models/threshold_rationale.txt")


if __name__ == "__main__":
    main()
