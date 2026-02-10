"""
Hyperparameter tuning script with stratified CV and precision-constrained thresholding.
"""

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score, classification_report
from scipy.stats import loguniform, randint, uniform
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import joblib

from modules.model_selection import find_best_threshold


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


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    X_train, X_val, X_test, y_train, y_val, y_test = load_data()

    precision_min = 0.50
    scorer = lambda est, X, y: constrained_recall_scorer(est, X, y, precision_min=precision_min)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

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

        # Threshold tuning on validation set
        tuned = find_best_threshold(best_model, X_val, y_val, precision_min=precision_min)

        results.append({
            "Model": name,
            "Best_Params": search.best_params_,
            "CV_Best_Recall": search.best_score_,
            "Val_Precision": tuned["Precision"],
            "Val_Recall": tuned["Recall"],
            "Val_F1": tuned["F1_Score"],
            "Val_PR_AUC": tuned["PR_AUC"],
            "Threshold": tuned["Threshold"],
            "Note": tuned.get("note", "")
        })

    results_df = pd.DataFrame(results).sort_values(
        by=["Val_Recall", "Val_PR_AUC", "Val_F1"],
        ascending=False
    ).reset_index(drop=True)

    # Select best model using the same criteria
    best_row = results_df.iloc[0]
    best_name = best_row["Model"]
    best_model = best_models[best_name]

    # Save best model and metadata
    joblib.dump(best_model, f"models/best_model_tuned.pkl")
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "selected_model": best_name,
        "selection_criteria": f"Recall (primary), Precision>={precision_min:.2f} constraint, PR_AUC secondary",
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

    results_df.to_csv("outputs/tuning_results.csv", index=False)
    print("\nTuning complete. Results saved to outputs/tuning_results.csv")
    print(results_df.to_string(index=False))

    # Final test report at selected threshold
    threshold = float(best_row["Threshold"])
    y_prob_test = best_model.predict_proba(X_test)[:, 1]
    y_pred_test = (y_prob_test >= threshold).astype(int)

    report = classification_report(y_test, y_pred_test, digits=4)
    with open("outputs/tuned_best_model_test_report.txt", "w") as f:
        f.write(f"Selected model: {best_name}\n")
        f.write(f"Selected threshold: {threshold:.3f}\n\n")
        f.write(report)

    test_metrics = {
        "Model": best_name,
        "Threshold": threshold,
        "Precision": precision_score(y_test, y_pred_test, zero_division=0),
        "Recall": recall_score(y_test, y_pred_test, zero_division=0),
        "F1_Score": f1_score(y_test, y_pred_test, zero_division=0)
    }
    pd.DataFrame([test_metrics]).to_csv("outputs/tuned_best_model_test_metrics.csv", index=False)
    print("Saved: outputs/tuned_best_model_test_report.txt")
    print("Saved: outputs/tuned_best_model_test_metrics.csv")


if __name__ == "__main__":
    main()
