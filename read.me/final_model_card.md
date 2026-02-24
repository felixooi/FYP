# Final Model Card

- Generated: 2026-02-24T17:48:26.254518
- Model: Logistic_Regression
- Model file: `models/best_model_tuned.pkl`
- Selected precision constraint: >= 0.70
- Threshold: 0.550
- Selection rule: validation sweep over precision constraints; maximize Val_F1, tie-break by Val_Recall then Val_PR_AUC

## Data Summary
- Train samples: 116986
- Validation samples: 15000
- Test samples: 20000
- Features used: 20
- Class balance (train, positive rate): 0.5000
- Class balance (validation, positive rate): 0.1001
- Class balance (test, positive rate): 0.1001

## Validation Performance (selected threshold)
- Precision: 0.7028
- Recall: 0.9340
- F1: 0.8021
- PR-AUC: 0.9332
- Brier score: 0.0399

## Test Performance (selected threshold)
- Precision: 0.7053
- Recall: 0.9291
- F1: 0.8019
- PR-AUC: 0.9312

## Artifact Index
- `outputs/final_model_test_report.txt`
- `outputs/final_model_test_metrics.csv`
- `outputs/precision_constraint_sweep.csv`
- `outputs/final_model_validation_calibration.png`
- `models/tuning_metadata.json`
- `models/threshold_rationale.txt`
