# Final Model Card

- Generated: 2026-02-24T13:59:46.323364
- Model: Logistic_Regression
- Model file: `models/best_model_tuned.pkl`
- Threshold: 0.280
- Selection rule: maximize recall with precision >= 0.50, then PR-AUC and F1

## Data Summary
- Train samples: 116986
- Validation samples: 15000
- Test samples: 20000
- Features used: 20
- Class balance (train, positive rate): 0.5000
- Class balance (validation, positive rate): 0.1001
- Class balance (test, positive rate): 0.1001

## Validation Performance (selected threshold)
- Precision: 0.5016
- Recall: 0.9687
- F1: 0.6609
- PR-AUC: 0.9332
- Brier score: 0.0399

## Test Performance (selected threshold)
- Precision: 0.4999
- Recall: 0.9625
- F1: 0.6580
- PR-AUC: 0.9312

## Artifact Index
- `outputs/final_model_test_report.txt`
- `outputs/final_model_test_metrics.csv`
- `outputs/final_model_validation_calibration.png`
- `models/tuning_metadata.json`
- `models/threshold_rationale.txt`
