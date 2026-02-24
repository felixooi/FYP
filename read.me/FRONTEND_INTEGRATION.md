# Frontend Integration Handoff

## Scope
This project is ready for frontend integration through an inference-only backend flow.
The frontend should upload a dataset and trigger server-side inference, then render returned outputs.

## Required Runtime Artifacts
- `models/best_model_tuned.pkl`
- `models/tuning_metadata.json`
- `models/feature_engineering_params.json`
- `data/selected_features.json`
- `data/scaler.pkl`

## Inference Entrypoint
- Script: `end_to_end_pipeline.py`
- Function: `run_inference_pipeline(input_file, output_dir="pipeline_output")`

## Backend Input Contract
- Input type: CSV file path on server side.
- Supported formats:
  - Raw HR dataset with expected source columns.
  - Already preprocessed dataset containing selected model features.

## Backend Output Contract
After `run_inference_pipeline(...)`, backend writes:
- `pipeline_output/inference_predictions.csv`
  - Includes original columns plus:
    - `Attrition_Probability`
    - `Predicted_Resigned` (0/1)
    - `Risk_Level` (`LOW`/`MEDIUM`/`HIGH`)
- `pipeline_output/inference_summary.json`
  - Includes:
    - `input_rows`
    - `threshold`
    - `predicted_resigned_count`
    - `predicted_stayed_count`
    - `mean_attrition_probability`
    - `used_feature_count`
    - `feature_engineering_param_source`
    - `predictions_path`

## Integration Notes
- Do not run training scripts from frontend requests.
- Call inference server-side only.
- Keep file paths configurable in backend environment variables.
- Validate upload schema and return readable errors to UI.

## Quick Backend Example
```python
from end_to_end_pipeline import run_inference_pipeline

summary = run_inference_pipeline(
    input_file="uploads/new_dataset.csv",
    output_dir="pipeline_output"
)
print(summary)
```

## Validation Checklist
- `threshold` in `inference_summary.json` matches `models/tuning_metadata.json`.
- `feature_engineering_param_source` should be `saved_training_params` for deterministic preprocessing.
- Row count in `inference_predictions.csv` equals uploaded dataset row count.
