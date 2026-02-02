# Model Training Pipeline - FYP2 Phase A

## Overview
This module implements the **Predictive Model Development** component of FYP2, training and evaluating multiple machine learning models for employee attrition prediction.

## Architecture

### Modules
```
modules/
├── model_training.py      # Core training logic for 4 baseline models
├── model_evaluation.py    # Comprehensive evaluation and visualization
└── model_selection.py     # Model selection with justification
```

### Models Trained
1. **Logistic Regression** - Linear baseline, highly interpretable
2. **Random Forest** - Ensemble method, robust to outliers
3. **XGBoost** - Gradient boosting, state-of-the-art performance
4. **LightGBM** - Fast gradient boosting, memory efficient

## Data Flow
```
train_data.csv (SMOTE-balanced) → Model Training → Trained Models (.pkl)
val_data.csv (original) → Model Evaluation → Metrics & Visualizations
                                           → Model Selection → Best Model
```

## Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Complete Pipeline
```bash
python train_models.py
```

### 3. Outputs Generated

**Models** (saved in `models/`):
- `Logistic_Regression.pkl`
- `Random_Forest.pkl`
- `XGBoost.pkl`
- `LightGBM.pkl`
- `best_model.pkl` (selected production model)
- `training_metadata.json`
- `selection_metadata.json`
- `model_selection_justification.txt`

**Visualizations** (saved in `outputs/`):
- `20_model_comparison.png` - Bar charts comparing all metrics
- `21_confusion_matrices.png` - Confusion matrices for all models
- `22_roc_curves.png` - ROC curves comparison

**Reports**:
- `outputs/model_evaluation_results.csv` - Detailed metrics table

## Evaluation Metrics

### Primary Metrics (Priority Order)
1. **Recall** (Primary) - Catch at-risk employees
2. **F1-Score** (Secondary) - Balance precision and recall
3. **ROC-AUC** - Threshold-independent performance
4. **Precision** - Avoid false alarms
5. **Accuracy** - Overall correctness

### Rationale
Recall is prioritized because **missing an at-risk employee is costlier than a false positive** in HR decision-making context.

## Model Selection Logic
```python
# Selection criteria
1. Highest Recall score (primary)
2. Highest F1-Score (tiebreaker)
3. Strong ROC-AUC (validation)
```

## Reproducibility
- All models use `random_state=42`
- Training metadata logged with timestamps
- Data partitioning preserved from FYP1

## Integration with FYP1
- Uses `train_data.csv` (SMOTE-balanced from Module 6)
- Uses `val_data.csv` (original distribution from Module 7)
- Maintains data integrity (no leakage)
- Test set (`test_data.csv`) reserved for final evaluation

## Future Extensions (FYP2 Phases B-F)
- **Phase B**: SHAP explainability integration
- **Phase C**: Risk scoring and categorization
- **Phase D**: Recommendation engine
- **Phase E**: What-if simulation
- **Phase F**: Conversational AI interface

## AWS Compatibility
All code is designed for future AWS deployment:
- Joblib serialization (S3-compatible)
- Configurable paths (local/S3)
- SageMaker-ready architecture

## Academic Justification
This implementation follows:
- Industry best practices for ML pipelines
- Reproducible research standards
- Modular, testable design
- Clear separation of concerns
- Comprehensive documentation

## Notes
- Models trained with default hyperparameters (Phase A baseline)
- Hyperparameter tuning can be added in future iterations
- Architecture supports easy addition of new models (e.g., LLMs)
