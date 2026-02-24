# Phase B: Explainable AI - Quick Start Guide

## 🚀 Ready to Execute

Phase B implementation is complete. Follow these steps to generate explainability outputs.

---

## ✅ Pre-Execution Checklist

Verify these files exist:

### Required Artifacts (from Phase A)
- ✅ `models/best_model_tuned.pkl` - Champion model
- ✅ `models/tuning_metadata.json` - Threshold and model name
- ✅ `data/val_data.csv` - Validation dataset (15,000 samples)
- ✅ `data/selected_features.json` - Feature names (20 features)

### New Modules (Phase B)
- ✅ `modules/explainability.py` - Core SHAP functions
- ✅ `modules/explanation_analysis.py` - Global + local analysis
- ✅ `modules/explanation_generator.py` - Natural language generation

### Orchestration Script
- ✅ `explain_models.py` - Main pipeline

---

## 🎯 Execution Options

### Option 1: Full Pipeline (Recommended)
```bash
python explain_models.py
```

**What it does**:
- Generates global explanations (feature importance, SHAP plots)
- Generates local explanations for 30 sample employees
- Creates natural language summaries
- Saves all outputs to `outputs/explainability/`

**Expected runtime**: 2-3 minutes (first run), 1-2 minutes (subsequent runs with cache)

**Expected outputs**: ~100 files
- 7 global files (JSON, images, cached SHAP values)
- 91 local files (30 employees × 3 files + 1 summary)
- 2 cached artifacts (explainer, SHAP values)

---

### Option 2: Global Explanations Only
```bash
python explain_models.py --global-only
```

**What it does**:
- Generates only global explanations
- Skips local explanations (faster)

**Expected runtime**: 1-2 minutes

**Expected outputs**: 7 files in `outputs/explainability/global/`

---

### Option 3: Explain Specific Employee
```bash
python explain_models.py --employee-idx 42
```

**What it does**:
- Generates global explanations
- Generates local explanation for employee at index 42 only
- Prints natural language explanation to console

**Expected runtime**: 1-2 minutes

**Expected outputs**: Global files + 3 files for employee 42

---

## 📊 Expected Console Output

```
================================================================================
PHASE B: EXPLAINABLE AI
================================================================================
Timestamp: 2024-XX-XXTXX:XX:XX

Loading artifacts...
✓ Model: Logistic_Regression
✓ Threshold: 0.55
✓ Validation samples: 15000
✓ Features: 20

================================================================================
GLOBAL EXPLAINABILITY
================================================================================
Initializing SHAP explainer (this may take 30-60 seconds)...
SHAP explainer initialized successfully.
Computing SHAP values (this may take 30-60 seconds)...
SHAP values computed successfully.
✓ Efficiency axiom verified (max diff: 0.000123)

Top 10 Features by Global Importance:
   rank                          feature  mean_abs_shap
      1  Employee_Satisfaction_Score       0.228
      2                    Burnout_Risk       0.156
      3               Workload_Intensity       0.143
      ...

✓ Feature importance saved to outputs/explainability/global/feature_importance.json

Generating visualizations...
✓ SHAP summary plot saved to outputs/explainability/global/shap_summary.png
✓ Feature importance bar plot saved to outputs/explainability/global/feature_importance_bar.png
✓ Dependence plot saved to outputs/explainability/global/dependence_1_Employee_Satisfaction_Score.png
✓ Dependence plot saved to outputs/explainability/global/dependence_2_Burnout_Risk.png
✓ Dependence plot saved to outputs/explainability/global/dependence_3_Workload_Intensity.png
✓ Global explanations complete

================================================================================
LOCAL EXPLAINABILITY
================================================================================
Selected 30 sample employees (10 high-risk, 10 low-risk, 10 borderline)

✓ Generated explanations for 30 employees
✓ Local explanations complete

================================================================================
PHASE B COMPLETE
================================================================================
✓ Global explanations: outputs/explainability/global/
✓ Local explanations: outputs/explainability/local/
✓ Cached explainer: outputs/explainability/explainers/
✓ Metadata: outputs/explainability/metadata.json
```

---

## 📁 Output Directory Structure

After execution, you'll have:

```
outputs/
└── explainability/
    ├── metadata.json                                    # Pipeline metadata
    │
    ├── global/                                          # Global explanations
    │   ├── feature_importance.json                      # Top 10 + all features
    │   ├── shap_summary.png                             # Beeswarm plot
    │   ├── feature_importance_bar.png                   # Bar chart (top 10)
    │   ├── dependence_1_Employee_Satisfaction_Score.png # Dependence plot
    │   ├── dependence_2_Burnout_Risk.png                # Dependence plot
    │   ├── dependence_3_Workload_Intensity.png          # Dependence plot
    │   └── shap_values.pkl                              # Cached SHAP values
    │
    ├── local/                                           # Local explanations
    │   ├── explanations_summary.csv                     # Summary of all 30
    │   ├── employee_[idx]_waterfall.png                 # Waterfall plot (×30)
    │   ├── employee_[idx]_contributions.csv             # Contribution table (×30)
    │   └── employee_[idx]_explanation.json              # Full explanation (×30)
    │
    └── explainers/                                      # Cached artifacts
        └── shap_explainer.pkl                           # SHAP explainer
```

---

## 🔍 Verify Outputs

### 1. Check Global Feature Importance
```bash
type outputs\explainability\global\feature_importance.json
```

**Expected**: JSON with top 10 features ranked by mean absolute SHAP value

### 2. View SHAP Summary Plot
Open: `outputs/explainability/global/shap_summary.png`

**Expected**: Beeswarm plot showing feature impacts across all employees

### 3. View Feature Importance Bar Plot
Open: `outputs/explainability/global/feature_importance_bar.png`

**Expected**: Bar chart of top 10 features

### 4. Check Local Explanation
```bash
type outputs\explainability\local\employee_[idx]_explanation.json
```

**Expected**: JSON with prediction, risk level, top contributors, natural language explanation

### 5. View Waterfall Plot
Open: `outputs/explainability/local/employee_[idx]_waterfall.png`

**Expected**: Waterfall plot showing feature contributions for individual employee

---

## 🐛 Troubleshooting

### Issue: "FileNotFoundError: models/best_model_tuned.pkl"
**Solution**: Run Phase A first (`python train_models.py` then `python tune_models.py`)

### Issue: "SHAP computation taking too long"
**Solution**: Normal for first run (30-60s). Subsequent runs use cached values.

### Issue: "Feature mismatch error"
**Solution**: Ensure `data/val_data.csv` has same features as `data/selected_features.json`

### Issue: "Memory error"
**Solution**: Reduce validation set size or use `--global-only` flag

---

## 📈 Performance Tips

1. **First Run**: Takes 2-3 minutes (computes and caches SHAP values)
2. **Subsequent Runs**: Takes 1-2 minutes (uses cached values)
3. **Speed Up**: Use `--global-only` to skip local explanations
4. **Cache Location**: `outputs/explainability/explainers/` and `outputs/explainability/global/`

---

## 🔗 Integration with Later Phases

### Phase C: Risk Scoring
**Input**: `outputs/explainability/global/shap_values.pkl`
**Use**: Enhance risk scoring with SHAP-based feature importance

### Phase D: Recommendations
**Input**: `outputs/explainability/local/employee_[idx]_explanation.json`
**Use**: Target interventions based on top risk-increasing factors

### Phase E: What-If Simulation
**Input**: `outputs/explainability/explainers/shap_explainer.pkl`
**Use**: Recompute SHAP values after feature perturbation

### Phase F: Conversational AI
**Input**: `explanation_text` from JSON files
**Use**: Natural language responses for chatbot

---

## ✅ Success Criteria

After execution, verify:

- ✅ `outputs/explainability/metadata.json` exists
- ✅ `outputs/explainability/global/feature_importance.json` has 10 features
- ✅ `outputs/explainability/global/shap_summary.png` is a valid image
- ✅ `outputs/explainability/global/feature_importance_bar.png` shows top 10 features
- ✅ `outputs/explainability/local/explanations_summary.csv` has 30 rows
- ✅ 30 waterfall plots in `outputs/explainability/local/`
- ✅ 30 JSON explanations in `outputs/explainability/local/`
- ✅ Cached explainer in `outputs/explainability/explainers/`

---

## 🎓 Academic Validation

### Verify Efficiency Axiom
Check console output for:
```
✓ Efficiency axiom verified (max diff: 0.000123)
```

This confirms: `sum(SHAP values) ≈ prediction - base_value`

### Verify Determinism
Run twice and compare outputs:
```bash
python explain_models.py
python explain_models.py
```

SHAP values should be identical (deterministic).

---

## 📞 Support

If you encounter issues:

1. Check `read.me/PHASE_B_EXPLAINABILITY.md` for detailed documentation
2. Review `read.me/PHASE_B_IMPLEMENTATION_SUMMARY.md` for implementation details
3. Verify all Phase A artifacts exist
4. Check Python dependencies: `pip install -r requirements.txt`

---

## 🚀 Ready to Execute!

Run the full pipeline:
```bash
python explain_models.py
```

Expected completion: 2-3 minutes

Good luck! 🎉
