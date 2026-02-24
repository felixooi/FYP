# Phase B: Explainable AI - Technical Documentation

## Overview

Phase B implements a two-level explainability framework using SHAP (SHapley Additive exPlanations) to provide interpretable explanations for employee attrition predictions.

**Objectives**:
- Global explainability: Identify features driving attrition risk across all employees
- Local explainability: Explain individual employee predictions
- Natural language generation: Translate technical outputs into HR-friendly insights

---

## Theoretical Foundation

### Shapley Values (Shapley, 1953)
- **Origin**: Cooperative game theory
- **Purpose**: Fair attribution of contribution among players
- **Axioms**:
  - **Efficiency**: Sum of contributions equals total payoff
  - **Symmetry**: Equal contributors receive equal credit
  - **Dummy**: Zero-impact players receive zero value
  - **Additivity**: Consistent across different games

### SHAP Framework (Lundberg & Lee, NeurIPS 2017)
- **Unified approach**: Connects multiple interpretability methods (LIME, DeepLIFT, etc.)
- **Model-agnostic**: Works with any machine learning model
- **Mathematically rigorous**: Based on Shapley values from game theory
- **Efficiency axiom**: `sum(SHAP values) = prediction - base_value`

### TreeExplainer (Lundberg et al., Nature Machine Intelligence 2020)
- **Exact computation**: No sampling approximation required
- **Polynomial time**: Efficient for tree-based models (XGBoost, LightGBM, Random Forest)
- **Fast**: Optimized algorithm for tree ensembles

---

## Module Architecture

### 1. `modules/explainability.py` (Core)

**Purpose**: SHAP explainer initialization, computation, and caching

**Key Functions**:
- `initialize_shap_explainer(model, X_background, model_type)`: Initialize TreeExplainer or LinearExplainer
- `compute_shap_values(explainer, X_data)`: Compute exact SHAP values
- `save_explainer(explainer, filepath)`: Cache explainer for reuse
- `load_explainer(filepath)`: Load cached explainer
- `validate_features(X_data, expected_features)`: Feature alignment checks
- `verify_efficiency_axiom(shap_values, predictions, base_value)`: Validate SHAP axiom

**Design Decisions**:
- Background dataset: 100 samples (SHAP standard practice)
- Model type detection: 'tree' for tree-based models, 'linear' for logistic regression
- Binary classification handling: Extract positive class SHAP values

---

### 2. `modules/explanation_analysis.py` (Global + Local)

**Purpose**: Global and local explainability logic with visualizations

#### Global Explainability Functions:
- `compute_global_feature_importance(shap_values, feature_names)`: Mean absolute SHAP values
- `plot_shap_summary(shap_values, X_data, feature_names, save_path)`: Beeswarm plot
- `plot_feature_importance_bar(importance_df, top_n, save_path)`: Bar chart (top 10)
- `plot_dependence(shap_values, X_data, feature_names, feature_name, save_path)`: Dependence plot

#### Local Explainability Functions:
- `extract_local_explanation(shap_values, base_value, feature_values, feature_names, instance_idx)`: Extract top contributors
- `plot_waterfall(shap_values, base_value, feature_values, feature_names, instance_idx, save_path)`: Waterfall plot
- `create_contribution_table(shap_values, feature_values, feature_names, instance_idx)`: Tabular breakdown

**Design Decisions**:
- Global importance: `np.abs(shap_values).mean(axis=0)` (mean absolute SHAP)
- Top N features: Limited to 10 for bar plot (user requirement)
- Dependence plots: Optional, max 2-3 key features
- Waterfall plots: Individual prediction breakdown

---

### 3. `modules/explanation_generator.py` (Natural Language)

**Purpose**: Convert SHAP values into human-readable explanations

**Key Functions**:
- `generate_explanation(prediction_proba, shap_values, feature_values, feature_names, threshold, base_value)`: Full NL explanation
- `format_risk_level(prediction_proba, threshold)`: HIGH/MEDIUM/LOW risk categorization
- `describe_top_contributors(shap_values, feature_values, feature_names, top_n)`: Risk-increasing factors
- `describe_protective_factors(shap_values, feature_values, feature_names, top_n)`: Risk-reducing factors
- `quantify_impact(shap_value, base_value)`: Convert to percentage points

**Design Decisions**:
- Risk levels:
  - HIGH: `prediction >= threshold + 0.2`
  - MEDIUM: `threshold <= prediction < threshold + 0.2`
  - LOW: `prediction < threshold`
- Template-based generation with dynamic values (no hardcoding)
- Top 3 risk-increasing factors, top 2 protective factors
- Impact quantification: SHAP value × 100 = percentage points

**Example Output**:
```
This employee is classified as HIGH RISK of attrition (probability: 88.3%).

The primary contributing factor is employee satisfaction score (2.10), which 
increases their attrition risk by 25.3 percentage points. Additionally, 
overtime hours (22.00) and monthly salary (3200.00) further elevate the risk 
by 18.7 and 14.5 percentage points respectively.

Positive factors such as performance score (4.20) and age (35.00) help reduce 
the overall risk by 5.1 and 3.2 percentage points respectively, but are 
insufficient to offset current risk factors.
```

---

## Orchestration Script: `explain_models.py`

### Workflow

1. **Load Artifacts**:
   - Best model: `models/best_model_tuned.pkl`
   - Threshold: `models/tuning_metadata.json`
   - Validation data: `data/val_data.csv`
   - Feature names: `data/selected_features.json`

2. **Global Explainability**:
   - Initialize SHAP explainer (100-sample background)
   - Compute SHAP values for full validation set (15,000 samples)
   - Verify efficiency axiom
   - Compute global feature importance
   - Generate visualizations:
     - SHAP summary plot (beeswarm)
     - Feature importance bar plot (top 10)
     - Dependence plots (top 3 features)
   - Export `feature_importance.json`

3. **Local Explainability**:
   - Select sample employees:
     - High-risk: Top 10 (highest predicted probability)
     - Low-risk: Top 10 (lowest predicted probability)
     - Borderline: 10 near threshold
   - For each employee:
     - Extract local SHAP explanation
     - Generate waterfall plot
     - Create contribution table (CSV)
     - Generate natural language explanation
     - Export `employee_[idx]_explanation.json`
   - Export `explanations_summary.csv`

4. **Save Metadata**:
   - Timestamp, model name, threshold
   - Top 3 features
   - Output paths

### Command-Line Interface

```bash
# Full pipeline (global + local)
python explain_models.py

# Global explanations only
python explain_models.py --global-only

# Explain specific employee
python explain_models.py --employee-idx 42
```

---

## Output Structure

```
outputs/
└── explainability/
    ├── metadata.json                          # Pipeline metadata
    ├── global/
    │   ├── feature_importance.json            # Top 10 + all features
    │   ├── shap_summary.png                   # Beeswarm plot
    │   ├── feature_importance_bar.png         # Bar chart (top 10)
    │   ├── dependence_1_[feature].png         # Dependence plot #1
    │   ├── dependence_2_[feature].png         # Dependence plot #2
    │   ├── dependence_3_[feature].png         # Dependence plot #3
    │   └── shap_values.pkl                    # Cached SHAP values
    ├── local/
    │   ├── explanations_summary.csv           # Summary of all explanations
    │   ├── employee_[idx]_waterfall.png       # Waterfall plot
    │   ├── employee_[idx]_contributions.csv   # Contribution table
    │   └── employee_[idx]_explanation.json    # Full explanation
    └── explainers/
        └── shap_explainer.pkl                 # Cached SHAP explainer

models/
└── explainers/
    └── shap_explainer.pkl                     # Cached explainer (alternative location)
```

---

## Output Specifications

### Global Explanation JSON (`feature_importance.json`)

```json
{
  "top_features": [
    {
      "feature": "Employee_Satisfaction_Score",
      "mean_abs_shap": 0.228,
      "rank": 1
    },
    {
      "feature": "Burnout_Risk",
      "mean_abs_shap": 0.156,
      "rank": 2
    }
  ],
  "all_features": [...]
}
```

### Local Explanation JSON (`employee_[idx]_explanation.json`)

```json
{
  "employee_index": 42,
  "prediction_probability": 0.883,
  "actual_resigned": 1,
  "risk_level": "HIGH",
  "base_value": 0.15,
  "threshold": 0.55,
  "top_risk_increasing_factors": [
    {
      "feature": "Employee_Satisfaction_Score",
      "feature_value": 2.1,
      "shap_value": 0.253,
      "impact_percentage_points": 25.3
    }
  ],
  "top_risk_reducing_factors": [
    {
      "feature": "Performance_Score",
      "feature_value": 4.2,
      "shap_value": -0.051,
      "impact_percentage_points": -5.1
    }
  ],
  "explanation_text": "This employee is classified as HIGH RISK...",
  "visualizations": {
    "waterfall_plot": "outputs/explainability/local/employee_42_waterfall.png",
    "contributions_table": "outputs/explainability/local/employee_42_contributions.csv"
  }
}
```

---

## Performance Characteristics

### Computation Time (Validation Set = 15,000 samples)

| Operation | Time | Notes |
|-----------|------|-------|
| SHAP Explainer Initialization | 30-60s | One-time, cached |
| SHAP Values Computation | 30-60s | Logistic Regression (fast) |
| Global Visualizations | 10-15s | 3 plots + 3 dependence plots |
| Local Explanation (per employee) | 1-2s | From cached SHAP values |
| **Total Pipeline** | **2-3 min** | Full global + 30 local explanations |

### Optimization Strategies

1. **Caching**: Save explainer and SHAP values to avoid recomputation
2. **Background Sampling**: Use 100 samples (SHAP standard)
3. **Batch Processing**: Compute SHAP values once for entire validation set
4. **Lazy Loading**: Generate local explanations on-demand

---

## Integration with Later Phases

### Phase C: Risk Scoring & Categorization
- **Input**: SHAP values (`shap_values.pkl`)
- **Use Case**: Risk score = f(prediction_proba, top_shap_contributors)
- **Integration**: Load SHAP values to identify high-impact features for risk categorization

### Phase D: Recommendation Engine
- **Input**: Top risk-increasing factors from local explanations
- **Use Case**: Recommend interventions targeting high-SHAP features
- **Integration**: Parse `employee_[idx]_explanation.json` to extract top contributors

### Phase E: What-If Simulation
- **Input**: SHAP explainer + feature perturbation
- **Use Case**: Recompute SHAP values after feature modification
- **Integration**: Load cached explainer, modify features, recompute SHAP

### Phase F: Conversational AI
- **Input**: Natural language explanations from `explanation_generator.py`
- **Use Case**: Chatbot responses using pre-generated explanations
- **Integration**: Load `explanation_text` from JSON for conversational responses

---

## Validation & Testing

### Efficiency Axiom Verification

```python
# Verify: sum(SHAP values) ≈ prediction - base_value
shap_sum = shap_values.sum(axis=1)
expected = predictions - base_value
diff = np.abs(shap_sum - expected)
assert np.all(diff < 1e-3), "Efficiency axiom violated"
```

### Feature Consistency Check

```python
# Ensure features match between model and data
validate_features(X_val, expected_features)
```

### Determinism Check

```python
# Same input → same SHAP values
shap_values_1 = compute_shap_values(explainer, X_val)
shap_values_2 = compute_shap_values(explainer, X_val)
assert np.allclose(shap_values_1, shap_values_2), "Non-deterministic SHAP values"
```

---

## Academic Rigor

### References

1. **Shapley, L. S. (1953)**. "A value for n-person games." *Contributions to the Theory of Games*, 2(28), 307-317.

2. **Lundberg, S. M., & Lee, S. I. (2017)**. "A unified approach to interpreting model predictions." *Advances in Neural Information Processing Systems* (NeurIPS), 30.

3. **Lundberg, S. M., Erion, G., Chen, H., DeGrave, A., Prutkin, J. M., Nair, B., ... & Lee, S. I. (2020)**. "From local explanations to global understanding with explainable AI for trees." *Nature Machine Intelligence*, 2(1), 56-67.

### Viva Defense Points

1. **Why SHAP?**
   - Mathematically rigorous (Shapley values)
   - Model-agnostic (works with all models)
   - Industry standard (widely adopted)
   - Academic credibility (NeurIPS, Nature MI)

2. **Why TreeExplainer?**
   - Exact Shapley values (no approximation)
   - Polynomial time complexity
   - Optimized for tree ensembles

3. **Why 100-sample background?**
   - SHAP standard practice
   - Balance between accuracy and speed
   - Sufficient for stable base value estimation

4. **Why mean absolute SHAP for importance?**
   - Captures magnitude of impact (positive or negative)
   - Standard metric in SHAP literature
   - Interpretable for stakeholders

---

## Limitations & Caveats

1. **Correlation vs Causation**: SHAP values show feature contributions, not causal relationships
2. **Model Dependence**: Explanations reflect model behavior, not ground truth
3. **Background Dataset**: Base value depends on background sample selection
4. **Computational Cost**: SHAP computation scales with dataset size
5. **Interpretation Complexity**: Stakeholders need training to interpret SHAP plots

---

## Success Criteria

✅ **Functional**:
- Generate global feature importance with exact SHAP values
- Generate local explanations for individual employees
- Produce HR-friendly natural language summaries
- Export JSON for frontend and later phases

✅ **Academic**:
- Reference Shapley values and SHAP framework in code
- Verify efficiency axiom (sum of SHAP = prediction - base)
- Deterministic and reproducible
- Easy to justify in viva

✅ **Integration**:
- Save SHAP values for Phase C (risk scoring)
- Save explainer for Phase E (what-if simulation)
- Save explanations for Phase F (conversational AI)

✅ **Quality**:
- Minimal code (no over-engineering)
- Industry-standard SHAP usage
- Publication-quality visualizations
- Clear, actionable natural language

---

## Usage Examples

### Full Pipeline

```bash
python explain_models.py
```

**Output**:
- Global explanations (feature importance, summary plot, bar plot, dependence plots)
- Local explanations for 30 employees (10 high-risk, 10 low-risk, 10 borderline)
- Natural language summaries
- All outputs in `outputs/explainability/`

### Global Only

```bash
python explain_models.py --global-only
```

**Output**:
- Global explanations only
- Skips local explanations (faster)

### Specific Employee

```bash
python explain_models.py --employee-idx 42
```

**Output**:
- Global explanations
- Local explanation for employee at index 42 only
- Prints natural language explanation to console

---

## Troubleshooting

### Issue: SHAP values don't sum to prediction difference

**Cause**: Numerical precision or incorrect base value

**Solution**: Check `verify_efficiency_axiom()` output, adjust tolerance if needed

### Issue: Feature mismatch error

**Cause**: Input data features don't match expected features

**Solution**: Ensure `X_val` columns match `selected_features.json`

### Issue: Slow SHAP computation

**Cause**: Large dataset or complex model

**Solution**: Use cached SHAP values (`shap_values.pkl`), reduce background sample size

### Issue: Waterfall plot error

**Cause**: SHAP version incompatibility

**Solution**: Ensure `shap>=0.42.0` installed

---

## Next Steps

**Phase C: Risk Scoring & Categorization**
- Use SHAP values to enhance risk scoring
- Categorize employees by risk level and contributing factors
- Integrate with Phase B outputs

**Phase D: Recommendation Engine**
- Target high-SHAP features for interventions
- Generate actionable recommendations based on explanations

**Phase E: What-If Simulation**
- Use cached explainer for feature perturbation
- Recompute SHAP values after modifications

**Phase F: Conversational AI**
- Integrate natural language explanations into chatbot
- Enable interactive explanation queries

---

## Conclusion

Phase B successfully implements a two-level explainability framework using SHAP, providing:
- **Global insights**: Feature importance rankings and impact patterns
- **Local explanations**: Individual prediction breakdowns with waterfall plots
- **Natural language**: HR-friendly summaries for stakeholder communication

The implementation is academically rigorous, deterministic, and ready for integration with later phases.
