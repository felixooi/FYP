"""
Module 10: Model Selection
Selects the best model based on evaluation metrics with justification.
"""

import pandas as pd
import logging
import json
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def _metrics_at_threshold(y_true, y_prob, threshold=0.5):
    """Compute metrics at a given threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    pr_auc = average_precision_score(y_true, y_prob)
    return {
        'Precision': precision,
        'Recall': recall,
        'F1_Score': f1,
        'PR_AUC': pr_auc,
        'Threshold': threshold
    }


def find_best_threshold(model, X_val, y_val, precision_min=0.50, grid_size=101):
    """
    Search thresholds to maximize recall subject to precision >= precision_min.
    Tie-break by PR-AUC then F1.
    """
    y_prob = model.predict_proba(X_val)[:, 1]
    thresholds = np.linspace(0, 1, grid_size)

    best = None
    for t in thresholds:
        m = _metrics_at_threshold(y_val, y_prob, t)
        if m['Precision'] + 1e-12 < precision_min:
            continue
        if best is None:
            best = m
        else:
            if m['Recall'] > best['Recall'] + 1e-12:
                best = m
            elif abs(m['Recall'] - best['Recall']) < 1e-12:
                if m['PR_AUC'] > best['PR_AUC'] + 1e-12:
                    best = m
                elif abs(m['PR_AUC'] - best['PR_AUC']) < 1e-12 and m['F1_Score'] > best['F1_Score']:
                    best = m
    if best is None:
        # If no threshold meets precision constraint, fall back to default 0.5 metrics
        best = _metrics_at_threshold(y_val, y_prob, 0.5)
        best['note'] = 'Precision constraint not met; fallback to threshold=0.5'
    best['y_prob'] = y_prob
    return best


def select_best_model(results_df, models, X_val, y_val,
                      precision_min=0.50, primary_metric='Recall', secondary_metric='PR_AUC'):
    """
    Select best model based on primary and secondary metrics.
    
    Args:
        results_df: DataFrame with evaluation results
        models: Dictionary of trained models
        primary_metric: Primary metric for selection (default: Recall)
        secondary_metric: Tiebreaker metric (default: F1_Score)
    
    Returns:
        best_model_name: Name of selected model
        best_model: The selected model object
        justification: Text justification for selection
    """
    logging.info("==== MODEL SELECTION START ====")
    
    tuned = []
    for name, model in models.items():
        m = find_best_threshold(model, X_val, y_val, precision_min=precision_min)
        tuned.append({
            'Model': name,
            'Precision': m['Precision'],
            'Recall': m['Recall'],
            'F1_Score': m['F1_Score'],
            'PR_AUC': m['PR_AUC'],
            'Threshold': m['Threshold'],
            'Note': m.get('note', '')
        })
    tuned_df = pd.DataFrame(tuned).sort_values(
        by=[primary_metric, secondary_metric, 'F1_Score'],
        ascending=False
    ).reset_index(drop=True)

    best_model_name = tuned_df.iloc[0]['Model']
    best_model = models[best_model_name]
    best_metrics = tuned_df.iloc[0].to_dict()
    
    # Generate justification
    justification = f"""
Model Selection Justification:
==============================
Selected Model: {best_model_name}

Selection Criteria:
- Primary: {primary_metric} (maximize)
- Constraint: Precision >= {precision_min:.2f}
- Secondary (tiebreak): {secondary_metric}, then F1
- Selected threshold: {best_metrics['Threshold']:.3f}

Performance Summary:
- Precision: {best_metrics['Precision']:.4f}
- Recall: {best_metrics['Recall']:.4f}
- F1-Score: {best_metrics['F1_Score']:.4f}
- PR-AUC: {best_metrics['PR_AUC']:.4f}
- Threshold: {best_metrics['Threshold']:.3f}

Rationale:
We maximize recall under a minimum precision constraint to control false positives.
{best_model_name} delivers the best recall while satisfying the constraint, with strong PR-AUC and F1 support.
"""
    
    logging.info(f"Selected Model: {best_model_name}")
    logging.info(f"{primary_metric}: {best_metrics[primary_metric]:.4f}, {secondary_metric}: {best_metrics[secondary_metric]:.4f}")
    logging.info("==== MODEL SELECTION COMPLETE ====")
    
    return best_model_name, best_model, justification, tuned_df

def compare_top_models(results_df, top_n=3):
    """Compare top N models and display detailed comparison."""
    top_models = results_df.head(top_n)
    
    logging.info(f"\nTop {top_n} Models Comparison:")
    logging.info("="*80)
    
    highlight_cols = [c for c in ['Precision', 'Recall', 'F1_Score', 'PR_AUC'] if c in top_models.columns]
    try:
        from IPython.display import display
        if highlight_cols:
            display(top_models.style.highlight_max(axis=0, subset=highlight_cols))
        else:
            display(top_models)
    except ImportError:
        print(top_models.to_string())
    
    return top_models

def save_selection_results(best_model_name, best_model, justification,
                           tuned_df, output_dir='models', precision_min=0.50):
    """Save model selection results and metadata."""
    import joblib
    import os
    from datetime import datetime
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save best model separately
    best_model_path = f"{output_dir}/best_model.pkl"
    joblib.dump(best_model, best_model_path)
    logging.info(f"Best model saved → {best_model_path}")
    
    # Save justification
    with open(f"{output_dir}/model_selection_justification.txt", 'w') as f:
        f.write(justification)
    logging.info("Selection justification saved.")
    
    # Save selection metadata
    best_metrics = tuned_df[tuned_df['Model'] == best_model_name].iloc[0].to_dict()
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'selected_model': best_model_name,
        'selection_criteria': f'Recall (primary), Precision>={precision_min:.2f} constraint, PR_AUC secondary',
        'threshold': best_metrics.get('Threshold', 0.5),
        'metrics': {k: float(v) if isinstance(v, (int, float)) else v 
                   for k, v in best_metrics.items()}
    }
    
    with open(f"{output_dir}/selection_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=4)
    logging.info("Selection metadata saved.")

def perform_model_selection(results_df, models, X_val, y_val,
                            primary_metric='Recall',
                            secondary_metric='PR_AUC',
                            precision_min=0.50,
                            save_results=True,
                            output_dir='models'):
    """
    Complete model selection pipeline.
    
    Returns:
        Dictionary containing best model info and comparison results
    """
    best_model_name, best_model, justification, tuned_df = select_best_model(
        results_df, models, X_val, y_val, precision_min, primary_metric, secondary_metric
    )

    top_models = compare_top_models(tuned_df, top_n=min(3, len(tuned_df)))

    if save_results:
        save_selection_results(best_model_name, best_model, justification,
                              tuned_df, output_dir, precision_min=precision_min)

    print(justification)

    return {
        'best_model_name': best_model_name,
        'best_model': best_model,
        'justification': justification,
        'top_models': top_models,
        'tuned_df': tuned_df,
        'threshold': float(tuned_df[tuned_df['Model'] == best_model_name].iloc[0]['Threshold'])
    }
