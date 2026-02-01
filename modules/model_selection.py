"""
Module 10: Model Selection
Selects the best model based on evaluation metrics with justification.
"""

import pandas as pd
import logging
import json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def select_best_model(results_df, models, primary_metric='Recall', secondary_metric='F1_Score'):
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
    
    # Sort by primary metric, then secondary
    sorted_df = results_df.sort_values(
        by=[primary_metric, secondary_metric], 
        ascending=False
    ).reset_index(drop=True)
    
    best_model_name = sorted_df.iloc[0]['Model']
    best_model = models[best_model_name]
    best_metrics = sorted_df.iloc[0].to_dict()
    
    # Generate justification
    justification = f"""
Model Selection Justification:
==============================
Selected Model: {best_model_name}

Selection Criteria:
- Primary Metric: {primary_metric} = {best_metrics[primary_metric]:.4f}
- Secondary Metric: {secondary_metric} = {best_metrics[secondary_metric]:.4f}

Performance Summary:
- Accuracy: {best_metrics['Accuracy']:.4f}
- Precision: {best_metrics['Precision']:.4f}
- Recall: {best_metrics['Recall']:.4f}
- F1-Score: {best_metrics['F1_Score']:.4f}
- ROC-AUC: {best_metrics['ROC_AUC']:.4f}

Rationale:
{best_model_name} achieved the highest {primary_metric} score, which is critical 
for attrition prediction as missing at-risk employees is costlier than false positives.
The model also demonstrates strong performance across other metrics, making it 
suitable for production deployment.
"""
    
    logging.info(f"Selected Model: {best_model_name}")
    logging.info(f"{primary_metric}: {best_metrics[primary_metric]:.4f}, {secondary_metric}: {best_metrics[secondary_metric]:.4f}")
    logging.info("==== MODEL SELECTION COMPLETE ====")
    
    return best_model_name, best_model, justification

def compare_top_models(results_df, top_n=3):
    """Compare top N models and display detailed comparison."""
    top_models = results_df.head(top_n)
    
    logging.info(f"\nTop {top_n} Models Comparison:")
    logging.info("="*80)
    
    try:
        from IPython.display import display
        display(top_models.style.highlight_max(axis=0, subset=['Accuracy', 'Precision', 'Recall', 'F1_Score', 'ROC_AUC']))
    except ImportError:
        print(top_models.to_string())
    
    return top_models

def save_selection_results(best_model_name, best_model, justification, 
                           results_df, output_dir='models'):
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
    best_metrics = results_df[results_df['Model'] == best_model_name].iloc[0].to_dict()
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'selected_model': best_model_name,
        'selection_criteria': 'Recall (primary), F1_Score (secondary)',
        'metrics': {k: float(v) if isinstance(v, (int, float)) else v 
                   for k, v in best_metrics.items()}
    }
    
    with open(f"{output_dir}/selection_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=4)
    logging.info("Selection metadata saved.")

def perform_model_selection(results_df, models, primary_metric='Recall', 
                            secondary_metric='F1_Score', save_results=True, 
                            output_dir='models'):
    """
    Complete model selection pipeline.
    
    Returns:
        Dictionary containing best model info and comparison results
    """
    best_model_name, best_model, justification = select_best_model(
        results_df, models, primary_metric, secondary_metric
    )
    
    top_models = compare_top_models(results_df, top_n=min(3, len(results_df)))
    
    if save_results:
        save_selection_results(best_model_name, best_model, justification, 
                              results_df, output_dir)
    
    print(justification)
    
    return {
        'best_model_name': best_model_name,
        'best_model': best_model,
        'justification': justification,
        'top_models': top_models
    }
