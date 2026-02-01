"""
Module 9: Model Evaluation
Evaluates trained models with comprehensive metrics and visualizations.
"""

import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def evaluate_model(model, X_val, y_val, model_name='Model'):
    """Evaluate a single model and return metrics dictionary."""
    logging.info(f"Evaluating {model_name}...")
    
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    
    metrics = {
        'Model': model_name,
        'Accuracy': accuracy_score(y_val, y_pred),
        'Precision': precision_score(y_val, y_pred),
        'Recall': recall_score(y_val, y_pred),
        'F1_Score': f1_score(y_val, y_pred),
        'ROC_AUC': roc_auc_score(y_val, y_pred_proba)
    }
    
    logging.info(f"{model_name} - Recall: {metrics['Recall']:.4f}, F1: {metrics['F1_Score']:.4f}, ROC-AUC: {metrics['ROC_AUC']:.4f}")
    return metrics

def evaluate_all_models(models, X_val, y_val):
    """Evaluate all models and return comparison DataFrame."""
    logging.info("==== MODEL EVALUATION START ====")
    
    results = []
    for name, model in models.items():
        metrics = evaluate_model(model, X_val, y_val, name)
        results.append(metrics)
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Recall', ascending=False).reset_index(drop=True)
    
    logging.info("==== MODEL EVALUATION COMPLETE ====")
    return results_df

def plot_model_comparison(results_df, output_path='outputs/model_comparison.png'):
    """Visualize model performance comparison."""
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1_Score', 'ROC_AUC']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        data = results_df.sort_values(metric, ascending=False)
        bars = ax.barh(data['Model'], data[metric], color=sns.color_palette('viridis', len(data)))
        ax.set_xlabel(metric, fontsize=11, fontweight='bold')
        ax.set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
        ax.set_xlim(0, 1)
        
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{width:.4f}', va='center', fontsize=9)
    
    axes[-1].axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logging.info(f"Model comparison plot saved → {output_path}")
    plt.show()

def plot_confusion_matrices(models, X_val, y_val, output_path='outputs/confusion_matrices.png'):
    """Plot confusion matrices for all models."""
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
    
    if n_models == 1:
        axes = [axes]
    
    for idx, (name, model) in enumerate(models.items()):
        y_pred = model.predict(X_val)
        cm = confusion_matrix(y_val, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], 
                   cbar=False, square=True)
        axes[idx].set_title(f'{name}', fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('Actual')
        axes[idx].set_xlabel('Predicted')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logging.info(f"Confusion matrices saved → {output_path}")
    plt.show()

def plot_roc_curves(models, X_val, y_val, output_path='outputs/roc_curves.png'):
    """Plot ROC curves for all models."""
    plt.figure(figsize=(10, 8))
    
    for name, model in models.items():
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
        auc = roc_auc_score(y_val, y_pred_proba)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.4f})', linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logging.info(f"ROC curves saved → {output_path}")
    plt.show()

def generate_classification_reports(models, X_val, y_val):
    """Generate detailed classification reports for all models."""
    reports = {}
    for name, model in models.items():
        y_pred = model.predict(X_val)
        report = classification_report(y_val, y_pred, output_dict=True)
        reports[name] = report
        logging.info(f"\n{name} Classification Report:\n{classification_report(y_val, y_pred)}")
    return reports

def save_evaluation_results(results_df, output_path='outputs/model_evaluation_results.csv'):
    """Save evaluation results to CSV."""
    results_df.to_csv(output_path, index=False)
    logging.info(f"Evaluation results saved → {output_path}")
