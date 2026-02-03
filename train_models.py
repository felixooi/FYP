"""
Model Training Pipeline - Main Execution Script
Orchestrates training, evaluation, and selection of attrition prediction models.
"""

import pandas as pd
import sys
import os

# Add modules to path
sys.path.append('modules')

from modules.model_training import train_all_models, save_models
from modules.model_evaluation import (
    evaluate_all_models, plot_model_comparison, 
    plot_confusion_matrices, plot_roc_curves,
    generate_classification_reports, save_evaluation_results
)
from modules.model_selection import perform_model_selection

def load_data(data_dir='data'):
    """Load partitioned training and validation data."""
    print("Loading data...")
    X_train = pd.read_csv(f'{data_dir}/train_data.csv')
    X_val = pd.read_csv(f'{data_dir}/val_data.csv')
    X_test = pd.read_csv(f'{data_dir}/test_data.csv')
    
    y_train = X_train['Resigned']
    y_val = X_val['Resigned']
    y_test = X_test['Resigned']
    
    X_train = X_train.drop(columns=['Resigned'])
    X_val = X_val.drop(columns=['Resigned'])
    X_test = X_test.drop(columns=['Resigned'])
    
    print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}, Test set: {X_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test

def main():
    """Execute complete model training pipeline."""
    print("="*80)
    print("EMPLOYEE ATTRITION PREDICTION - MODEL TRAINING PIPELINE")
    print("="*80)
    
    # Create output directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)
    
    # Step 1: Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()
    
    # Step 2: Train all models
    print("\n" + "="*80)
    print("STEP 1: MODEL TRAINING")
    print("="*80)
    models = train_all_models(X_train, y_train, random_state=42, use_class_weight=False)
    save_models(models, output_dir='models', random_state=42)
    
    # Step 3: Evaluate all models
    print("\n" + "="*80)
    print("STEP 2: MODEL EVALUATION")
    print("="*80)
    results_df = evaluate_all_models(models, X_val, y_val)
    print("\nEvaluation Results:")
    print(results_df.to_string(index=False))
    
    save_evaluation_results(results_df, output_path='outputs/model_evaluation_results.csv')

    # Step 2B: Evaluate all models on test set (final unbiased performance)
    print("\n" + "="*80)
    print("STEP 2B: TEST SET EVALUATION")
    print("="*80)
    test_results_df = evaluate_all_models(models, X_test, y_test)
    print("\nTest Set Results:")
    print(test_results_df.to_string(index=False))
    save_evaluation_results(test_results_df, output_path='outputs/model_evaluation_results_test.csv')
    
    # Step 4: Generate visualizations
    print("\n" + "="*80)
    print("STEP 3: GENERATING VISUALIZATIONS")
    print("="*80)
    plot_model_comparison(results_df, output_path='outputs/20_model_comparison.png')
    plot_confusion_matrices(models, X_val, y_val, output_path='outputs/21_confusion_matrices.png')
    plot_roc_curves(models, X_val, y_val, output_path='outputs/22_roc_curves.png')
    
    # Step 5: Generate classification reports
    print("\n" + "="*80)
    print("STEP 4: CLASSIFICATION REPORTS")
    print("="*80)
    reports = generate_classification_reports(models, X_val, y_val)
    
    # Step 6: Model selection
    print("\n" + "="*80)
    print("STEP 5: MODEL SELECTION")
    print("="*80)
    selection_results = perform_model_selection(
        results_df, models, 
        primary_metric='Recall',
        secondary_metric='F1_Score',
        save_results=True,
        output_dir='models'
    )
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    print(f"\nBest Model: {selection_results['best_model_name']}")
    print(f"Models saved in: models/")
    print(f"Visualizations saved in: outputs/")
    print(f"Evaluation results: outputs/model_evaluation_results.csv")

if __name__ == "__main__":
    main()
