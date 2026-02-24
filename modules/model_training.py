"""
Module 8: Model Training
Trains baseline ML models for employee attrition prediction.
"""

import pandas as pd
import numpy as np
import logging
import joblib
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def train_logistic_regression(X_train, y_train, random_state=42, class_weight=None):
    """Train Logistic Regression with L2 regularization."""
    logging.info("Training Logistic Regression...")
    model = LogisticRegression(
        random_state=random_state,
        max_iter=1000,
        solver='lbfgs',
        class_weight=class_weight
    )
    model.fit(X_train, y_train)
    logging.info("Logistic Regression training complete.")
    return model

def train_random_forest(X_train, y_train, random_state=42, class_weight=None):
    """Train Random Forest classifier."""
    logging.info("Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=random_state,
        class_weight=class_weight,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    logging.info("Random Forest training complete.")
    return model

def train_xgboost(X_train, y_train, random_state=42):
    """Train XGBoost classifier."""
    logging.info("Training XGBoost...")
    model = xgb.XGBClassifier(
        n_estimators=100,
        random_state=random_state,
        eval_metric='logloss',
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    logging.info("XGBoost training complete.")
    return model

def train_lightgbm(X_train, y_train, random_state=42, class_weight=None):
    """Train LightGBM classifier."""
    logging.info("Training LightGBM...")
    model = lgb.LGBMClassifier(
        n_estimators=100,
        random_state=random_state,
        class_weight=class_weight,
        n_jobs=-1,
        verbose=-1
    )
    model.fit(X_train, y_train)
    logging.info("LightGBM training complete.")
    return model

def train_tabnet(X_train, y_train, random_state=42):
    """Train TabNet classifier with attention mechanism (Advanced Tier - Deep Learning)."""
    logging.info("Training TabNet (Advanced Tier - Deep Learning)...")
    
    try:
        from pytorch_tabnet.tab_model import TabNetClassifier
    except ImportError:
        logging.error("pytorch-tabnet not installed. Run: pip install pytorch-tabnet")
        raise
    
    # Set random seeds for reproducibility
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    
    model = TabNetClassifier(
        n_d=64,                    # Width of decision prediction layer
        n_a=64,                    # Width of attention embedding
        n_steps=5,                 # Number of decision steps
        gamma=1.5,                 # Relaxation parameter for feature reuse
        n_independent=2,           # Number of independent GLU layers
        n_shared=2,                # Number of shared GLU layers
        lambda_sparse=1e-4,        # Sparsity regularization
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        scheduler_params={"step_size":50, "gamma":0.9},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        mask_type='entmax',        # Attention mask type
        verbose=0,
        seed=random_state
    )
    
    # Train with early stopping
    model.fit(
        X_train.values, y_train.values,
        max_epochs=100,
        patience=20,
        batch_size=1024,
        virtual_batch_size=128
    )
    
    logging.info("TabNet training complete.")
    return model

def train_all_models(X_train, y_train, random_state=42, use_class_weight=False):
    """Train all baseline and advanced models and return dictionary."""
    logging.info("==== MODEL TRAINING PIPELINE START ====")
    logging.info("Training Baseline Models (Traditional ML)...")
    
    models = {
        'Logistic_Regression': train_logistic_regression(X_train, y_train, random_state, class_weight='balanced' if use_class_weight else None),
        'Random_Forest': train_random_forest(X_train, y_train, random_state, class_weight='balanced' if use_class_weight else None),
        'XGBoost': train_xgboost(X_train, y_train, random_state),
        'LightGBM': train_lightgbm(X_train, y_train, random_state, class_weight='balanced' if use_class_weight else None)
    }
    
    logging.info("Training Advanced Model (Deep Learning)...")
    try:
        models['TabNet'] = train_tabnet(X_train, y_train, random_state)
    except Exception as e:
        logging.warning(f"TabNet training failed: {e}")
        logging.warning("Continuing with baseline models only.")
    
    logging.info(f"All {len(models)} models trained successfully.")
    logging.info("==== MODEL TRAINING PIPELINE COMPLETE ====")
    return models

def save_models(models, output_dir='models', random_state=42):
    """Save trained models to disk."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    for name, model in models.items():
        path = f"{output_dir}/{name}.pkl"
        joblib.dump(model, path)
        logging.info(f"Saved {name} → {path}")
    
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'models': list(models.keys()),
        'random_state': random_state
    }
    
    import json
    with open(f"{output_dir}/training_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=4)
    logging.info("Training metadata saved.")

def load_model(model_path):
    """Load a trained model from disk."""
    return joblib.load(model_path)
