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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def train_logistic_regression(X_train, y_train, random_state=42):
    """Train Logistic Regression with L2 regularization."""
    logging.info("Training Logistic Regression...")
    model = LogisticRegression(
        random_state=random_state,
        max_iter=1000,
        solver='lbfgs',
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    logging.info("Logistic Regression training complete.")
    return model

def train_random_forest(X_train, y_train, random_state=42):
    """Train Random Forest classifier."""
    logging.info("Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=random_state,
        class_weight='balanced',
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
        use_label_encoder=False,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    logging.info("XGBoost training complete.")
    return model

def train_lightgbm(X_train, y_train, random_state=42):
    """Train LightGBM classifier."""
    logging.info("Training LightGBM...")
    model = lgb.LGBMClassifier(
        n_estimators=100,
        random_state=random_state,
        class_weight='balanced',
        n_jobs=-1,
        verbose=-1
    )
    model.fit(X_train, y_train)
    logging.info("LightGBM training complete.")
    return model

def train_all_models(X_train, y_train, random_state=42):
    """Train all baseline models and return dictionary."""
    logging.info("==== MODEL TRAINING PIPELINE START ====")
    
    models = {
        'Logistic_Regression': train_logistic_regression(X_train, y_train, random_state),
        'Random_Forest': train_random_forest(X_train, y_train, random_state),
        'XGBoost': train_xgboost(X_train, y_train, random_state),
        'LightGBM': train_lightgbm(X_train, y_train, random_state)
    }
    
    logging.info(f"All {len(models)} models trained successfully.")
    logging.info("==== MODEL TRAINING PIPELINE COMPLETE ====")
    return models

def save_models(models, output_dir='models'):
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
        'random_state': 42
    }
    
    import json
    with open(f"{output_dir}/training_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=4)
    logging.info("Training metadata saved.")

def load_model(model_path):
    """Load a trained model from disk."""
    return joblib.load(model_path)
