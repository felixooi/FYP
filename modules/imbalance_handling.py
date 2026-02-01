"""
Module 4: Handling Data Imbalance
Provides analysis and resampling utilities using SMOTE, ADASYN, and hybrid methods.
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict, Any
from collections import Counter
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTETomek
import matplotlib.pyplot as plt
from IPython.display import display

# =============================================================================
# Logging Configuration
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# =============================================================================
# Core Functions
# =============================================================================
def analyze_imbalance(df: pd.DataFrame, target_col: str = "Resigned") -> float:
    """
    Analyze class imbalance in the dataset.
    Displays a scrollable summary and returns imbalance ratio.
    """
    logger.info("Analyzing class distribution...")

    class_dist = df[target_col].value_counts()
    class_pct = (class_dist / len(df) * 100).round(2)
    imbalance_ratio = class_dist.max() / class_dist.min()

    summary = pd.DataFrame({
        "Class": class_dist.index,
        "Count": class_dist.values,
        "Percentage (%)": class_pct.values
    })
    
    display(summary)
    logger.info(f"Imbalance ratio: {imbalance_ratio:.2f}:1")

    if imbalance_ratio > 3:
        logger.warning("Severe imbalance detected (ratio > 3:1)")
    elif imbalance_ratio > 1.5:
        logger.warning("Moderate imbalance detected (ratio > 1.5:1)")
    else:
        logger.info("Dataset is relatively balanced.")

    return imbalance_ratio


def _apply_sampler(X: pd.DataFrame, y: pd.Series, sampler, name: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Apply a resampling method (SMOTE, ADASYN, etc.).
    """
    logger.info(f"Applying {name}...")
    try:
        X_res, y_res = sampler.fit_resample(X, y)
        before, after = Counter(y), Counter(y_res)
        logger.info(f"{name} completed. Before: {before} | After: {after}")
        return X_res, y_res
    except Exception as e:
        logger.error(f"{name} failed: {e}")
        raise


def compare_resampling_methods(X: pd.DataFrame, y: pd.Series, random_state: int = 42, plot: bool = True) -> pd.DataFrame:
    """
    Compare SMOTE, ADASYN, and SMOTE-Tomek resampling effects.
    Returns a summary DataFrame and optionally plots comparison.
    """
    methods = {
        "Original": None,
        "SMOTE": SMOTE(random_state=random_state),
        "ADASYN": ADASYN(random_state=random_state),
        "SMOTE-Tomek": SMOTETomek(random_state=random_state)
    }

    results = []
    for name, sampler in methods.items():
        if sampler:
            X_res, y_res = _apply_sampler(X, y, sampler, name)
            dist = Counter(y_res)
        else:
            dist = Counter(y)
        results.append({
            "Method": name,
            "Total Samples": sum(dist.values()),
            "Majority (0)": dist.get(0, 0),
            "Minority (1)": dist.get(1, 0),
            "Imbalance Ratio": round(dist.get(0, 0) / max(dist.get(1, 1), 1), 2)
        })

    comparison_df = pd.DataFrame(results).set_index("Method")
    display(comparison_df.style.background_gradient(cmap="coolwarm"))

    if plot:
        _plot_resampling_comparison(comparison_df)

    return comparison_df


def handle_imbalance(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str = "Resigned",
    method: str = "smote",
    random_state: int = 42,
    plot: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Main imbalance handling pipeline.
    Supports SMOTE, ADASYN, and SMOTE-Tomek.
    """
    logger.info("Starting imbalance handling pipeline...")
    imbalance_ratio = analyze_imbalance(df, target_col)

    X, y = df[feature_cols], df[target_col]
    method = method.lower()

    sampler_map = {
        "smote": SMOTE(random_state=random_state),
        "adasyn": ADASYN(random_state=random_state),
        "smote-tomek": SMOTETomek(random_state=random_state)
    }

    if method not in sampler_map:
        logger.warning(f"Unknown method '{method}'. Defaulting to SMOTE.")
        sampler = sampler_map["smote"]
    else:
        sampler = sampler_map[method]

    X_res, y_res = _apply_sampler(X, y, sampler, method.upper())
    df_resampled = pd.DataFrame(X_res, columns=feature_cols)
    df_resampled[target_col] = y_res

    logger.info(f"Resampled dataset shape: {df_resampled.shape}")

    if plot:
        _plot_class_distribution(y, y_res)

    return df_resampled, X_res, y_res

# =============================================================================
# Visualization Utilities
# =============================================================================
def _plot_class_distribution(y_before, y_after):
    """Visualize class distribution before and after resampling."""
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    pd.Series(y_before).value_counts().plot(kind="bar", ax=ax[0], color="lightcoral", title="Before Resampling")
    pd.Series(y_after).value_counts().plot(kind="bar", ax=ax[1], color="skyblue", title="After Resampling")
    plt.suptitle("Class Distribution Before vs After Resampling", fontsize=12)
    plt.tight_layout()
    plt.show()


def _plot_resampling_comparison(comparison_df: pd.DataFrame):
    """Plot comparison of resampling methods."""
    fig, ax = plt.subplots(figsize=(8, 5))
    comparison_df[["Majority (0)", "Minority (1)"]].plot(
        kind="bar", ax=ax, colormap="coolwarm"
    )
    plt.title("Resampling Methods Comparison", fontsize=13)
    plt.ylabel("Sample Count")
    plt.xticks(rotation=0)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()
