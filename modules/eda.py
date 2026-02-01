"""
Author: Felix Ooi
Date: 30/10/2025
Module 3: Exploratory Data Analysis (EDA)
Comprehensive analysis of distributions, patterns, and workload-attrition relationships.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
import logging
from typing import Dict, Any
from IPython.display import display  # for scrollable tables in notebooks

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

PLOT_CONFIG = {
    "cmap": "coolwarm",
    "palette_resigned": ["#2ecc71", "#e74c3c"],
    "font_title": {"fontsize": 14, "fontweight": "bold"},
    "figure_size_large": (12, 6),
    "figure_size_square": (10, 10)
}


# -----------------------------------------------------------------------------
# UTILITIES
# -----------------------------------------------------------------------------

def set_plot_style():
    """Configure global plotting style."""
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "figure.figsize": PLOT_CONFIG["figure_size_large"],
        "font.size": 10
    })


def save_plot(fig, filename: str, desc: str):
    """Save a plot with consistent naming and logging."""
    output_path = f"outputs/{filename}"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"✅ Saved {desc}: {output_path}")


def display_scrollable_table(df: pd.DataFrame, max_height: int = 300, max_width: int = 800):
    """Return a styled DataFrame with scrollbars for Jupyter or notebook environments."""
    styled = (
        df.style
        .set_table_attributes(
            f'style="display:inline-block; overflow:auto; height:{max_height}px; width:{max_width}px;"'
        )
        .set_properties(**{"text-align": "center"})
    )
    return styled


# -----------------------------------------------------------------------------
# EDA COMPONENTS
# -----------------------------------------------------------------------------

def analyze_target_distribution(df: pd.DataFrame):
    """Analyze target variable distribution (Resigned)."""
    logger.info("=== TARGET VARIABLE ANALYSIS: ATTRITION ===")

    attrition_counts = df["Resigned"].value_counts()
    attrition_pct = df["Resigned"].value_counts(normalize=True) * 100
    ratio = attrition_counts[0] / attrition_counts[1]

    summary_df = pd.DataFrame({
        "Count": attrition_counts,
        "Percentage (%)": attrition_pct
    }).round(2)

    summary_df.loc["Imbalance Ratio", ["Count", "Percentage (%)"]] = [f"{ratio:.2f}:1", ""]

    # 📊 Display scrollable table
    display(display_scrollable_table(summary_df))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.countplot(data=df, x="Resigned", ax=axes[0], palette=PLOT_CONFIG["palette_resigned"])
    axes[0].set_title("Attrition Distribution (Count)", **PLOT_CONFIG["font_title"])
    axes[0].set_xlabel("Resigned Status")
    axes[0].set_ylabel("Count")
    axes[0].set_xticklabels(["Not Resigned", "Resigned"])

    axes[1].pie(
        attrition_counts,
        labels=["Not Resigned", "Resigned"],
        autopct="%1.1f%%",
        colors=PLOT_CONFIG["palette_resigned"],
        startangle=90
    )
    axes[1].set_title("Attrition Proportion", **PLOT_CONFIG["font_title"])

    plt.tight_layout()
    save_plot(fig, "01_attrition_distribution.png", "Attrition Distribution Plots")
    plt.show()

    return summary_df


def analyze_workload_features(df: pd.DataFrame):
    """Analyze workload-related features by attrition."""
    logger.info("=== WORKLOAD FEATURES ANALYSIS ===")

    workload_cols = [
        "Work_Hours_Per_Week", "Overtime_Hours", "Projects_Handled",
        "Sick_Days", "Years_At_Company", "Training_Hours"
    ]
    existing_cols = [c for c in workload_cols if c in df.columns]

    if not existing_cols:
        logger.warning("⚠ No workload features found.")
        return None

    summary = df.groupby("Resigned")[existing_cols].agg(["mean", "median", "std"]).round(2)
    display(display_scrollable_table(summary))

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    for idx, col in enumerate(existing_cols[:6]):
        sns.boxplot(data=df, x="Resigned", y=col, ax=axes[idx],
                    palette=PLOT_CONFIG["palette_resigned"])
        axes[idx].set_title(f"{col} by Attrition", fontweight="bold")
        axes[idx].set_xticklabels(["Not Resigned", "Resigned"])

    for idx in range(len(existing_cols), 6):
        axes[idx].axis("off")

    plt.tight_layout()
    save_plot(fig, "02_workload_features_boxplot.png", "Workload Boxplots")
    plt.show()

    return summary


def analyze_correlations(df: pd.DataFrame):
    """Analyze correlations, focusing on attrition."""
    logger.info("=== CORRELATION ANALYSIS ===")

    numeric_df = df.select_dtypes(include=[np.number]).copy()
    numeric_df.drop(columns=["Employee_ID"], errors="ignore", inplace=True)

    corr_matrix = numeric_df.corr().round(2)
    resigned_corr = corr_matrix["Resigned"].sort_values(ascending=False)

    # 📊 Scrollable correlation matrix
    display(display_scrollable_table(corr_matrix))

    # Full correlation matrix
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        corr_matrix, annot=True, fmt=".2f", cmap=PLOT_CONFIG["cmap"],
        center=0, linewidths=0.5, cbar_kws={"label": "Correlation Coefficient"}, ax=ax
    )
    ax.set_title("Correlation Matrix (All Numerical Features)", **PLOT_CONFIG["font_title"])
    plt.tight_layout()
    save_plot(fig, "03a_full_correlation_matrix.png", "Full Correlation Matrix")
    plt.show()

    # Correlation with target
    resigned_corr_plot = resigned_corr.drop("Resigned", errors="ignore")
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#e74c3c" if x < 0 else "#2ecc71" for x in resigned_corr_plot.values]
    resigned_corr_plot.plot(kind="barh", color=colors, ax=ax)
    ax.set_title("Feature Correlation with Attrition (Resigned)", **PLOT_CONFIG["font_title"])
    ax.set_xlabel("Correlation Coefficient")
    ax.axvline(x=0, color="black", linestyle="--", linewidth=0.8)
    plt.tight_layout()
    save_plot(fig, "03b_feature_correlation_with_attrition.png", "Feature Correlation with Target")
    plt.show()

    return corr_matrix, resigned_corr


def analyze_categorical_features(df: pd.DataFrame):
    """Analyze categorical variables vs. attrition using chi-square test."""
    logger.info("=== CATEGORICAL FEATURES VS ATTRITION ===")

    cat_cols = ["Department", "Gender", "Job_Title", "Education_Level", "Remote_Work_Frequency"]
    existing_cols = [c for c in cat_cols if c in df.columns]

    if not existing_cols:
        logger.warning("⚠ No categorical features found.")
        return None

    chi2_results = []
    for col in existing_cols:
        contingency = pd.crosstab(df[col], df["Resigned"])
        chi2, p_value, _, _ = chi2_contingency(contingency)
        chi2_results.append({
            "Feature": col,
            "Chi2": chi2,
            "P-Value": p_value,
            "Significant": "Yes" if p_value < 0.05 else "No"
        })

    chi2_df = pd.DataFrame(chi2_results).round(4)
    display(display_scrollable_table(chi2_df))

    # Visualize attrition rate by category
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()
    for idx, col in enumerate(existing_cols[:5]):
        attr_rate = df.groupby(col)["Resigned"].mean() * 100
        sns.barplot(x=attr_rate.index, y=attr_rate.values, ax=axes[idx], palette=PLOT_CONFIG["cmap"])
        axes[idx].set_title(f"Attrition Rate by {col}", fontweight="bold")
        axes[idx].set_ylabel("Attrition Rate (%)")
        axes[idx].tick_params(axis="x", rotation=45)

    for idx in range(len(existing_cols), 6):
        axes[idx].axis("off")

    plt.tight_layout()
    save_plot(fig, "04_categorical_attrition_rates.png", "Categorical Attrition Rates")
    plt.show()

    return chi2_df


def analyze_satisfaction_performance(df: pd.DataFrame):
    """Examine satisfaction and performance trends by attrition."""
    logger.info("=== SATISFACTION & PERFORMANCE ANALYSIS ===")

    has_sat = "Employee_Satisfaction_Score" in df.columns
    has_perf = "Performance_Score" in df.columns

    if not (has_sat or has_perf):
        logger.warning("⚠ Satisfaction and performance columns not found.")
        return None

    if has_sat and has_perf:
        summary = df.groupby("Resigned")[["Employee_Satisfaction_Score", "Performance_Score"]].mean().round(2)
        display(display_scrollable_table(summary))

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        sns.violinplot(data=df, x="Resigned", y="Employee_Satisfaction_Score",
                       ax=axes[0], palette=PLOT_CONFIG["palette_resigned"])
        sns.violinplot(data=df, x="Resigned", y="Performance_Score",
                       ax=axes[1], palette=PLOT_CONFIG["palette_resigned"])
        axes[0].set_title("Satisfaction Score by Attrition", fontweight="bold")
        axes[1].set_title("Performance Score by Attrition", fontweight="bold")
        plt.tight_layout()
        save_plot(fig, "05_satisfaction_performance.png", "Satisfaction & Performance")
        plt.show()
        return summary


def analyze_workload_intensity(df: pd.DataFrame):
    """Deep dive into workload intensity categories."""
    logger.info("=== WORKLOAD INTENSITY ANALYSIS ===")

    if "Work_Hours_Per_Week" not in df.columns:
        logger.warning("⚠ Work_Hours_Per_Week not found.")
        return None

    df["Workload_Category"] = pd.cut(
        df["Work_Hours_Per_Week"],
        bins=[0, 35, 40, 45, 100],
        labels=["Low (<35h)", "Normal (35-40h)", "High (40-45h)", "Very High (>45h)"]
    )

    workload_attr = df.groupby("Workload_Category")["Resigned"].agg(["sum", "count", "mean"])
    workload_attr["Attrition_Rate_%"] = workload_attr["mean"] * 100
    workload_attr = workload_attr.round(2)
    display(display_scrollable_table(workload_attr))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.countplot(x="Workload_Category", data=df, ax=axes[0], palette=PLOT_CONFIG["cmap"])
    axes[0].set_title("Employee Distribution by Workload", fontweight="bold")
    axes[0].tick_params(axis="x", rotation=45)

    sns.barplot(x=workload_attr.index, y=workload_attr["Attrition_Rate_%"],
                ax=axes[1], palette=PLOT_CONFIG["cmap"])
    axes[1].set_title("Attrition Rate by Workload Intensity", fontweight="bold")
    axes[1].set_ylabel("Attrition Rate (%)")
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    save_plot(fig, "06_workload_intensity.png", "Workload Intensity Analysis")
    plt.show()

    return workload_attr


# -----------------------------------------------------------------------------
# MAIN EDA EXECUTION
# -----------------------------------------------------------------------------

def perform_eda(df: pd.DataFrame, phase: str = "initial") -> dict:
    """Execute full EDA pipeline."""
    logger.info("=" * 80)
    logger.info(f"🚀 STARTING EXPLORATORY DATA ANALYSIS ({phase.upper()} PHASE)")
    logger.info("=" * 80)

    set_plot_style()

    try:
        attrition = analyze_target_distribution(df)
        workload = analyze_workload_features(df)
        corr_matrix, corr_target = analyze_correlations(df)
        chi2 = analyze_categorical_features(df)
        satisfaction = analyze_satisfaction_performance(df)
        workload_intensity = analyze_workload_intensity(df)
    except Exception as e:
        logger.exception(f"EDA process failed: {e}")
        return {}

    logger.info("=" * 80)
    logger.info(f"✅ {phase.upper()} EDA COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)

    return {
        "attrition_dist": attrition,
        "workload_summary": workload,
        "corr_matrix": corr_matrix,
        "resigned_corr": corr_target,
        "chi2_results": chi2,
        "satisfaction_summary": satisfaction,
        "workload_intensity": workload_intensity
    }
