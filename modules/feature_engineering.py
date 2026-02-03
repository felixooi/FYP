"""
Module 5: Feature Engineering & Selection
Creates workload-related features and identifies influential predictors.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif, f_classif
import pandas.io.formats.style as style

sns.set(style="whitegrid", palette="coolwarm")

# ----------------------------------------------------------------
# Helper function: display scrollable table
# ----------------------------------------------------------------
def show_scrollable_table(df, title="Table", height=300):
    """Display a scrollable HTML table for cleaner notebook visualization."""
    try:
        from IPython.display import HTML, display
        styles = (
            "<style>.scrollable-table {overflow-y: scroll; height:"
            + str(height)
            + "px; border: 1px solid #ddd;}</style>"
        )
        html = (
            styles
            + "<div class='scrollable-table'><h4>"
            + str(title)
            + "</h4>"
            + df.to_html(index=False)
            + "</div>"
        )
        display(HTML(html))
    except ImportError:
        # Fallback for non-notebook environments
        print(f"\n{title}:")
        print(df.head(10).to_string(index=False))
        print("... (table truncated for console view) ...")



# ----------------------------------------------------------------
# Workload Feature Parameters (fit on train, apply to val/test)
# ----------------------------------------------------------------

def fit_workload_feature_params(df):
    'Compute training-only parameters for workload feature engineering.'
    params = {
        "overtime_hours_max": float(df["Overtime_Hours"].max()),
        "projects_handled_max": float(df["Projects_Handled"].max()),
        "overtime_hours_q75": float(df["Overtime_Hours"].quantile(0.75)),
        "sick_days_q75": float(df["Sick_Days"].quantile(0.75)),
        "monthly_salary_mean": float(df["Monthly_Salary"].mean()),
        "monthly_salary_std": float(df["Monthly_Salary"].std()) or 1.0,
        "performance_score_mean": float(df["Performance_Score"].mean()),
        "performance_score_std": float(df["Performance_Score"].std()) or 1.0,
    }
    return params


def apply_workload_features(df, params):
    'Apply workload feature engineering using precomputed params.'
    return create_workload_features(df, params=params)


def align_encoded_columns(train_df, other_df, target_col="Resigned"):
    'Align encoded columns of other_df to match train_df (adds missing, drops extra).'
    train_features = [c for c in train_df.columns if c != target_col]
    aligned = other_df.copy()

    for col in train_features:
        if col not in aligned.columns:
            aligned[col] = 0

    extra_cols = [c for c in aligned.columns if c not in train_features + [target_col]]
    if extra_cols:
        aligned = aligned.drop(columns=extra_cols)

    ordered_cols = train_features + ([target_col] if target_col in aligned.columns else [])
    return aligned.reindex(columns=ordered_cols)

# ----------------------------------------------------------------
# Feature Engineering
# ----------------------------------------------------------------
def create_workload_features(df, params=None):
    """Engineer new workload and performance-related features."""
    print("\n" + "="*80)
    print("FEATURE ENGINEERING: Workload-Based Features")
    print("="*80)

    df_eng = df.copy()

    if params is None:
        params = fit_workload_feature_params(df_eng)

    overtime_hours_max = params.get('overtime_hours_max', 1.0) or 1.0
    projects_handled_max = params.get('projects_handled_max', 1.0) or 1.0
    overtime_hours_q75 = params.get('overtime_hours_q75', 0.0)
    sick_days_q75 = params.get('sick_days_q75', 0.0)
    monthly_salary_mean = params.get('monthly_salary_mean', df_eng['Monthly_Salary'].mean())
    monthly_salary_std = params.get('monthly_salary_std', df_eng['Monthly_Salary'].std()) or 1.0
    performance_score_mean = params.get('performance_score_mean', df_eng['Performance_Score'].mean())
    performance_score_std = params.get('performance_score_std', df_eng['Performance_Score'].std()) or 1.0

    # Workload composite indicators
    df_eng["Workload_Intensity"] = (
        (df_eng["Work_Hours_Per_Week"] / 40) * 0.4
        + (df_eng["Overtime_Hours"] / overtime_hours_max) * 0.3
        + (df_eng["Projects_Handled"] / projects_handled_max) * 0.3
    )

    df_eng["Is_Overworked"] = (df_eng["Work_Hours_Per_Week"] > 45).astype(int)
    df_eng["Is_Underutilized"] = (df_eng["Work_Hours_Per_Week"] < 35).astype(int)
    df_eng["Overtime_Ratio"] = (
        df_eng["Overtime_Hours"] / (df_eng["Work_Hours_Per_Week"] * 4)
    ).fillna(0)
    df_eng["Project_Load"] = (
        df_eng["Projects_Handled"] / df_eng["Work_Hours_Per_Week"]
    ).replace([np.inf, -np.inf], np.nan)

    df_eng["Burnout_Risk"] = (
        (df_eng["Work_Hours_Per_Week"] > 45).astype(int) * 0.3
        + (df_eng["Overtime_Hours"] > overtime_hours_q75).astype(int)
        * 0.3
        + (df_eng["Sick_Days"] > sick_days_q75).astype(int) * 0.2
        + (df_eng["Employee_Satisfaction_Score"] < 5).astype(int) * 0.2
    )

    df_eng["Work_Life_Balance"] = (
        10
        - (df_eng["Work_Hours_Per_Week"] - 40).clip(0, 20) / 2
        - df_eng["Overtime_Hours"] / 20
    ).clip(0, 10)

    df_eng["Tenure_Performance_Ratio"] = df_eng["Years_At_Company"] / (
        df_eng["Performance_Score"] + 0.1
    )
    df_eng["Training_Per_Year"] = df_eng["Training_Hours"] / (
        df_eng["Years_At_Company"] + 1
    )

    df_eng["Salary_Performance_Gap"] = (
        (df_eng["Monthly_Salary"] - monthly_salary_mean)
        / monthly_salary_std
        - (df_eng["Performance_Score"] - performance_score_mean)
        / performance_score_std
    )

    df_eng["Age_Group"] = pd.cut(
        df_eng["Age"],
        bins=[0, 30, 40, 50, 100],
        labels=["Young", "Mid-Career", "Senior", "Veteran"],
    )
    df_eng["Tenure_Category"] = pd.cut(
        df_eng["Years_At_Company"],
        bins=[-1, 2, 5, 10],
        labels=["New", "Established", "Long-term"],
    )

    print(f"✓ Created 12 new engineered features. Dataset shape: {df_eng.shape}")
    return df_eng


# ----------------------------------------------------------------
# Categorical Encoding
# ----------------------------------------------------------------
def encode_categorical_features(df):
    """Encode categorical variables (ordinal and nominal)."""
    print("\n" + "="*80)
    print("ENCODING CATEGORICAL FEATURES")
    print("="*80)

    df_encoded = df.copy()

    # Ordinal encoding
    ordinal_mappings = {
        "Education_Level": {"High School": 1, "Bachelor": 2, "Master": 3, "PhD": 4},
        "Remote_Work_Frequency": {
            "Never": 0,
            "Rarely": 1,
            "Sometimes": 2,
            "Often": 3,
            "Always": 4,
        },
    }

    for col, mapping in ordinal_mappings.items():
        if col in df_encoded.columns:
            df_encoded[col] = df_encoded[col].map(mapping)
            print(f"✓ Ordinal encoded: {col}")

    # One-hot encoding
    nominal_cols = ["Department", "Gender", "Job_Title"]
    df_encoded = pd.get_dummies(df_encoded, columns=nominal_cols, dtype=int)
    print(f"✓ One-hot encoded nominal features: {', '.join(nominal_cols)}")

    # Ordinal for grouped features
    age_mapping = {"Young": 1, "Mid-Career": 2, "Senior": 3, "Veteran": 4}
    tenure_mapping = {"New": 1, "Established": 2, "Long-term": 3}
    df_encoded["Age_Group"] = df_encoded["Age_Group"].astype(str).map(age_mapping)
    df_encoded["Tenure_Category"] = (
        df_encoded["Tenure_Category"].astype(str).map(tenure_mapping)
    )

    # Drop non-numeric
    for col in ["Employee_ID", "Hire_Date"]:
        if col in df_encoded.columns:
            df_encoded.drop(columns=[col], inplace=True)

    print(f"✓ Encoding complete. Total features: {df_encoded.shape[1]-1}")
    return df_encoded


# ----------------------------------------------------------------
# Feature Selection
# ----------------------------------------------------------------
def select_features_importance(df, target_col="Resigned", top_k=20):
    """Select features via Random Forest importance."""
    print("\n" + "="*80)
    print("FEATURE SELECTION: Random Forest Importance")
    print("="*80)

    X = df.drop(target_col, axis=1)
    y = df[target_col]

    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)

    importances = (
        pd.DataFrame({"Feature": X.columns, "Importance": rf.feature_importances_})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )

    show_scrollable_table(importances.head(top_k), "Top Features (Random Forest)")
    sns.barplot(
        data=importances.head(top_k),
        x="Importance",
        y="Feature",
        palette="coolwarm",
    )
    plt.title("Top Feature Importances (Random Forest)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()
    return importances


def select_features_mutual_info(df, target_col="Resigned", top_k=20):
    """Select features via Mutual Information."""
    print("\n" + "="*80)
    print("FEATURE SELECTION: Mutual Information")
    print("="*80)

    X = df.drop(target_col, axis=1)
    y = df[target_col]

    mi_scores = mutual_info_classif(X, y, random_state=42)
    mi_df = (
        pd.DataFrame({"Feature": X.columns, "MI_Score": mi_scores})
        .sort_values("MI_Score", ascending=False)
        .reset_index(drop=True)
    )

    show_scrollable_table(mi_df.head(top_k), "Top Features (Mutual Information)")
    sns.barplot(data=mi_df.head(top_k), x="MI_Score", y="Feature", palette="coolwarm")
    plt.title("Top Features by Mutual Information", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()
    return mi_df


def select_features_statistical(df, target_col="Resigned", top_k=20):
    """Select features using ANOVA F-statistic."""
    print("\n" + "="*80)
    print("FEATURE SELECTION: ANOVA F-Statistic")
    print("="*80)

    X = df.drop(target_col, axis=1)
    y = df[target_col]

    f_scores, p_values = f_classif(X, y)
    f_df = (
        pd.DataFrame({"Feature": X.columns, "F_Score": f_scores, "P_Value": p_values})
        .sort_values("F_Score", ascending=False)
        .reset_index(drop=True)
    )

    show_scrollable_table(f_df.head(top_k), "Top Features (ANOVA F-Statistic)")
    return f_df


# ----------------------------------------------------------------
# Consensus Ranking
# ----------------------------------------------------------------
def get_consensus_features(rf_importance, mi_scores, f_scores, top_k=20):
    """Get consensus features from multiple methods."""
    print("\n" + "="*80)
    print("CONSENSUS FEATURE SELECTION")
    print("="*80)
    
    # Rank features by each method
    rf_importance['RF_Rank'] = range(1, len(rf_importance) + 1)
    mi_scores['MI_Rank'] = range(1, len(mi_scores) + 1)
    f_scores['F_Rank'] = range(1, len(f_scores) + 1)
    
    # Merge rankings
    consensus = rf_importance[['Feature', 'RF_Rank']].merge(
        mi_scores[['Feature', 'MI_Rank']], on='Feature'
    ).merge(
        f_scores[['Feature', 'F_Rank']], on='Feature'
    )
    
    # Calculate average rank
    consensus['Avg_Rank'] = consensus[['RF_Rank', 'MI_Rank', 'F_Rank']].mean(axis=1)
    consensus = consensus.sort_values('Avg_Rank')
    
    print(f"\nTop {top_k} Consensus Features:")
    print(consensus.head(top_k).to_string(index=False))
    
    # Visualize
    fig, ax = plt.subplots(figsize=(12, 8))
    top_features = consensus.head(top_k)
    x = np.arange(len(top_features))
    width = 0.25
    
    ax.barh(x - width, top_features['RF_Rank'], width, label='RF Rank', color='#3498db')
    ax.barh(x, top_features['MI_Rank'], width, label='MI Rank', color='#e74c3c')
    ax.barh(x + width, top_features['F_Rank'], width, label='F-Stat Rank', color='#2ecc71')
    
    ax.set_yticks(x)
    ax.set_yticklabels(top_features['Feature'])
    ax.set_xlabel('Rank (lower is better)')
    ax.set_title(f'Top {top_k} Features: Consensus Ranking', fontsize=14, fontweight='bold')
    ax.legend()
    ax.invert_yaxis()
    ax.invert_xaxis()
    
    plt.tight_layout()
    plt.savefig('outputs/09_consensus_features.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return consensus


# ----------------------------------------------------------------
# Master Pipeline
# ----------------------------------------------------------------
def engineer_and_select_features(df, target_col="Resigned", top_k=20):
    """Run complete feature engineering & selection pipeline."""
    print("\n" + "="*80)
    print("FEATURE ENGINEERING & SELECTION PIPELINE STARTED")
    print("="*80)

    df_eng = create_workload_features(df)
    df_enc = encode_categorical_features(df_eng)

    rf_imp = select_features_importance(df_enc, target_col, top_k)
    mi_imp = select_features_mutual_info(df_enc, target_col, top_k)
    f_imp = select_features_statistical(df_enc, target_col, top_k)
    consensus = get_consensus_features(rf_imp, mi_imp, f_imp, top_k)

    print("\n" + "="*80)
    print("✓ FEATURE ENGINEERING & SELECTION COMPLETED SUCCESSFULLY")
    print("="*80)

    return df_eng, df_enc, {
        "rf_importance": rf_imp,
        "mi_scores": mi_imp,
        "f_scores": f_imp,
        "consensus": consensus,
    }