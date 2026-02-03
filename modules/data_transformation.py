"""
Module 6: Data Transformation
Scales and normalizes features for modeling readiness.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import matplotlib.pyplot as plt
import seaborn as sns

def compare_scaling_methods(df, target_col='Resigned', sample_features=5):
    """Compare different scaling methods."""
    print("\n" + "="*80)
    print("COMPARING SCALING METHODS")
    print("="*80)
    
    X = df.drop(target_col, axis=1)
    
    # Select sample features for visualization
    numeric_cols = X.select_dtypes(include=[np.number]).columns[:sample_features]
    
    scalers = {
        'StandardScaler': StandardScaler(),
        'MinMaxScaler': MinMaxScaler(),
        'RobustScaler': RobustScaler()
    }
    
    fig, axes = plt.subplots(len(numeric_cols), 4, figsize=(16, len(numeric_cols) * 3))
    
    for i, col in enumerate(numeric_cols):
        # Original
        axes[i, 0].hist(X[col].dropna(), bins=30, color='#3498db', edgecolor='black')
        axes[i, 0].set_title(f'{col} - Original')
        axes[i, 0].set_ylabel('Frequency')
        
        # Apply each scaler
        for j, (name, scaler) in enumerate(scalers.items(), 1):
            scaled_data = scaler.fit_transform(X[[col]].fillna(X[col].median()))
            axes[i, j].hist(scaled_data, bins=30, color='#e74c3c', edgecolor='black')
            axes[i, j].set_title(f'{col} - {name}')
    
    plt.tight_layout()
    plt.savefig('outputs/10_scaling_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n📊 Scaling Method Characteristics:")
    print("  - StandardScaler: Mean=0, Std=1 (sensitive to outliers)")
    print("  - MinMaxScaler: Range [0,1] (sensitive to outliers)")
    print("  - RobustScaler: Uses median & IQR (robust to outliers)")
    print("\n  Recommendation: StandardScaler (most common for ML)")

def apply_scaling(df, target_col='Resigned', method='standard'):
    """Apply selected scaling method."""
    print("\n" + "="*80)
    print(f"APPLYING {method.upper()} SCALING")
    print("="*80)
    
    df_scaled = df.copy()
    X = df_scaled.drop(target_col, axis=1)
    y = df_scaled[target_col]
    
    # Select scaler
    if method.lower() == 'standard':
        scaler = StandardScaler()
    elif method.lower() == 'minmax':
        scaler = MinMaxScaler()
    elif method.lower() == 'robust':
        scaler = RobustScaler()
    else:
        print(f"⚠ Unknown method '{method}'. Using StandardScaler.")
        scaler = StandardScaler()
    
    # Identify numeric columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    
    print(f"\nScaling {len(numeric_cols)} numeric features...")
    
    # Fit and transform
    X_scaled = X.copy()
    if numeric_cols:  # Only scale if there are numeric columns
        X_scaled[numeric_cols] = scaler.fit_transform(X[numeric_cols])
        print(f"✓ Scaling completed using {scaler.__class__.__name__}")
        print(f"\nScaled data statistics (first 5 features):")
        # Reconstruct dataframe first
        df_scaled = X_scaled.copy()
        df_scaled[target_col] = y
        print(df_scaled[numeric_cols[:5]].describe())
    else:
        print("⚠ No numeric columns found. Scaler created but not fitted.")
        # Reconstruct dataframe
        df_scaled = X_scaled.copy()
        df_scaled[target_col] = y
    
    return df_scaled, scaler



def fit_scaler(df, target_col='Resigned', method='standard'):
    'Fit a scaler on training data only and return scaler + numeric columns.'
    X = df.drop(target_col, axis=1)
    if method.lower() == 'standard':
        scaler = StandardScaler()
    elif method.lower() == 'minmax':
        scaler = MinMaxScaler()
    elif method.lower() == 'robust':
        scaler = RobustScaler()
    else:
        print(f"Unknown method '{method}'. Using StandardScaler.")
        scaler = StandardScaler()

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        scaler.fit(X[numeric_cols])
    return scaler, numeric_cols


def transform_with_scaler(df, scaler, numeric_cols, target_col='Resigned'):
    'Transform data using a pre-fitted scaler and consistent numeric columns.'
    df_scaled = df.copy()
    X = df_scaled.drop(target_col, axis=1)

    for col in numeric_cols:
        if col not in X.columns:
            X[col] = 0

    if numeric_cols:
        X[numeric_cols] = scaler.transform(X[numeric_cols])

    df_out = X.copy()
    df_out[target_col] = df_scaled[target_col].values
    return df_out

def visualize_before_after(df_before, df_after, target_col='Resigned', n_features=6):
    """Visualize distributions before and after transformation."""
    print("\n" + "="*80)
    print("BEFORE vs AFTER TRANSFORMATION COMPARISON")
    print("="*80)
    
    # Select numeric features
    numeric_cols = df_before.select_dtypes(include=[np.number]).columns.drop(target_col).tolist()[:n_features]
    
    fig, axes = plt.subplots(n_features, 2, figsize=(14, n_features * 3))
    
    for i, col in enumerate(numeric_cols):
        # Before
        if col in df_before.columns:
            axes[i, 0].hist(df_before[col].dropna(), bins=30, color='#3498db', edgecolor='black', alpha=0.7)
            axes[i, 0].set_title(f'{col} - Before Transformation', fontweight='bold')
            axes[i, 0].set_ylabel('Frequency')
            axes[i, 0].axvline(df_before[col].mean(), color='red', linestyle='--', label='Mean')
            axes[i, 0].legend()
        
        # After
        if col in df_after.columns:
            axes[i, 1].hist(df_after[col].dropna(), bins=30, color='#2ecc71', edgecolor='black', alpha=0.7)
            axes[i, 1].set_title(f'{col} - After Transformation', fontweight='bold')
            axes[i, 1].axvline(df_after[col].mean(), color='red', linestyle='--', label='Mean')
            axes[i, 1].legend()
    
    plt.tight_layout()
    plt.savefig('outputs/11_before_after_transformation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ Visualization completed")

def analyze_transformed_correlations(df, target_col='Resigned'):
    """Analyze correlations after transformation."""
    print("\n" + "="*80)
    print("POST-TRANSFORMATION CORRELATION ANALYSIS")
    print("="*80)
    
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()
    
    # Correlation with target
    target_corr = corr_matrix[target_col].sort_values(ascending=False)
    print("\nTop 15 correlations with Attrition (after transformation):")
    print(target_corr.head(15))
    
    # Heatmap - separate figure
    top_features = target_corr.abs().sort_values(ascending=False).head(20).index
    fig1, ax1 = plt.subplots(figsize=(16, 14))
    sns.heatmap(corr_matrix.loc[top_features, top_features], annot=True, fmt='.2f',
                cmap='coolwarm', center=0, ax=ax1, cbar_kws={'label': 'Correlation'},
                annot_kws={'size': 9})
    ax1.set_title('Correlation Matrix (Top 20 Features)', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('outputs/12a_post_transformation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Bar plot - separate figure
    target_corr_plot = target_corr.drop(target_col).head(15)
    colors = ['#e74c3c' if x < 0 else '#2ecc71' for x in target_corr_plot.values]
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    target_corr_plot.plot(kind='barh', ax=ax2, color=colors)
    ax2.set_title('Top 15 Feature Correlations with Attrition', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Correlation Coefficient', fontsize=12)
    ax2.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
    plt.tight_layout()
    plt.savefig('outputs/12b_post_transformation_barplot.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return corr_matrix, target_corr

def transform_data(df_before, target_col='Resigned', scaling_method='standard'):
    """Main data transformation pipeline."""
    print("\n" + "="*80)
    print("DATA TRANSFORMATION PIPELINE")
    print("="*80)
    
    # Compare scaling methods
    compare_scaling_methods(df_before, target_col)
    
    # Apply scaling
    df_after, scaler = apply_scaling(df_before, target_col, scaling_method)
    
    # Visualize before/after
    visualize_before_after(df_before, df_after, target_col)
    
    # Analyze correlations
    corr_matrix, target_corr = analyze_transformed_correlations(df_after, target_col)
    
    print("\n" + "="*80)
    print("✓ DATA TRANSFORMATION COMPLETED")
    print("="*80)
    print(f"Transformed dataset shape: {df_after.shape}")
    
    # Save scaler for future use
    import joblib
    joblib.dump(scaler, 'data/scaler.pkl')
    print("✓ Scaler saved to: data/scaler.pkl")
    
    return df_after, scaler, {
        'corr_matrix': corr_matrix,
        'target_corr': target_corr
    }
