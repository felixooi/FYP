"""
Post-Transformation EDA Module
Validates that feature engineering and scaling improved data quality for modeling
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def compare_distributions(df_before, df_after, features=None, n_cols=3):
    """Compare feature distributions before and after transformation"""
    print("\n" + "="*80)
    print("DISTRIBUTION COMPARISON: Before vs After Transformation")
    print("="*80)
    
    if features is None:
        features = df_before.select_dtypes(include=[np.number]).columns[:9]
    
    n_features = len(features)
    n_rows = int(np.ceil(n_features / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 3))
    axes = axes.flatten()
    
    for i, col in enumerate(features):
        if col in df_before.columns and col in df_after.columns:
            axes[i].hist(df_before[col], bins=30, alpha=0.5, label='Before', color='#e74c3c')
            axes[i].hist(df_after[col], bins=30, alpha=0.5, label='After', color='#2ecc71')
            axes[i].set_title(f'{col}', fontweight='bold')
            axes[i].legend()
            axes[i].set_ylabel('Frequency')
    
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.savefig('outputs/16_distribution_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Distribution comparison saved")

def analyze_skewness_reduction(df_before, df_after):
    """Analyze skewness reduction after transformation"""
    print("\n" + "="*80)
    print("SKEWNESS REDUCTION ANALYSIS")
    print("="*80)
    
    # Get common numeric columns
    numeric_before = df_before.select_dtypes(include=[np.number]).columns.drop('Resigned', errors='ignore')
    numeric_after = df_after.select_dtypes(include=[np.number]).columns.drop('Resigned', errors='ignore')
    numeric_cols = list(set(numeric_before) & set(numeric_after))
    
    skew_before = df_before[numeric_cols].skew()
    skew_after = df_after[numeric_cols].skew()
    
    comparison = pd.DataFrame({
        'Feature': numeric_cols,
        'Skew_Before': skew_before.values,
        'Skew_After': skew_after.values,
        'Improvement': (abs(skew_before) - abs(skew_after)).values
    }).sort_values('Improvement', ascending=False)
    
    print("\nTop 10 Features with Skewness Reduction:")
    print(comparison.head(10).to_string(index=False))
    
    # Visualization
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(comparison.head(15)))
    width = 0.35
    
    ax.bar(x - width/2, abs(comparison.head(15)['Skew_Before']), width, label='Before', color='#e74c3c')
    ax.bar(x + width/2, abs(comparison.head(15)['Skew_After']), width, label='After', color='#2ecc71')
    
    ax.set_xlabel('Features')
    ax.set_ylabel('Absolute Skewness')
    ax.set_title('Skewness Reduction: Top 15 Features', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(comparison.head(15)['Feature'], rotation=45, ha='right')
    ax.legend()
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Acceptable threshold')
    
    plt.tight_layout()
    plt.savefig('outputs/17_skewness_reduction.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return comparison

def correlation_strength_analysis(df_before, df_after, target='Resigned'):
    """Compare correlation strength with target before and after"""
    print("\n" + "="*80)
    print("CORRELATION STRENGTH ANALYSIS")
    print("="*80)
    
    # Select only numeric columns for correlation
    df_before_numeric = df_before.select_dtypes(include=[np.number])
    df_after_numeric = df_after.select_dtypes(include=[np.number])
    
    corr_before = df_before_numeric.corr()[target].drop(target).abs().sort_values(ascending=False)
    corr_after = df_after_numeric.corr()[target].drop(target).abs().sort_values(ascending=False)
    
    # Find common features
    common_features = list(set(corr_before.index) & set(corr_after.index))[:15]
    
    comparison = pd.DataFrame({
        'Feature': common_features,
        'Corr_Before': [corr_before.get(f, 0) for f in common_features],
        'Corr_After': [corr_after.get(f, 0) for f in common_features]
    })
    comparison['Change'] = comparison['Corr_After'] - comparison['Corr_Before']
    comparison = comparison.sort_values('Corr_After', ascending=False)
    
    print("\nTop 15 Features by Correlation with Target:")
    print(comparison.to_string(index=False))
    
    # Visualization
    fig, ax = plt.subplots(figsize=(12, 8))
    y_pos = np.arange(len(comparison))
    
    ax.barh(y_pos, comparison['Corr_After'], color='#2ecc71', alpha=0.8, label='After')
    ax.barh(y_pos, comparison['Corr_Before'], color='#e74c3c', alpha=0.5, label='Before')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(comparison['Feature'])
    ax.set_xlabel('Absolute Correlation with Attrition')
    ax.set_title('Feature-Target Correlation: Before vs After', fontweight='bold', fontsize=14)
    ax.legend()
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig('outputs/18_correlation_strength.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return comparison

def multicollinearity_check(df_before, df_after):
    """Check multicollinearity reduction"""
    print("\n" + "="*80)
    print("MULTICOLLINEARITY ANALYSIS")
    print("="*80)
    
    def get_high_corr_pairs(df, threshold=0.8):
        corr_matrix = df.select_dtypes(include=[np.number]).corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_corr = [(col, row, upper.loc[row, col]) 
                     for col in upper.columns 
                     for row in upper.index 
                     if upper.loc[row, col] > threshold]
        return high_corr
    
    high_before = get_high_corr_pairs(df_before)
    high_after = get_high_corr_pairs(df_after)
    
    print(f"\nHigh correlation pairs (>0.8):")
    print(f"  Before: {len(high_before)} pairs")
    print(f"  After: {len(high_after)} pairs")
    print(f"  Reduction: {len(high_before) - len(high_after)} pairs")
    
    if len(high_after) > 0:
        print("\nRemaining high correlations:")
        for feat1, feat2, corr in high_after[:5]:
            print(f"  {feat1} <-> {feat2}: {corr:.3f}")
    
    return len(high_before), len(high_after)

def class_separability_analysis(df_before, df_after, target='Resigned'):
    """Analyze class separability improvement"""
    print("\n" + "="*80)
    print("CLASS SEPARABILITY ANALYSIS")
    print("="*80)
    
    numeric_cols = df_before.select_dtypes(include=[np.number]).columns.drop(target, errors='ignore')[:6]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, col in enumerate(numeric_cols):
        if col in df_after.columns:
            # After transformation
            for class_val in [0, 1]:
                data = df_after[df_after[target] == class_val][col]
                axes[i].hist(data, bins=30, alpha=0.6, 
                           label=f'Class {class_val}', 
                           color='#e74c3c' if class_val == 1 else '#3498db')
            
            axes[i].set_title(f'{col}', fontweight='bold')
            axes[i].set_xlabel('Value')
            axes[i].set_ylabel('Frequency')
            axes[i].legend()
    
    plt.suptitle('Class Separability After Transformation', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/19_class_separability.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ Class separability visualization saved")

def variance_analysis(df_before, df_after):
    """Analyze variance changes"""
    print("\n" + "="*80)
    print("VARIANCE ANALYSIS")
    print("="*80)
    
    # Get common numeric columns
    numeric_before = df_before.select_dtypes(include=[np.number]).columns.drop('Resigned', errors='ignore')
    numeric_after = df_after.select_dtypes(include=[np.number]).columns.drop('Resigned', errors='ignore')
    numeric_cols = list(set(numeric_before) & set(numeric_after))
    
    var_before = df_before[numeric_cols].var()
    var_after = df_after[numeric_cols].var()
    
    comparison = pd.DataFrame({
        'Feature': numeric_cols,
        'Var_Before': var_before.values,
        'Var_After': var_after.values,
        'Ratio': (var_after / var_before).values
    }).sort_values('Var_After', ascending=False)
    
    print("\nVariance Comparison (Top 10):")
    print(comparison.head(10).to_string(index=False))
    
    # Check for near-zero variance features
    low_var = comparison[comparison['Var_After'] < 0.01]
    if len(low_var) > 0:
        print(f"\n⚠ Warning: {len(low_var)} features with very low variance (<0.01)")
    else:
        print("\n✓ All features have sufficient variance")
    
    return comparison

def generate_summary_report(df_before, df_after):
    """Generate comprehensive summary report"""
    print("\n" + "="*80)
    print("POST-TRANSFORMATION SUMMARY REPORT")
    print("="*80)
    
    report = {
        'Metric': [],
        'Before': [],
        'After': [],
        'Status': []
    }
    
    # Dataset size
    report['Metric'].append('Dataset Size')
    report['Before'].append(f"{df_before.shape[0]} rows")
    report['After'].append(f"{df_after.shape[0]} rows")
    report['Status'].append('✓' if df_before.shape[0] == df_after.shape[0] else '⚠')
    
    # Feature count
    report['Metric'].append('Feature Count')
    report['Before'].append(df_before.shape[1])
    report['After'].append(df_after.shape[1])
    report['Status'].append('✓')
    
    # Missing values
    report['Metric'].append('Missing Values')
    report['Before'].append(df_before.isnull().sum().sum())
    report['After'].append(df_after.isnull().sum().sum())
    report['Status'].append('✓' if df_after.isnull().sum().sum() == 0 else '⚠')
    
    # Mean skewness
    skew_before = abs(df_before.select_dtypes(include=[np.number]).skew()).mean()
    skew_after = abs(df_after.select_dtypes(include=[np.number]).skew()).mean()
    report['Metric'].append('Mean Abs Skewness')
    report['Before'].append(f"{skew_before:.3f}")
    report['After'].append(f"{skew_after:.3f}")
    report['Status'].append('✓' if skew_after < skew_before else '⚠')
    
    # Data types
    report['Metric'].append('Numeric Features')
    report['Before'].append(len(df_before.select_dtypes(include=[np.number]).columns))
    report['After'].append(len(df_after.select_dtypes(include=[np.number]).columns))
    report['Status'].append('✓')
    
    summary_df = pd.DataFrame(report)
    print("\n", summary_df.to_string(index=False))
    
    return summary_df

def run_post_transformation_eda(df_before, df_after, target='Resigned'):
    """Main function to run complete post-transformation analysis"""
    print("\n" + "="*80)
    print("STARTING POST-TRANSFORMATION EDA")
    print("="*80)
    
    # 1. Distribution comparison
    compare_distributions(df_before, df_after)
    
    # 2. Skewness reduction
    skew_comp = analyze_skewness_reduction(df_before, df_after)
    
    # 3. Correlation strength
    corr_comp = correlation_strength_analysis(df_before, df_after, target)
    
    # 4. Multicollinearity
    multi_before, multi_after = multicollinearity_check(df_before, df_after)
    
    # 5. Class separability
    class_separability_analysis(df_before, df_after, target)
    
    # 6. Variance analysis
    var_comp = variance_analysis(df_before, df_after)
    
    # 7. Summary report
    summary = generate_summary_report(df_before, df_after)
    
    print("\n" + "="*80)
    print("✓ POST-TRANSFORMATION EDA COMPLETED")
    print("="*80)
    print("\nKey Findings:")
    print(f"  • Skewness improved for {(skew_comp['Improvement'] > 0).sum()} features")
    print(f"  • Multicollinearity reduced by {multi_before - multi_after} pairs")
    print(f"  • All features scaled and ready for modeling")
    
    return {
        'skewness': skew_comp,
        'correlation': corr_comp,
        'variance': var_comp,
        'summary': summary
    }
