import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import inspect
from IPython.display import display, HTML

"""1.0 Display basic info about the dataset."""
def basic_info(df_raw):
    print("="*60)
    print("Basic Data Information")
    print("="*60)
    print(f"Shape: {df_raw.shape}")
    print("\nData Types:")
    print(df_raw.dtypes)

def missing_values_summary(df_raw):
    """Show missing values summary."""
    missing = df_raw.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        print("No missing values found.")
    else:
        print(missing)
        display(pd.DataFrame({'Missing Count': missing, 'Percent': (missing/len(df_raw))*100}))

def numerical_summary(df_raw):
    """Show summary for numeric columns only."""
    num_cols = df_raw.select_dtypes(include=np.number).columns.tolist()
    if not num_cols:
        print("No numeric columns found.")
        return
    display(df_raw[num_cols].describe())

def categorical_summary(df_raw):
    """Show summary for categorical columns only."""
    cat_cols = df_raw.select_dtypes(exclude=np.number).columns.tolist()
    if not cat_cols:
        print("No categorical columns found.")
        return
    print("Categorical column counts:")
    for col in cat_cols:
        print(f"{col}: {df_raw[col].nunique()} unique values")


# ========== VISUALIZATION FUNCTIONS ==========
def plot_numerical(df_raw, ncols=3, figsize=(15, 10)):
    num_cols = df_raw.select_dtypes(include=np.number).columns.tolist()
    
    if not num_cols:
        print("No numerical columns found.")
        return
    
    n_plots = len(num_cols)
    nrows = int(np.ceil(n_plots / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = axes.flatten()

    for i, col in enumerate(num_cols):
        sns.histplot(df_raw[col], kde=True, ax=axes[i])
        axes[i].set_title(f"Distribution of {col}")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Frequency")

    # Remove unused subplots (if any)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("Numerical Feature Distributions", fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_categorical(df_raw, figsize=(8, 4)):
    cat_cols = df_raw.select_dtypes(exclude='number').columns.tolist()
    
    if not cat_cols:
        print("No categorical columns found.")
        return

    for col in cat_cols:
        plt.figure(figsize=figsize)
        order = df_raw[col].value_counts().index
        
        # Categories on x-axis, count on y-axis
        sns.countplot(x=df_raw[col], order=order)
        plt.title(f"Distribution of {col}")
        plt.ylabel("Count")
        plt.xlabel(col)
        
        # Rotate labels for date-like or long categorical values
        if pd.api.types.is_datetime64_any_dtype(df_raw[col]) or "date" in col.lower():
            plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        plt.show()

def detect_outliers(df_raw, ncols=3, figsize=(15, 10)):
    num_cols = df_raw.select_dtypes(include=np.number).columns.tolist()
    n_plots = len(num_cols)
    nrows = int(np.ceil(n_plots / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = axes.flatten()  # Flatten to easily iterate

    for i, col in enumerate(num_cols):
        Q1 = df_raw[col].quantile(0.25)
        Q3 = df_raw[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = df_raw[(df_raw[col] < lower) | (df_raw[col] > upper)]

        print(f"{col}: {len(outliers)} outliers")

        sns.boxplot(x=df_raw[col], ax=axes[i])
        axes[i].set_title(f"{col}")
    
    # Hide any empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("Outlier Detection by Boxplots", fontsize=16)
    plt.tight_layout()
    plt.show()

def correlation_heatmap(df_raw):
    num_cols2 = df_raw.select_dtypes(include=np.number).columns.tolist()
    if len(num_cols2) < 2:
        print("Not enough numeric columns for correlation heatmap.")
        return
    plt.figure(figsize=(15,11))
    sns.heatmap(df_raw[num_cols2].corr(), annot=True, cmap='coolwarm')
    plt.title("Correlation Heatmap")
    plt.show()


def run_eda(df_raw, mode="all", max_plots=5):
    """
    Run a full or partial EDA workflow on a dataframe.
    
    Parameters
    ----------
    df_raw : pandas.DataFrame
        The dataframe to analyze.
    mode : str, optional
        Choose which analyses to run:
        - 'summary' : only text/table summaries
        - 'visual'  : only visual analyses
        - 'all'     : both (default)
    max_plots : int, optional
        Limit number of plots per type for faster execution.
    """
    print("="*80)
    print(f" Running EDA (mode='{mode}')")
    print("="*80)
    print()

    module = sys.modules[__name__]
    all_funcs = inspect.getmembers(module, inspect.isfunction)
    summary_funcs = [f for f in all_funcs if 'summary' in f[0] or 'info' in f[0]]
    visual_funcs  = [f for f in all_funcs if 'plot' in f[0] or 
                     'heatmap' in f[0] or 'outlier' in f[0]]

    def run_section(funcs, section_name):
        print(f"\n\n{'='*30}  {section_name}  {'='*30}\n")
        for name, func in funcs:
            print(f"\n→ {name}")
            print('-'*70)
            try:
                func(df_raw.head(max_plots*10))
            except Exception as e:
                print(f"Skipped {name}: {e}")
            print("\n")  # add blank space between functions

    if mode in ["all", "summary"]:
        run_section(summary_funcs, "SUMMARY FUNCTIONS")

    if mode in ["all", "visual"]:
        run_section(visual_funcs, "VISUALIZATION FUNCTIONS")

    print("\n EDA Completed!")