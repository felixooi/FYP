"""
Module 1: Data Ingestion & Understanding
Loads and provides initial inspection of the employee dataset.
"""
import pandas as pd
import numpy as np
from IPython.display import display, HTML

def load_data(filepath=None, s3_bucket=None, s3_key=None):
    """
    Load data from local file or AWS S3.
    
    Args:
        filepath: Local file path (e.g., 'data/file.csv')
        s3_bucket: AWS S3 bucket name
        s3_key: S3 object key
    
    Returns:
        DataFrame: Loaded dataset
    """
    if filepath:
        # Local file loading
        df = pd.read_csv(filepath)
        source = filepath
        print(f"✓ Dataset loaded from local: {source}")
    elif s3_bucket and s3_key:
        # AWS S3 loading
        import boto3
        from io import StringIO
        
        s3 = boto3.client('s3')
        obj = s3.get_object(Bucket=s3_bucket, Key=s3_key)
        data = obj['Body'].read().decode('utf-8')
        df = pd.read_csv(StringIO(data))
        source = f"s3://{s3_bucket}/{s3_key}"
        print(f"✓ Dataset loaded from S3: {source}")
    else:
        raise ValueError("Provide either 'filepath' for local or both 's3_bucket' and 's3_key' for S3.")

    print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    return df

def inspect_data(df):
    """Comprehensive data inspection."""
    def _scrollable_table(dataframe, max_height="300px"):
        """Return HTML for scrollable dataframe."""
        html = dataframe.to_html(classes="table table-striped table-sm", index=False)
        return f'<div style="overflow-y: scroll; height:{max_height}; border:1px solid #ccc; padding:8px;">{html}</div>'
    
    display(HTML("<h2 style='color:#2c3e50;'>DATA INSPECTION SUMMARY</h2><hr>"))

    # 1️.0 Dataset shape
    display(HTML(f"<h3>1. Dataset Shape</h3><p>Rows: {df.shape[0]} | Columns: {df.shape[1]}</p>"))

    # 2️.0 Column overview
    info_df = pd.DataFrame({
        'Column': df.columns,
        'Type': df.dtypes.values,
        'Non-Null': df.count().values,
        'Null': df.isnull().sum().values,
        'Null%': (df.isnull().sum() / len(df) * 100).round(2).values,
        'Unique': [df[col].nunique() for col in df.columns]
    })
    display(HTML("<h3>2. Column Overview</h3>"))
    display(HTML(_scrollable_table(info_df, max_height="300px")))

    # 3️.0 Target variable
    if 'Resigned' in df.columns:
        display(HTML("<h3>3. Target Variable (Resigned)</h3>"))
        vc = df['Resigned'].value_counts()
        target_summary = pd.DataFrame({
            'Resigned': vc.index,
            'Count': vc.values,
            'Percentage': (vc.values / len(df) * 100).round(2)
        })
        display(HTML(_scrollable_table(target_summary, max_height="150px")))
        display(HTML(f"<p><b>Attrition Rate:</b> {(df['Resigned'].sum() / len(df) * 100):.2f}%</p>"))
    else:
        display(HTML("<h3>3. Target Variable:</h3><p><i>Column 'Resigned' not found.</i></p>"))

    # 4️.0 Numerical summary
    num_summary = df.describe().T.round(2).reset_index()
    num_summary = num_summary.rename(columns={'index': 'Feature'})
    display(HTML("<h3>4. Numerical Features Summary</h3>"))
    display(HTML(_scrollable_table(num_summary, max_height="300px")))

    # 5️.0 Categorical overview
    display(HTML("<h3>5. Categorical Features</h3>"))
    cat_cols = df.select_dtypes(include=['object', 'bool']).columns
    if len(cat_cols) == 0:
        display(HTML("<p><i>No categorical columns detected.</i></p>"))
    else:
        for col in cat_cols:
            display(HTML(f"<b>{col}</b> ({df[col].nunique()} unique values):"))
            display(HTML(_scrollable_table(df[col].value_counts().head(10).reset_index().rename(
                columns={'index': col, col: 'Count'}
            ), max_height="150px")))

    return info_df
