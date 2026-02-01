"""
Module 2: Data Cleaning & Preprocessing
Handles missing values, data types, duplicates, and outliers.
"""
import pandas as pd
import numpy as np
from scipy import stats
import logging

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def handle_missing_values(df):
    """Identify and handle missing values."""
    logger.info("="*80)
    logger.info("MISSING VALUE ANALYSIS")
    logger.info("="*80)
    
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({'Missing': missing, 'Percentage': missing_pct})
    missing_df = missing_df[missing_df['Missing'] > 0].sort_values('Missing', ascending=False)
    
    if len(missing_df) > 0:
        logger.info("\nColumns with missing values:\n%s", missing_df)
        
        # Imputation strategy: Median for Training_Hours
        if 'Training_Hours' in missing_df.index:
            median_val = df['Training_Hours'].median()
            df['Training_Hours'].fillna(median_val, inplace=True)
            logger.info(f"✓ Training_Hours: Imputed {missing_df.loc['Training_Hours', 'Missing']} missing values with median ({median_val:.1f})")
    else:
        logger.info("✓ No missing values detected")
    
    return df


def check_duplicates(df):
    """Check and remove duplicate records."""
    logger.info("="*80)
    logger.info("DUPLICATE ANALYSIS")
    logger.info("="*80)
    
    duplicates = df.duplicated().sum()
    logger.info(f"Duplicate rows: {duplicates}")
    
    if duplicates > 0:
        df = df.drop_duplicates()
        logger.info(f"✓ Removed {duplicates} duplicate rows")
    else:
        logger.info("✓ No duplicates found")
    
    # Check Employee_ID uniqueness
    id_duplicates = df['Employee_ID'].duplicated().sum()
    if id_duplicates > 0:
        logger.warning(f"⚠ Duplicate Employee_IDs found: {id_duplicates}")
    else:
        logger.info("✓ Employee_IDs are unique")
    
    return df


def fix_data_types(df):
    """Convert columns to appropriate data types."""
    logger.info("="*80)
    logger.info("DATA TYPE CONVERSION")
    logger.info("="*80)
    
    # Convert Hire_Date to datetime
    if 'Hire_Date' in df.columns:
        df['Hire_Date'] = pd.to_datetime(df['Hire_Date'], errors='coerce')
        logger.info("✓ Hire_Date converted to datetime")
    
    # Convert Resigned to integer (0/1)
    if 'Resigned' in df.columns:
        df['Resigned'] = df['Resigned'].apply(
        lambda x: 1 if str(x).strip().lower() in ['true', '1', 'yes', 'resigned'] else 0
    )
    logger.info("✓ Resigned column standardized to binary (1=Resigned, 0=Active)")
    
    # Ensure numeric columns have correct types
    numeric_cols = ['Age', 'Years_At_Company', 'Performance_Score', 'Monthly_Salary', 
                    'Work_Hours_Per_Week', 'Projects_Handled', 'Overtime_Hours', 
                    'Sick_Days', 'Team_Size', 'Training_Hours', 'Promotions', 
                    'Employee_Satisfaction_Score']
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    logger.info(f"✓ Ensured {len(numeric_cols)} numeric columns have correct types")
    
    # Display final datatypes summary
    dtype_summary = df.dtypes.reset_index()
    dtype_summary.columns = ['Feature', 'DataType']
    logger.info("\nFinal Data Types:\n%s", dtype_summary.to_string(index=False))
    
    return df


def detect_outliers(df):
    """Detect outliers using IQR and Z-score methods."""
    logger.info("="*80)
    logger.info("OUTLIER DETECTION")
    logger.info("="*80)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.drop(['Employee_ID', 'Resigned'], errors='ignore')
    outlier_summary = []
    
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        iqr_outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        
        # Z-score method
        z_scores = np.abs(stats.zscore(df[col].dropna()))
        z_outliers = (z_scores > 3).sum()
        
        outlier_summary.append({
            'Feature': col,
            'IQR_Outliers': iqr_outliers,
            'ZScore_Outliers': z_outliers,
            'Min': round(df[col].min(), 2),
            'Max': round(df[col].max(), 2)
        })
    
    outlier_df = pd.DataFrame(outlier_summary).sort_values('IQR_Outliers', ascending=False)
    
    logger.info("\nOutlier Summary (Top 10 by IQR Outliers):\n%s", outlier_df.head(10).to_string(index=False))
    logger.info("\n⚠ Outliers retained for now. These will be revisited during feature engineering if they distort model performance.")
    
    return df, outlier_df


def clean_data(df):
    """Execute complete cleaning pipeline."""
    logger.info("="*80)
    logger.info("STARTING DATA CLEANING PIPELINE")
    logger.info("="*80)
    
    df = handle_missing_values(df)
    df = check_duplicates(df)
    df = fix_data_types(df)
    df, outlier_info = detect_outliers(df)
    
    logger.info("="*80)
    logger.info("✓ DATA CLEANING COMPLETED")
    logger.info("="*80)
    logger.info(f"Final dataset shape: {df.shape}")
    
    return df, outlier_info