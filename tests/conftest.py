"""
Pytest configuration and shared fixtures
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

@pytest.fixture
def sample_df():
    """Create sample employee dataset for testing"""
    np.random.seed(42)
    return pd.DataFrame({
        'Employee_ID': range(1, 101),
        'Department': np.random.choice(['IT', 'HR', 'Sales'], 100),
        'Gender': np.random.choice(['Male', 'Female'], 100),
        'Age': np.random.randint(22, 60, 100),
        'Job_Title': np.random.choice(['Manager', 'Developer', 'Analyst'], 100),
        'Hire_Date': pd.date_range('2015-01-01', periods=100, freq='W'),
        'Years_At_Company': np.random.randint(1, 15, 100),
        'Education_Level': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 100),
        'Performance_Score': np.random.uniform(1, 10, 100),
        'Monthly_Salary': np.random.uniform(3000, 10000, 100),
        'Work_Hours_Per_Week': np.random.randint(35, 60, 100),
        'Projects_Handled': np.random.randint(1, 10, 100),
        'Overtime_Hours': np.random.randint(0, 30, 100),
        'Sick_Days': np.random.randint(0, 15, 100),
        'Remote_Work_Frequency': np.random.choice(['Never', 'Rarely', 'Sometimes', 'Often', 'Always'], 100),
        'Team_Size': np.random.randint(3, 20, 100),
        'Training_Hours': np.random.randint(0, 100, 100).astype(float),
        'Promotions': np.random.randint(0, 5, 100),
        'Employee_Satisfaction_Score': np.random.uniform(1, 10, 100),
        'Resigned': np.random.choice([0, 1], 100, p=[0.7, 0.3])
    })

@pytest.fixture
def sample_df_with_missing(sample_df):
    """Sample dataset with missing values"""
    df = sample_df.copy()
    df.loc[0:5, 'Training_Hours'] = np.nan
    df.loc[10:12, 'Performance_Score'] = np.nan
    return df

@pytest.fixture
def sample_df_with_duplicates(sample_df):
    """Sample dataset with duplicates"""
    return pd.concat([sample_df, sample_df.iloc[:5]], ignore_index=True)

@pytest.fixture
def imbalanced_df():
    """Create imbalanced dataset with ONLY numeric features"""
    np.random.seed(42)
    return pd.DataFrame({
        'Age': np.random.randint(22, 60, 100),
        'Years_At_Company': np.random.randint(1, 15, 100),
        'Performance_Score': np.random.uniform(1, 10, 100),
        'Monthly_Salary': np.random.uniform(3000, 10000, 100),
        'Work_Hours_Per_Week': np.random.randint(35, 60, 100),
        'Projects_Handled': np.random.randint(1, 10, 100),
        'Overtime_Hours': np.random.randint(0, 30, 100),
        'Sick_Days': np.random.randint(0, 15, 100),
        'Team_Size': np.random.randint(3, 20, 100),
        'Training_Hours': np.random.randint(0, 100, 100),
        'Promotions': np.random.randint(0, 5, 100),
        'Employee_Satisfaction_Score': np.random.uniform(1, 10, 100),
        'Resigned': np.random.choice([0, 1], 100, p=[0.9, 0.1])
    })

@pytest.fixture
def temp_csv_path(tmp_path):
    """Temporary CSV file path"""
    return tmp_path / "test_data.csv"
