"""
Unit tests for data_cleaning module
"""
import pytest
import pandas as pd
import numpy as np
from modules.data_cleaning import (
    handle_missing_values,
    check_duplicates,
    fix_data_types,
    detect_outliers,
    clean_data
)

class TestHandleMissingValues:
    def test_no_missing_values(self, sample_df):
        result = handle_missing_values(sample_df.copy())
        assert result.isnull().sum().sum() == 0
    
    def test_impute_training_hours(self, sample_df_with_missing):
        df = sample_df_with_missing.copy()
        result = handle_missing_values(df)
        assert result['Training_Hours'].isnull().sum() == 0

class TestCheckDuplicates:
    def test_no_duplicates(self, sample_df):
        result = check_duplicates(sample_df.copy())
        assert len(result) == len(sample_df)
    
    def test_remove_duplicates(self, sample_df_with_duplicates):
        result = check_duplicates(sample_df_with_duplicates)
        assert len(result) < len(sample_df_with_duplicates)
        assert result.duplicated().sum() == 0

class TestFixDataTypes:
    def test_hire_date_conversion(self, sample_df):
        df = sample_df.copy()
        df['Hire_Date'] = df['Hire_Date'].astype(str)
        result = fix_data_types(df)
        assert pd.api.types.is_datetime64_any_dtype(result['Hire_Date'])
    
    def test_resigned_binary_conversion(self, sample_df):
        df = sample_df.copy()
        df['Resigned'] = ['true', 'false', '1', '0'] * 25
        result = fix_data_types(df)
        assert result['Resigned'].isin([0, 1]).all()

class TestDetectOutliers:
    def test_outlier_detection_returns_df(self, sample_df):
        result_df, outlier_info = detect_outliers(sample_df.copy())
        assert isinstance(result_df, pd.DataFrame)
        assert isinstance(outlier_info, pd.DataFrame)

class TestCleanData:
    def test_full_pipeline(self, sample_df):
        result, outlier_info = clean_data(sample_df.copy())
        assert isinstance(result, pd.DataFrame)
        assert isinstance(outlier_info, pd.DataFrame)
