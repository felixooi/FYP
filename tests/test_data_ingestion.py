"""
Unit tests for data_ingestion module
"""
import pytest
import pandas as pd
from modules.data_ingestion import load_data, inspect_data

class TestLoadData:
    def test_load_from_local_file(self, sample_df, temp_csv_path):
        sample_df.to_csv(temp_csv_path, index=False)
        result = load_data(filepath=str(temp_csv_path))
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_df)
    
    def test_load_missing_file_raises_error(self):
        with pytest.raises(Exception):
            load_data(filepath="nonexistent_file.csv")
    
    def test_load_no_params_raises_error(self):
        with pytest.raises(ValueError):
            load_data()
    
    def test_load_preserves_columns(self, sample_df, temp_csv_path):
        sample_df.to_csv(temp_csv_path, index=False)
        result = load_data(filepath=str(temp_csv_path))
        assert list(result.columns) == list(sample_df.columns)

class TestInspectData:
    def test_inspect_returns_info_df(self, sample_df):
        result = inspect_data(sample_df)
        assert isinstance(result, pd.DataFrame)
    
    def test_inspect_info_structure(self, sample_df):
        result = inspect_data(sample_df)
        expected_cols = ['Column', 'Type', 'Non-Null', 'Null', 'Null%', 'Unique']
        assert all(col in result.columns for col in expected_cols)
