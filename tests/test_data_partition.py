"""
Unit tests for data_partition module
"""
import pytest
import pandas as pd
import numpy as np
from modules.data_partition import partition_data, generate_partition_report, create_data_partition

class TestPartitionData:
    # TC-DP-01: Correct split sizes
    def test_correct_split_sizes(self, sample_df):
        X_train, X_val, X_test, y_train, y_val, y_test = partition_data(
            sample_df.copy(), visualize=False
        )
        total_samples = len(sample_df)
        # Check approximate split ratios (70%, 15%, 15%)
        assert abs(len(X_train) / total_samples - 0.70) < 0.05
        assert abs(len(X_val) / total_samples - 0.15) < 0.05
        assert abs(len(X_test) / total_samples - 0.15) < 0.05
    
    # TC-DP-02: Stratification check
    def test_stratification_check(self, sample_df):
        X_train, X_val, X_test, y_train, y_val, y_test = partition_data(
            sample_df.copy(), visualize=False
        )
        # Check class distribution is similar across splits (±10%)
        original_ratio = sample_df['Resigned'].mean()
        train_ratio = y_train.mean()
        val_ratio = y_val.mean()
        test_ratio = y_test.mean()
        
        assert abs(train_ratio - original_ratio) < 0.10
        assert abs(val_ratio - original_ratio) < 0.10
        assert abs(test_ratio - original_ratio) < 0.10
    
    # TC-DP-03: No data loss
    def test_no_data_loss(self, sample_df):
        X_train, X_val, X_test, y_train, y_val, y_test = partition_data(
            sample_df.copy(), visualize=False
        )
        total_output = len(X_train) + len(X_val) + len(X_test)
        assert total_output == len(sample_df)
        
        # Check target splits match feature splits
        assert len(X_train) == len(y_train)
        assert len(X_val) == len(y_val)
        assert len(X_test) == len(y_test)

class TestGeneratePartitionReport:
    # TC-DP-04: Report generation
    def test_report_generation(self, sample_df):
        X_train, X_val, X_test, y_train, y_val, y_test = partition_data(
            sample_df.copy(), visualize=False
        )
        report = generate_partition_report(X_train, X_val, X_test, y_train, y_val, y_test)
        assert isinstance(report, pd.DataFrame)
        assert len(report) == 3  # Should have 3 rows (train, val, test)
    
    # TC-DP-05: Report structure
    def test_report_structure(self, sample_df):
        X_train, X_val, X_test, y_train, y_val, y_test = partition_data(
            sample_df.copy(), visualize=False
        )
        report = generate_partition_report(X_train, X_val, X_test, y_train, y_val, y_test)
        
        # Check required columns exist
        required_columns = ['Split', 'Samples', 'Class_0', 'Class_1']
        for col in required_columns:
            assert col in report.columns
        
        # Check split names
        expected_splits = ['Training', 'Validation', 'Test']
        assert list(report['Split']) == expected_splits

class TestCreateDataPartition:
    # TC-DP-08: Full pipeline
    def test_full_pipeline(self, sample_df):
        result = create_data_partition(sample_df.copy(), visualize=False, save_data=False)
        
        # Check all expected keys exist
        expected_keys = ['X_train', 'X_val', 'X_test', 'y_train', 'y_val', 'y_test', 'report']
        for key in expected_keys:
            assert key in result
        
        # Check report is DataFrame
        assert isinstance(result['report'], pd.DataFrame)

class TestNegativeCases:
    # TC-DP-N01: Invalid split sizes
    def test_invalid_split_sizes(self, sample_df):
        with pytest.raises(ValueError):
            partition_data(sample_df.copy(), test_size=0.6, val_size=0.5, visualize=False)
    
    # TC-DP-N02: Missing target column
    def test_missing_target_column(self, sample_df):
        with pytest.raises(ValueError):
            partition_data(sample_df.copy(), target_col='NonExistent', visualize=False)
    
    # TC-DP-N03: Single class only
    def test_single_class_only(self):
        # Create dataset with only one class
        single_class_df = pd.DataFrame({
            'Feature1': [1, 2, 3, 4, 5],
            'Feature2': [10, 20, 30, 40, 50],
            'Resigned': [0, 0, 0, 0, 0]  # All same class
        })
        with pytest.raises(ValueError):
            partition_data(single_class_df, visualize=False)
