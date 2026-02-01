"""
Unit tests for data_transformation module
"""
import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from modules.data_transformation import apply_scaling, analyze_transformed_correlations, transform_data

class TestApplyScaling:
    # TC-DT-01: StandardScaler
    def test_standard_scaler(self, sample_df):
        df_scaled, scaler = apply_scaling(sample_df.copy(), method='standard')
        assert isinstance(scaler, StandardScaler)
        assert isinstance(df_scaled, pd.DataFrame)
        # Verify mean≈0, std≈1 for numeric columns
        numeric_cols = df_scaled.select_dtypes(include=[np.number]).columns.drop('Resigned')
        for col in numeric_cols[:3]:  # Check first 3 columns
            assert abs(df_scaled[col].mean()) < 0.1
            assert abs(df_scaled[col].std() - 1.0) < 0.1
    
    # TC-DT-02: MinMaxScaler
    def test_minmax_scaler(self, sample_df):
        df_scaled, scaler = apply_scaling(sample_df.copy(), method='minmax')
        assert isinstance(scaler, MinMaxScaler)
        # Verify range [0,1] for numeric columns (with floating point tolerance)
        numeric_cols = df_scaled.select_dtypes(include=[np.number]).columns.drop('Resigned')
        for col in numeric_cols[:3]:  # Check first 3 columns
            assert df_scaled[col].min() >= -1e-10  # Small tolerance for floating point
            assert df_scaled[col].max() <= 1 + 1e-10  # Small tolerance for floating point
    
    # TC-DT-03: RobustScaler
    def test_robust_scaler(self, sample_df):
        df_scaled, scaler = apply_scaling(sample_df.copy(), method='robust')
        assert isinstance(scaler, RobustScaler)
        assert isinstance(df_scaled, pd.DataFrame)
    
    # TC-DT-04: Preserve target
    def test_preserves_target(self, sample_df):
        df_scaled, _ = apply_scaling(sample_df.copy())
        assert 'Resigned' in df_scaled.columns
        # Verify target values unchanged
        original_target = sample_df['Resigned'].values
        scaled_target = df_scaled['Resigned'].values
        np.testing.assert_array_equal(original_target, scaled_target)
    
    # TC-DT-05: Invalid method fallback
    def test_invalid_method_fallback(self, sample_df):
        df_scaled, scaler = apply_scaling(sample_df.copy(), method='invalid')
        assert isinstance(scaler, StandardScaler)  # Should default to StandardScaler
        assert isinstance(df_scaled, pd.DataFrame)
    
    # TC-DT-08: Full pipeline
    def test_full_pipeline(self, sample_df):
        df_scaled, scaler, results = transform_data(sample_df.copy(), scaling_method='standard')
        assert isinstance(df_scaled, pd.DataFrame)
        assert isinstance(scaler, StandardScaler)
        assert 'corr_matrix' in results
        assert 'target_corr' in results

class TestAnalyzeTransformedCorrelations:
    # TC-DT-06: Correlation analysis
    def test_returns_correlation_matrix(self, sample_df):
        corr_matrix, target_corr = analyze_transformed_correlations(sample_df.copy())
        assert isinstance(corr_matrix, pd.DataFrame)
        assert isinstance(target_corr, pd.Series)
    
    # TC-DT-07: Correlation range
    def test_correlation_range(self, sample_df):
        corr_matrix, target_corr = analyze_transformed_correlations(sample_df.copy())
        # All correlation values should be in [-1, 1]
        assert (corr_matrix.values >= -1).all()
        assert (corr_matrix.values <= 1).all()
        assert (target_corr.values >= -1).all()
        assert (target_corr.values <= 1).all()

class TestNegativeCases:
    # TC-DT-N01: No numeric columns
    def test_no_numeric_columns(self):
        df_categorical = pd.DataFrame({
            'Category1': ['A', 'B', 'C'],
            'Category2': ['X', 'Y', 'Z'],
            'Resigned': [0, 1, 0]
        })
        # Current implementation fails with no numeric columns - expect ValueError
        with pytest.raises(ValueError):
            apply_scaling(df_categorical.copy())
    
    # TC-DT-N02: Missing target column
    def test_missing_target_column(self, sample_df):
        df_no_target = sample_df.drop('Resigned', axis=1)
        with pytest.raises(KeyError):
            apply_scaling(df_no_target.copy())
    
    # TC-DT-N03: All constant values
    def test_constant_values(self):
        df_constant = pd.DataFrame({
            'Constant1': [5.0] * 10,
            'Constant2': [10.0] * 10,
            'Resigned': [0, 1] * 5
        })
        # Should handle gracefully - scaler handles zero variance
        df_scaled, scaler = apply_scaling(df_constant.copy())
        assert isinstance(df_scaled, pd.DataFrame)
        assert 'Resigned' in df_scaled.columns
