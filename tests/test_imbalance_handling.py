"""
Unit tests for imbalance_handling module
"""
import pytest
import pandas as pd
from modules.imbalance_handling import analyze_imbalance, handle_imbalance, compare_resampling_methods

class TestAnalyzeImbalance:
    def test_returns_ratio(self, imbalanced_df):
        ratio = analyze_imbalance(imbalanced_df)
        assert isinstance(ratio, float)
        assert ratio >= 1.0

class TestHandleImbalance:
    def test_smote_increases_minority(self, imbalanced_df):
        df = imbalanced_df.copy()
        feature_cols = [col for col in df.columns if col != 'Resigned']
        
        df_res, X_res, y_res = handle_imbalance(
            df=df, feature_cols=feature_cols, method='smote', plot=False
        )
        
        assert len(df_res) >= len(df)
    
    def test_adasyn_method(self, imbalanced_df):
        df = imbalanced_df.copy()
        feature_cols = [col for col in df.columns if col != 'Resigned']
        
        df_res, X_res, y_res = handle_imbalance(
            df=df, feature_cols=feature_cols, method='adasyn', plot=False
        )
        
        assert len(df_res) >= len(df)

class TestCompareResamplingMethods:
    def test_returns_comparison_df(self, imbalanced_df):
        df = imbalanced_df.copy()
        X = df.drop('Resigned', axis=1)
        y = df['Resigned']
        
        result = compare_resampling_methods(X, y, plot=False)
        assert isinstance(result, pd.DataFrame)

class TestNegativeCases:
    def test_missing_parameters(self):
        with pytest.raises(TypeError):
            handle_imbalance()
    
    def test_single_class_only(self):
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [5, 4, 3, 2, 1],
            'Resigned': [0, 0, 0, 0, 0]
        })
        feature_cols = ['feature1', 'feature2']
        with pytest.raises(Exception):
            handle_imbalance(df=df, feature_cols=feature_cols, method='smote', plot=False)
