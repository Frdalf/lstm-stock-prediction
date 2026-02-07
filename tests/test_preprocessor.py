"""
Unit Tests for Preprocessor Module
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys
import tempfile
import shutil

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from preprocessor import StockPreprocessor, MultiFeaturePreprocessor


class TestStockPreprocessor:
    """Test cases for StockPreprocessor class"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample stock data for testing"""
        dates = pd.date_range(start='2024-01-01', periods=200, freq='D')
        np.random.seed(42)
        
        # Create realistic price data
        base_price = 100
        prices = [base_price]
        for _ in range(199):
            change = np.random.normal(0, 2)
            prices.append(prices[-1] + change)
        
        data = pd.DataFrame({
            'Date': dates,
            'Open': prices,
            'High': [p + np.random.uniform(0, 5) for p in prices],
            'Low': [p - np.random.uniform(0, 5) for p in prices],
            'Close': prices,
            'Volume': np.random.randint(1000000, 10000000, 200)
        })
        return data
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_init_default_values(self):
        """Test default initialization values"""
        preprocessor = StockPreprocessor()
        
        assert preprocessor.sequence_length == 60
        assert preprocessor.feature_columns == ['Close']
        assert preprocessor.is_fitted == False
    
    def test_init_custom_values(self):
        """Test custom initialization values"""
        preprocessor = StockPreprocessor(
            sequence_length=30,
            feature_columns=['Open', 'Close']
        )
        
        assert preprocessor.sequence_length == 30
        assert preprocessor.feature_columns == ['Open', 'Close']
    
    def test_prepare_data_shapes(self, sample_data):
        """Test that prepare_data returns correct shapes"""
        preprocessor = StockPreprocessor(sequence_length=60)
        X_train, y_train, X_test, y_test = preprocessor.prepare_data(
            sample_data, train_ratio=0.8
        )
        
        # Total sequences = len(data) - sequence_length = 200 - 60 = 140
        # Train = 80% of 140 = 112
        # Test = 20% of 140 = 28
        
        assert X_train.shape[1] == 60  # sequence_length
        assert X_train.shape[2] == 1   # n_features
        assert len(y_train) == len(X_train)
        assert len(y_test) == len(X_test)
    
    def test_prepare_data_sets_fitted(self, sample_data):
        """Test that prepare_data sets is_fitted flag"""
        preprocessor = StockPreprocessor()
        
        assert preprocessor.is_fitted == False
        
        preprocessor.prepare_data(sample_data)
        
        assert preprocessor.is_fitted == True
    
    def test_create_sequences(self, sample_data):
        """Test sequence creation"""
        preprocessor = StockPreprocessor(sequence_length=10)
        
        # Use simple array for testing
        data = np.arange(100).reshape(-1, 1)
        X, y = preprocessor.create_sequences(data)
        
        # Should have 90 sequences (100 - 10)
        assert len(X) == 90
        assert len(y) == 90
        
        # First sequence should be 0-9, target should be 10
        np.testing.assert_array_equal(X[0].flatten(), np.arange(10))
        assert y[0] == 10
    
    def test_inverse_transform_predictions(self, sample_data):
        """Test inverse transform of predictions"""
        preprocessor = StockPreprocessor()
        preprocessor.prepare_data(sample_data)
        
        # Create scaled predictions (between 0 and 1)
        scaled_preds = np.array([[0.5], [0.7], [0.3]])
        
        original_preds = preprocessor.inverse_transform_predictions(scaled_preds)
        
        # Should be back in original price range
        assert original_preds.min() >= sample_data['Close'].min() - 10
        assert original_preds.max() <= sample_data['Close'].max() + 10
    
    def test_inverse_transform_not_fitted(self):
        """Test inverse transform raises error when not fitted"""
        preprocessor = StockPreprocessor()
        
        with pytest.raises(ValueError, match="not fitted"):
            preprocessor.inverse_transform_predictions(np.array([[0.5]]))
    
    def test_prepare_prediction_data(self, sample_data):
        """Test preparing data for prediction"""
        preprocessor = StockPreprocessor(sequence_length=60)
        preprocessor.prepare_data(sample_data)
        
        pred_data = preprocessor.prepare_prediction_data(sample_data)
        
        assert pred_data.shape == (1, 60, 1)
    
    def test_prepare_prediction_data_not_fitted(self, sample_data):
        """Test prepare_prediction_data raises error when not fitted"""
        preprocessor = StockPreprocessor()
        
        with pytest.raises(ValueError, match="not fitted"):
            preprocessor.prepare_prediction_data(sample_data)
    
    def test_save_and_load_scaler(self, sample_data, temp_dir):
        """Test saving and loading scaler"""
        preprocessor = StockPreprocessor(sequence_length=30)
        preprocessor.prepare_data(sample_data)
        
        scaler_path = os.path.join(temp_dir, 'models', 'scaler.pkl')
        preprocessor.save_scaler(scaler_path)
        
        # Create new preprocessor and load
        new_preprocessor = StockPreprocessor()
        new_preprocessor.load_scaler(scaler_path)
        
        assert new_preprocessor.is_fitted == True
        assert new_preprocessor.sequence_length == 30
    
    def test_normalization_range(self, sample_data):
        """Test that scaled data is in [0, 1] range"""
        preprocessor = StockPreprocessor()
        X_train, y_train, _, _ = preprocessor.prepare_data(sample_data)
        
        # All values should be between 0 and 1 (with small tolerance for floating point)
        assert X_train.min() >= -1e-10
        assert X_train.max() <= 1 + 1e-10
        assert y_train.min() >= -1e-10
        assert y_train.max() <= 1 + 1e-10


class TestMultiFeaturePreprocessor:
    """Test cases for MultiFeaturePreprocessor class"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample stock data with all required columns"""
        dates = pd.date_range(start='2024-01-01', periods=200, freq='D')
        np.random.seed(42)
        
        data = pd.DataFrame({
            'Date': dates,
            'Open': np.random.uniform(100, 200, 200),
            'High': np.random.uniform(100, 200, 200),
            'Low': np.random.uniform(100, 200, 200),
            'Close': np.random.uniform(100, 200, 200),
            'Volume': np.random.randint(1000000, 10000000, 200)
        })
        return data
    
    def test_init_default_features(self):
        """Test default feature columns"""
        preprocessor = MultiFeaturePreprocessor()
        
        expected_features = ['Open', 'High', 'Low', 'Close', 'Volume']
        assert preprocessor.feature_columns == expected_features
    
    def test_add_technical_indicators(self, sample_data):
        """Test adding technical indicators"""
        preprocessor = MultiFeaturePreprocessor()
        
        data_with_indicators = preprocessor.add_technical_indicators(sample_data)
        
        # Check that new columns are added
        assert 'MA_7' in data_with_indicators.columns
        assert 'MA_21' in data_with_indicators.columns
        assert 'RSI' in data_with_indicators.columns
        assert 'MACD' in data_with_indicators.columns
        assert 'BB_Upper' in data_with_indicators.columns
        assert 'BB_Lower' in data_with_indicators.columns
    
    def test_add_technical_indicators_no_nan(self, sample_data):
        """Test that NaN values are dropped after adding indicators"""
        preprocessor = MultiFeaturePreprocessor()
        
        data_with_indicators = preprocessor.add_technical_indicators(sample_data)
        
        # Should have no NaN values
        assert data_with_indicators.isna().sum().sum() == 0
    
    def test_add_technical_indicators_length(self, sample_data):
        """Test that length is reduced after adding indicators (due to rolling windows)"""
        preprocessor = MultiFeaturePreprocessor()
        
        original_length = len(sample_data)
        data_with_indicators = preprocessor.add_technical_indicators(sample_data)
        
        # Length should be less due to NaN dropping from rolling calculations
        assert len(data_with_indicators) < original_length


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
