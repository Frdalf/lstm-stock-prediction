"""
Unit Tests for LSTM Model Module
"""

import pytest
import numpy as np
import os
import sys
import tempfile
import shutil

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from model import LSTMStockPredictor, StackedLSTMPredictor


class TestLSTMStockPredictor:
    """Test cases for LSTMStockPredictor class"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data"""
        np.random.seed(42)
        sequence_length = 60
        n_samples = 200
        n_features = 1
        
        X = np.random.random((n_samples, sequence_length, n_features))
        y = np.random.random(n_samples)
        
        return X, y
    
    @pytest.fixture
    def small_sample_data(self):
        """Create smaller sample for faster tests"""
        np.random.seed(42)
        sequence_length = 10
        n_samples = 50
        n_features = 1
        
        X = np.random.random((n_samples, sequence_length, n_features))
        y = np.random.random(n_samples)
        
        return X, y
    
    def test_init_default_values(self):
        """Test default initialization values"""
        predictor = LSTMStockPredictor()
        
        assert predictor.sequence_length == 60
        assert predictor.n_features == 1
        assert predictor.lstm_units == [50, 50, 50]
        assert predictor.dropout_rate == 0.2
        assert predictor.learning_rate == 0.001
        assert predictor.model is None
    
    def test_init_custom_values(self):
        """Test custom initialization values"""
        predictor = LSTMStockPredictor(
            sequence_length=30,
            n_features=5,
            lstm_units=[100, 50],
            dropout_rate=0.3,
            learning_rate=0.0001
        )
        
        assert predictor.sequence_length == 30
        assert predictor.n_features == 5
        assert predictor.lstm_units == [100, 50]
        assert predictor.dropout_rate == 0.3
        assert predictor.learning_rate == 0.0001
    
    def test_build_model(self):
        """Test model building"""
        predictor = LSTMStockPredictor(
            sequence_length=60,
            n_features=1,
            lstm_units=[50, 50]
        )
        
        model = predictor.build_model()
        
        assert predictor.model is not None
        assert model is not None
        
        # Check output shape
        assert model.output_shape == (None, 1)
    
    def test_build_model_input_shape(self):
        """Test that model has correct input shape"""
        predictor = LSTMStockPredictor(
            sequence_length=30,
            n_features=5,
            lstm_units=[50]
        )
        
        predictor.build_model()
        
        # Input shape should be (None, sequence_length, n_features)
        assert predictor.model.input_shape == (None, 30, 5)
    
    def test_train_creates_history(self, small_sample_data, temp_dir):
        """Test that training creates history"""
        X, y = small_sample_data
        X_train, X_val = X[:40], X[40:]
        y_train, y_val = y[:40], y[40:]
        
        predictor = LSTMStockPredictor(
            sequence_length=10,
            n_features=1,
            lstm_units=[16]  # Small for faster test
        )
        predictor.build_model()
        
        model_path = os.path.join(temp_dir, 'models', 'test_model.keras')
        history = predictor.train(
            X_train, y_train,
            X_val=X_val, y_val=y_val,
            epochs=2,
            batch_size=16,
            model_path=model_path,
            patience=5
        )
        
        assert 'loss' in history
        assert 'val_loss' in history
        assert len(history['loss']) > 0
    
    def test_predict_without_training(self):
        """Test that predict raises error without training"""
        predictor = LSTMStockPredictor()
        
        X = np.random.random((10, 60, 1))
        
        with pytest.raises(ValueError, match="Model not trained"):
            predictor.predict(X)
    
    def test_predict_after_build(self):
        """Test prediction after building model"""
        predictor = LSTMStockPredictor(
            sequence_length=10,
            n_features=1,
            lstm_units=[16]
        )
        predictor.build_model()
        
        X = np.random.random((5, 10, 1))
        predictions = predictor.predict(X)
        
        assert predictions.shape == (5, 1)
    
    def test_evaluate_without_training(self):
        """Test that evaluate raises error without training"""
        predictor = LSTMStockPredictor()
        
        X = np.random.random((10, 60, 1))
        y = np.random.random(10)
        
        with pytest.raises(ValueError, match="Model not trained"):
            predictor.evaluate(X, y)
    
    def test_evaluate_returns_metrics(self, small_sample_data, temp_dir):
        """Test that evaluate returns correct metrics"""
        X, y = small_sample_data
        
        predictor = LSTMStockPredictor(
            sequence_length=10,
            n_features=1,
            lstm_units=[16]
        )
        predictor.build_model()
        
        model_path = os.path.join(temp_dir, 'models', 'test_model.keras')
        predictor.train(
            X[:40], y[:40],
            epochs=1,
            batch_size=16,
            model_path=model_path
        )
        
        metrics = predictor.evaluate(X[40:], y[40:])
        
        assert 'loss' in metrics
        assert 'mae' in metrics
        assert 'rmse' in metrics
        assert 'mape' in metrics
    
    def test_save_model(self, temp_dir):
        """Test saving model"""
        predictor = LSTMStockPredictor(
            sequence_length=10,
            lstm_units=[16]
        )
        predictor.build_model()
        
        model_path = os.path.join(temp_dir, 'models', 'saved_model.keras')
        predictor.save_model(model_path)
        
        assert os.path.exists(model_path)
    
    def test_save_model_without_building(self, temp_dir):
        """Test that save raises error without model"""
        predictor = LSTMStockPredictor()
        
        model_path = os.path.join(temp_dir, 'model.keras')
        
        with pytest.raises(ValueError, match="No model to save"):
            predictor.save_model(model_path)
    
    def test_load_saved_model(self, temp_dir):
        """Test loading saved model"""
        # Save a model first
        predictor = LSTMStockPredictor(
            sequence_length=10,
            lstm_units=[16]
        )
        predictor.build_model()
        
        model_path = os.path.join(temp_dir, 'models', 'model_to_load.keras')
        predictor.save_model(model_path)
        
        # Load in new predictor
        new_predictor = LSTMStockPredictor()
        new_predictor.load_saved_model(model_path)
        
        assert new_predictor.model is not None
    
    def test_loaded_model_can_predict(self, temp_dir):
        """Test that loaded model can make predictions"""
        # Save a model
        predictor = LSTMStockPredictor(
            sequence_length=10,
            n_features=1,
            lstm_units=[16]
        )
        predictor.build_model()
        
        model_path = os.path.join(temp_dir, 'models', 'model.keras')
        predictor.save_model(model_path)
        
        # Load and predict
        new_predictor = LSTMStockPredictor()
        new_predictor.load_saved_model(model_path)
        
        X = np.random.random((3, 10, 1))
        predictions = new_predictor.predict(X)
        
        assert predictions.shape == (3, 1)


class TestStackedLSTMPredictor:
    """Test cases for StackedLSTMPredictor class"""
    
    def test_inherits_from_lstm_predictor(self):
        """Test that StackedLSTMPredictor inherits from LSTMStockPredictor"""
        predictor = StackedLSTMPredictor()
        
        assert isinstance(predictor, LSTMStockPredictor)
    
    def test_build_model(self):
        """Test building stacked model"""
        predictor = StackedLSTMPredictor(
            sequence_length=30,
            n_features=1
        )
        
        model = predictor.build_model()
        
        assert predictor.model is not None
        assert model.output_shape == (None, 1)
    
    def test_stacked_model_has_batch_normalization(self):
        """Test that stacked model uses batch normalization"""
        predictor = StackedLSTMPredictor()
        predictor.build_model()
        
        layer_names = [layer.name for layer in predictor.model.layers]
        
        # Should have batch normalization layers
        has_batch_norm = any('batch_normalization' in name for name in layer_names)
        assert has_batch_norm


class TestModelArchitecture:
    """Test model architecture properties"""
    
    def test_different_lstm_layers(self):
        """Test model with different number of LSTM layers"""
        configs = [
            [50],           # 1 layer
            [50, 50],       # 2 layers
            [100, 50, 25],  # 3 layers
        ]
        
        for lstm_units in configs:
            predictor = LSTMStockPredictor(
                sequence_length=10,
                lstm_units=lstm_units
            )
            model = predictor.build_model()
            
            # Count LSTM layers
            lstm_layers = [l for l in model.layers if 'lstm' in l.name.lower()]
            assert len(lstm_layers) == len(lstm_units)
    
    def test_dropout_layers_present(self):
        """Test that dropout layers are included"""
        predictor = LSTMStockPredictor(
            sequence_length=10,
            lstm_units=[50, 50],
            dropout_rate=0.2
        )
        predictor.build_model()
        
        dropout_layers = [l for l in predictor.model.layers if 'dropout' in l.name.lower()]
        
        # Should have dropout after each LSTM layer
        assert len(dropout_layers) >= 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
