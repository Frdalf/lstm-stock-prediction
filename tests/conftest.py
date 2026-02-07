"""
Pytest Configuration and Shared Fixtures
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import shutil
import os
import sys

# Add src to path for all tests
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def pytest_configure(config):
    """Configure custom markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )


@pytest.fixture(scope='session')
def sample_stock_data():
    """
    Create realistic sample stock data for testing.
    Session-scoped for performance.
    """
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=300, freq='D')
    
    # Generate realistic price movement
    base_price = 150
    prices = [base_price]
    for _ in range(299):
        change = np.random.normal(0, 2)  # Daily change
        prices.append(max(50, prices[-1] + change))  # Ensure positive
    
    prices = np.array(prices)
    
    data = pd.DataFrame({
        'Date': dates,
        'Open': prices + np.random.uniform(-2, 2, 300),
        'High': prices + np.random.uniform(0, 5, 300),
        'Low': prices - np.random.uniform(0, 5, 300),
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, 300)
    })
    
    return data


@pytest.fixture
def temp_directory():
    """Create a temporary directory for test files"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def small_lstm_data():
    """
    Create small dataset for fast LSTM tests.
    Uses short sequences and few samples.
    """
    np.random.seed(42)
    sequence_length = 10
    n_samples = 50
    n_features = 1
    
    X = np.random.random((n_samples, sequence_length, n_features))
    y = np.random.random(n_samples)
    
    # Split into train/test
    split = 40
    return {
        'X_train': X[:split],
        'y_train': y[:split],
        'X_test': X[split:],
        'y_test': y[split:],
        'sequence_length': sequence_length,
        'n_features': n_features
    }
