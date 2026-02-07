"""
Unit Tests for DataLoader Module
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

from data_loader import DataLoader


class TestDataLoader:
    """Test cases for DataLoader class"""
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary directory for test data"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample stock data for testing"""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'Date': dates,
            'Open': np.random.uniform(100, 200, 100),
            'High': np.random.uniform(100, 200, 100),
            'Low': np.random.uniform(100, 200, 100),
            'Close': np.random.uniform(100, 200, 100),
            'Volume': np.random.randint(1000000, 10000000, 100)
        })
        return data
    
    def test_init_creates_directory(self, temp_data_dir):
        """Test that DataLoader creates data directory on init"""
        new_dir = os.path.join(temp_data_dir, 'new_data')
        loader = DataLoader(data_dir=new_dir)
        assert os.path.exists(new_dir)
    
    def test_save_data(self, temp_data_dir, sample_data):
        """Test saving data to CSV"""
        loader = DataLoader(data_dir=temp_data_dir)
        filepath = loader.save_data(sample_data, 'TEST')
        
        assert os.path.exists(filepath)
        assert filepath.endswith('TEST_data.csv')
    
    def test_save_data_with_dot_in_ticker(self, temp_data_dir, sample_data):
        """Test saving data with Indonesian stock ticker (has dot)"""
        loader = DataLoader(data_dir=temp_data_dir)
        filepath = loader.save_data(sample_data, 'BBCA.JK')
        
        assert os.path.exists(filepath)
        assert 'BBCA_JK_data.csv' in filepath
    
    def test_load_data(self, temp_data_dir, sample_data):
        """Test loading data from CSV"""
        loader = DataLoader(data_dir=temp_data_dir)
        loader.save_data(sample_data, 'TEST')
        
        loaded = loader.load_data('TEST')
        
        assert isinstance(loaded, pd.DataFrame)
        assert len(loaded) == len(sample_data)
        assert 'Date' in loaded.columns
        assert 'Close' in loaded.columns
    
    def test_load_data_date_type(self, temp_data_dir, sample_data):
        """Test that loaded data has proper datetime type"""
        loader = DataLoader(data_dir=temp_data_dir)
        loader.save_data(sample_data, 'TEST')
        
        loaded = loader.load_data('TEST')
        
        assert pd.api.types.is_datetime64_any_dtype(loaded['Date'])
    
    def test_load_data_file_not_found(self, temp_data_dir):
        """Test loading non-existent file raises error"""
        loader = DataLoader(data_dir=temp_data_dir)
        
        with pytest.raises(FileNotFoundError):
            loader.load_data('NONEXISTENT')
    
    def test_download_stock_data_invalid_ticker(self, temp_data_dir):
        """Test downloading invalid ticker raises error"""
        loader = DataLoader(data_dir=temp_data_dir)
        
        with pytest.raises(ValueError, match="No data found"):
            loader.download_stock_data('INVALIDTICKER12345', save=False)


class TestDataLoaderIntegration:
    """Integration tests that may require network access"""
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary directory for test data"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.mark.slow
    @pytest.mark.skip(reason="Skipped by default - may fail due to API rate limiting")
    def test_download_real_stock(self, temp_data_dir):
        """Test downloading real stock data (requires internet)"""
        loader = DataLoader(data_dir=temp_data_dir)
        
        data = loader.download_stock_data('AAPL', period='1mo', save=False)
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0
        assert 'Close' in data.columns
        assert 'Date' in data.columns


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
