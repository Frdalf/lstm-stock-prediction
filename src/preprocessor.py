"""
Preprocessor Module
Handles data preprocessing for LSTM model including normalization and sequence creation
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List
import pickle
import os

from logger import get_logger

logger = get_logger(__name__)


class StockPreprocessor:
    """Class to handle stock data preprocessing for LSTM"""
    
    def __init__(self, sequence_length: int = 60, feature_columns: List[str] = None):
        """
        Initialize preprocessor
        
        Args:
            sequence_length: Number of time steps to look back (default 60 days)
            feature_columns: List of columns to use as features (default: ['Close'])
        """
        self.sequence_length = sequence_length
        self.feature_columns = feature_columns or ['Close']
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.target_scaler = MinMaxScaler(feature_range=(0, 1))
        self.is_fitted = False
    
    def prepare_data(
        self,
        data: pd.DataFrame,
        target_column: str = 'Close',
        train_ratio: float = 0.8
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for LSTM training
        
        Args:
            data: DataFrame with stock data
            target_column: Column to predict
            train_ratio: Ratio of data for training (default 0.8)
            
        Returns:
            Tuple of (X_train, y_train, X_test, y_test)
        """
        # Extract feature columns
        if target_column not in self.feature_columns:
            self.feature_columns = [target_column] + [c for c in self.feature_columns if c != target_column]
        
        # Get feature data
        feature_data = data[self.feature_columns].values
        
        # Normalize data
        scaled_data = self.scaler.fit_transform(feature_data)
        self.is_fitted = True
        
        # Fit target scaler separately for inverse transform later
        self.target_scaler.fit(data[[target_column]].values)
        
        # Create sequences
        X, y = self.create_sequences(scaled_data)
        
        # Split into train and test
        train_size = int(len(X) * train_ratio)
        
        X_train = X[:train_size]
        y_train = y[:train_size]
        X_test = X[train_size:]
        y_test = y[train_size:]
        
        logger.info(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
        logger.info(f"Testing data shape: X={X_test.shape}, y={y_test.shape}")
        
        return X_train, y_train, X_test, y_test
    
    def create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM input
        
        Args:
            data: Normalized data array
            
        Returns:
            Tuple of (X sequences, y targets)
        """
        X, y = [], []
        
        for i in range(self.sequence_length, len(data)):
            X.append(data[i - self.sequence_length:i])
            y.append(data[i, 0])  # Target is first column (usually Close price)
        
        return np.array(X), np.array(y)
    
    def inverse_transform_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """
        Convert scaled predictions back to original scale
        
        Args:
            predictions: Scaled predictions
            
        Returns:
            Predictions in original scale
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted. Call prepare_data first.")
        
        # Reshape if needed
        if len(predictions.shape) == 1:
            predictions = predictions.reshape(-1, 1)
        
        return self.target_scaler.inverse_transform(predictions)
    
    def inverse_transform_actual(self, actual: np.ndarray) -> np.ndarray:
        """
        Convert scaled actual values back to original scale
        
        Args:
            actual: Scaled actual values
            
        Returns:
            Actual values in original scale
        """
        if len(actual.shape) == 1:
            actual = actual.reshape(-1, 1)
        
        return self.target_scaler.inverse_transform(actual)
    
    def prepare_prediction_data(self, data: pd.DataFrame) -> np.ndarray:
        """
        Prepare the most recent data for prediction
        
        Args:
            data: DataFrame with stock data
            
        Returns:
            Prepared sequence for prediction
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted. Call prepare_data first.")
        
        # Get last sequence_length days
        feature_data = data[self.feature_columns].values[-self.sequence_length:]
        
        # Scale data
        scaled_data = self.scaler.transform(feature_data)
        
        # Reshape for LSTM (1, sequence_length, n_features)
        return scaled_data.reshape(1, self.sequence_length, len(self.feature_columns))
    
    def save_scaler(self, filepath: str = "models/scaler.pkl"):
        """Save scaler to file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'scaler': self.scaler,
                'target_scaler': self.target_scaler,
                'sequence_length': self.sequence_length,
                'feature_columns': self.feature_columns
            }, f)
        logger.info(f"Scaler saved to {filepath}")
    
    def load_scaler(self, filepath: str = "models/scaler.pkl"):
        """Load scaler from file"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.scaler = data['scaler']
            self.target_scaler = data['target_scaler']
            self.sequence_length = data['sequence_length']
            self.feature_columns = data['feature_columns']
            self.is_fitted = True
        logger.info(f"Scaler loaded from {filepath}")


class MultiFeaturePreprocessor(StockPreprocessor):
    """Extended preprocessor for multi-feature LSTM"""
    
    def __init__(
        self,
        sequence_length: int = 60,
        feature_columns: List[str] = None
    ):
        """
        Initialize multi-feature preprocessor
        
        Args:
            sequence_length: Number of time steps
            feature_columns: Columns to use as features
        """
        default_features = ['Open', 'High', 'Low', 'Close', 'Volume']
        super().__init__(
            sequence_length=sequence_length,
            feature_columns=feature_columns or default_features
        )
    
    def add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators as additional features
        
        Args:
            data: Original stock data
            
        Returns:
            DataFrame with added indicators
        """
        df = data.copy()
        
        # Moving Averages
        df['MA_7'] = df['Close'].rolling(window=7).mean()
        df['MA_21'] = df['Close'].rolling(window=21).mean()
        df['MA_50'] = df['Close'].rolling(window=50).mean()
        
        # Exponential Moving Average
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        
        # Price Rate of Change
        df['ROC'] = df['Close'].pct_change(periods=10) * 100
        
        # Volatility
        df['Volatility'] = df['Close'].rolling(window=20).std()
        
        # Drop NaN values created by indicators
        df = df.dropna()
        
        logger.info(f"Added technical indicators. New shape: {df.shape}")
        return df


# Example usage
if __name__ == "__main__":
    import sys
    sys.path.append('..')
    from data_loader import DataLoader
    
    # Load data
    loader = DataLoader()
    data = loader.download_stock_data("AAPL", period="2y", save=False)
    
    # Basic preprocessing
    preprocessor = StockPreprocessor(sequence_length=60)
    X_train, y_train, X_test, y_test = preprocessor.prepare_data(data)
    
    print("\n--- Multi-feature preprocessing with technical indicators ---")
    multi_preprocessor = MultiFeaturePreprocessor(sequence_length=60)
    data_with_indicators = multi_preprocessor.add_technical_indicators(data)
    print("\nFeatures available:")
    print(data_with_indicators.columns.tolist())
