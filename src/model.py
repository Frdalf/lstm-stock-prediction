"""
LSTM Model Module
Defines the LSTM neural network architecture for stock price prediction
"""

import numpy as np
import os
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from typing import Tuple, List, Optional
import matplotlib.pyplot as plt

from logger import get_logger

logger = get_logger(__name__)


class LSTMStockPredictor:
    """LSTM model for stock price prediction"""
    
    def __init__(
        self,
        sequence_length: int = 60,
        n_features: int = 1,
        lstm_units: List[int] = None,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001
    ):
        """
        Initialize LSTM predictor
        
        Args:
            sequence_length: Number of time steps in input
            n_features: Number of features per time step
            lstm_units: List of units for each LSTM layer
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.lstm_units = lstm_units or [50, 50, 50]
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = None
        self.history = None
    
    def build_model(self) -> Sequential:
        """
        Build the LSTM model architecture
        
        Returns:
            Compiled Keras Sequential model
        """
        model = Sequential()
        
        # Input layer
        model.add(Input(shape=(self.sequence_length, self.n_features)))
        
        # First LSTM layer with return sequences
        model.add(LSTM(
            units=self.lstm_units[0],
            return_sequences=True if len(self.lstm_units) > 1 else False
        ))
        model.add(Dropout(self.dropout_rate))
        
        # Additional LSTM layers
        for i, units in enumerate(self.lstm_units[1:], 1):
            return_sequences = i < len(self.lstm_units) - 1
            model.add(LSTM(units=units, return_sequences=return_sequences))
            model.add(Dropout(self.dropout_rate))
        
        # Output layer
        model.add(Dense(units=25, activation='relu'))
        model.add(Dense(units=1))
        
        # Compile model
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
        
        self.model = model
        logger.info("Model architecture:")
        model.summary(print_fn=logger.info)
        return model
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 32,
        model_path: str = "models/lstm_model.keras",
        patience: int = 10
    ) -> dict:
        """
        Train the LSTM model
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional, will split from train if not provided)
            y_val: Validation targets
            epochs: Number of training epochs
            batch_size: Batch size for training
            model_path: Path to save best model
            patience: Early stopping patience
            
        Returns:
            Training history
        """
        if self.model is None:
            self.build_model()
        
        # Create model directory if needed
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=model_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.0001,
                verbose=1
            )
        ]
        
        # Prepare validation data
        validation_data = None
        validation_split = 0.1
        
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
            validation_split = 0.0
        
        # Train model
        logger.info(f"Training model for {epochs} epochs with batch size {batch_size}...")
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Input features
            
        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() or load_model() first.")
        
        return self.model.predict(X, verbose=0)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Evaluate model on test data
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() or load_model() first.")
        
        loss, mae = self.model.evaluate(X_test, y_test, verbose=0)
        
        # Make predictions for additional metrics
        predictions = self.predict(X_test)
        
        # Calculate additional metrics
        mse = np.mean((predictions.flatten() - y_test) ** 2)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_test - predictions.flatten()) / y_test)) * 100
        
        metrics = {
            'loss': loss,
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'mape': mape
        }
        
        logger.info("=== Model Evaluation ===")
        logger.info(f"Loss (MSE): {loss:.6f}")
        logger.info(f"MAE: {mae:.6f}")
        logger.info(f"RMSE: {rmse:.6f}")
        logger.info(f"MAPE: {mape:.2f}%")
        
        return metrics
    
    def save_model(self, filepath: str = "models/lstm_model.keras"):
        """Save the model to file"""
        if self.model is None:
            raise ValueError("No model to save.")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_saved_model(self, filepath: str = "models/lstm_model.keras"):
        """Load model from file"""
        self.model = load_model(filepath)
        logger.info(f"Model loaded from {filepath}")
    
    def plot_training_history(self, save_path: str = None):
        """
        Plot training history
        
        Args:
            save_path: Optional path to save the plot
        """
        if self.history is None:
            raise ValueError("No training history. Train the model first.")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot loss
        axes[0].plot(self.history.history['loss'], label='Training Loss')
        axes[0].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0].set_title('Model Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot MAE
        axes[1].plot(self.history.history['mae'], label='Training MAE')
        axes[1].plot(self.history.history['val_mae'], label='Validation MAE')
        axes[1].set_title('Model MAE')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MAE')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Training history plot saved to {save_path}")
        
        plt.show()


class StackedLSTMPredictor(LSTMStockPredictor):
    """
    Stacked LSTM with more sophisticated architecture
    """
    
    def build_model(self) -> Sequential:
        """Build a more sophisticated stacked LSTM model"""
        from tensorflow.keras.layers import BatchNormalization
        
        model = Sequential()
        
        # Input layer
        model.add(Input(shape=(self.sequence_length, self.n_features)))
        
        # First LSTM block
        model.add(LSTM(units=128, return_sequences=True))
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout_rate))
        
        # Second LSTM block
        model.add(LSTM(units=64, return_sequences=True))
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout_rate))
        
        # Third LSTM block
        model.add(LSTM(units=32, return_sequences=False))
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout_rate))
        
        # Dense layers
        model.add(Dense(units=32, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(units=16, activation='relu'))
        model.add(Dense(units=1))
        
        # Compile
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
        
        self.model = model
        print(model.summary())
        return model


# Example usage
if __name__ == "__main__":
    # Example with dummy data
    sequence_length = 60
    n_features = 1
    n_samples = 1000
    
    # Create dummy data
    X_train = np.random.random((n_samples, sequence_length, n_features))
    y_train = np.random.random(n_samples)
    X_test = np.random.random((200, sequence_length, n_features))
    y_test = np.random.random(200)
    
    # Create and train model
    predictor = LSTMStockPredictor(
        sequence_length=sequence_length,
        n_features=n_features,
        lstm_units=[50, 50],
        dropout_rate=0.2
    )
    
    predictor.build_model()
    print("\nModel architecture built successfully!")
