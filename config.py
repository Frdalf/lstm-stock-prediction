"""
Centralized Configuration for Stock Prediction Project
All hyperparameters and settings in one place
"""

import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class ModelConfig:
    """LSTM Model Configuration"""
    sequence_length: int = 60          # Days to look back
    lstm_units: List[int] = field(default_factory=lambda: [50, 50, 50])
    dropout_rate: float = 0.2
    learning_rate: float = 0.001
    dense_units: int = 25


@dataclass
class TrainingConfig:
    """Training Configuration"""
    epochs: int = 50
    batch_size: int = 32
    train_ratio: float = 0.8           # 80% train, 20% test
    patience: int = 15                 # Early stopping patience
    min_lr: float = 0.0001             # Minimum learning rate


@dataclass
class DataConfig:
    """Data Configuration"""
    data_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    models_dir: str = "models"
    results_dir: str = "results"
    default_period: str = "2y"         # Default download period


@dataclass
class Config:
    """Main Configuration Class"""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    # Default tickers
    default_ticker: str = "AAPL"
    
    # Indonesian stock tickers
    indonesian_tickers: List[str] = field(default_factory=lambda: [
        "BBCA.JK", "BBRI.JK", "TLKM.JK", "ASII.JK", "BMRI.JK"
    ])
    
    # US stock tickers
    us_tickers: List[str] = field(default_factory=lambda: [
        "AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "META", "NVDA"
    ])
    
    def __post_init__(self):
        """Create directories if they don't exist"""
        for dir_path in [self.data.data_dir, self.data.processed_dir, 
                         self.data.models_dir, self.data.results_dir]:
            os.makedirs(dir_path, exist_ok=True)
    
    def get_model_path(self, ticker: str) -> str:
        """Get model file path for a ticker"""
        safe_ticker = ticker.replace(".", "_")
        return os.path.join(self.data.models_dir, f"lstm_{safe_ticker}.keras")
    
    def get_scaler_path(self, ticker: str) -> str:
        """Get scaler file path for a ticker"""
        safe_ticker = ticker.replace(".", "_")
        return os.path.join(self.data.models_dir, f"scaler_{safe_ticker}.pkl")
    
    def is_indonesian_stock(self, ticker: str) -> bool:
        """Check if ticker is Indonesian stock"""
        return ".JK" in ticker.upper()
    
    def get_currency(self, ticker: str) -> str:
        """Get currency symbol for ticker"""
        return "Rp" if self.is_indonesian_stock(ticker) else "$"
    
    def get_currency_format(self, ticker: str) -> str:
        """Get currency format string for ticker"""
        return "{:,.0f}" if self.is_indonesian_stock(ticker) else "{:.2f}"


# Global config instance - import this in other modules
config = Config()


# Quick access shortcuts
MODEL = config.model
TRAINING = config.training
DATA = config.data


if __name__ == "__main__":
    # Print configuration for verification
    print("=" * 50)
    print("STOCK PREDICTION CONFIGURATION")
    print("=" * 50)
    
    print("\n[Model Config]")
    print(f"  Sequence Length: {MODEL.sequence_length}")
    print(f"  LSTM Units: {MODEL.lstm_units}")
    print(f"  Dropout Rate: {MODEL.dropout_rate}")
    print(f"  Learning Rate: {MODEL.learning_rate}")
    
    print("\n[Training Config]")
    print(f"  Epochs: {TRAINING.epochs}")
    print(f"  Batch Size: {TRAINING.batch_size}")
    print(f"  Train Ratio: {TRAINING.train_ratio}")
    print(f"  Early Stopping Patience: {TRAINING.patience}")
    
    print("\n[Data Config]")
    print(f"  Data Directory: {DATA.data_dir}")
    print(f"  Models Directory: {DATA.models_dir}")
    print(f"  Results Directory: {DATA.results_dir}")
    
    print("\n[Tickers]")
    print(f"  Default: {config.default_ticker}")
    print(f"  US: {', '.join(config.us_tickers)}")
    print(f"  Indonesian: {', '.join(config.indonesian_tickers)}")
