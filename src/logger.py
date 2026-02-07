"""
Centralized Logging Configuration for Stock Prediction Project
Provides consistent logging across all modules
"""

import logging
import sys
import os
from datetime import datetime
from typing import Optional


class StockPredictionLogger:
    """Centralized logger for the stock prediction project"""
    
    _instance: Optional['StockPredictionLogger'] = None
    _initialized: bool = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if StockPredictionLogger._initialized:
            return
        
        self.logger = logging.getLogger('stock_prediction')
        self.logger.setLevel(logging.DEBUG)
        
        # Prevent duplicate handlers
        if self.logger.handlers:
            return
        
        # Console handler - INFO level
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)
        
        # File handler - DEBUG level (optional, created on demand)
        self._file_handler = None
        
        StockPredictionLogger._initialized = True
    
    def enable_file_logging(self, log_dir: str = 'logs'):
        """Enable logging to file"""
        if self._file_handler:
            return
        
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(
            log_dir, 
            f'stock_prediction_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        )
        
        self._file_handler = logging.FileHandler(log_file)
        self._file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s'
        )
        self._file_handler.setFormatter(file_format)
        self.logger.addHandler(self._file_handler)
        self.logger.info(f"File logging enabled: {log_file}")
    
    def get_logger(self, name: str = None) -> logging.Logger:
        """Get a logger instance, optionally with a specific name"""
        if name:
            return logging.getLogger(f'stock_prediction.{name}')
        return self.logger


# Initialize singleton instance
_logger_instance = StockPredictionLogger()


def get_logger(name: str = None) -> logging.Logger:
    """
    Get a logger instance.
    
    Usage:
        from logger import get_logger
        logger = get_logger(__name__)
        logger.info("Message")
    """
    return _logger_instance.get_logger(name)


def enable_file_logging(log_dir: str = 'logs'):
    """Enable file logging"""
    _logger_instance.enable_file_logging(log_dir)


# Convenience functions for quick logging
def info(msg: str):
    """Log info message"""
    _logger_instance.logger.info(msg)

def debug(msg: str):
    """Log debug message"""
    _logger_instance.logger.debug(msg)

def warning(msg: str):
    """Log warning message"""
    _logger_instance.logger.warning(msg)

def error(msg: str):
    """Log error message"""
    _logger_instance.logger.error(msg)

def success(msg: str):
    """Log success message (info level with prefix)"""
    _logger_instance.logger.info(f"[OK] {msg}")


if __name__ == "__main__":
    # Demo logging
    logger = get_logger('demo')
    
    print("=" * 50)
    print("LOGGING DEMO")
    print("=" * 50)
    
    info("This is an info message")
    debug("This is a debug message (hidden in console)")
    warning("This is a warning message")
    error("This is an error message")
    success("This is a success message")
    
    # Enable file logging
    enable_file_logging()
    info("Now also logging to file")
