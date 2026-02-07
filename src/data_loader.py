"""
Data Loader Module
Handles downloading and loading stock data using yfinance
"""

import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta

from logger import get_logger

# Initialize logger for this module
logger = get_logger(__name__)


class DataLoader:
    """Class to handle stock data downloading and loading"""
    
    def __init__(self, data_dir: str = "data/raw"):
        """
        Initialize DataLoader
        
        Args:
            data_dir: Directory to save/load raw data
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def download_stock_data(
        self,
        ticker: str,
        start_date: str = None,
        end_date: str = None,
        period: str = "5y",
        save: bool = True
    ) -> pd.DataFrame:
        """
        Download stock data from Yahoo Finance
        
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL', 'BBCA.JK')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            period: Period to download if dates not specified (e.g., '1y', '5y', 'max')
            save: Whether to save the data to CSV
            
        Returns:
            DataFrame with stock data
        """
        logger.info(f"Downloading data for {ticker}...")
        
        if start_date and end_date:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        else:
            data = yf.download(ticker, period=period, progress=False)
        
        if data.empty:
            raise ValueError(f"No data found for ticker: {ticker}")
        
        # Flatten multi-level columns if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        # Reset index to have Date as a column
        data = data.reset_index()
        
        logger.info(f"Downloaded {len(data)} rows of data")
        logger.info(f"Date range: {data['Date'].min()} to {data['Date'].max()}")
        
        if save:
            self.save_data(data, ticker)
        
        return data
    
    def save_data(self, data: pd.DataFrame, ticker: str) -> str:
        """
        Save data to CSV file
        
        Args:
            data: DataFrame to save
            ticker: Ticker symbol for filename
            
        Returns:
            Path to saved file
        """
        filename = f"{ticker.replace('.', '_')}_data.csv"
        filepath = os.path.join(self.data_dir, filename)
        data.to_csv(filepath, index=False)
        logger.info(f"Data saved to {filepath}")
        return filepath
    
    def load_data(self, ticker: str) -> pd.DataFrame:
        """
        Load data from CSV file
        
        Args:
            ticker: Ticker symbol to load
            
        Returns:
            DataFrame with stock data
        """
        filename = f"{ticker.replace('.', '_')}_data.csv"
        filepath = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        data = pd.read_csv(filepath)
        data['Date'] = pd.to_datetime(data['Date'])
        logger.info(f"Loaded {len(data)} rows from {filepath}")
        return data
    
    def get_multiple_stocks(
        self,
        tickers: list,
        start_date: str = None,
        end_date: str = None,
        period: str = "5y"
    ) -> dict:
        """
        Download data for multiple stocks
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date
            end_date: End date
            period: Period if dates not specified
            
        Returns:
            Dictionary with ticker as key and DataFrame as value
        """
        stock_data = {}
        for ticker in tickers:
            try:
                stock_data[ticker] = self.download_stock_data(
                    ticker, start_date, end_date, period
                )
            except Exception as e:
                logger.error(f"Error downloading {ticker}: {e}")
        return stock_data


# Example usage
if __name__ == "__main__":
    loader = DataLoader()
    
    # Download Indonesian stock (BCA)
    # bbca_data = loader.download_stock_data("BBCA.JK")
    
    # Download US stock (Apple)
    aapl_data = loader.download_stock_data("AAPL")
    print("\nSample data:")
    print(aapl_data.head())
    print("\nData info:")
    print(aapl_data.info())
