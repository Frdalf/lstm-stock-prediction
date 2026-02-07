"""
Visualizer Module
Handles all visualization tasks for stock price prediction
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Optional, List
import os


class StockVisualizer:
    """Class to handle stock data visualization"""
    
    def __init__(self, style: str = 'seaborn-v0_8-whitegrid', figsize: tuple = (14, 7)):
        """
        Initialize visualizer
        
        Args:
            style: Matplotlib style to use
            figsize: Default figure size
        """
        try:
            plt.style.use(style)
        except:
            plt.style.use('seaborn-v0_8')
        self.figsize = figsize
        self.colors = {
            'actual': '#2E86AB',
            'predicted': '#E94F37',
            'train': '#1B998B',
            'test': '#F46036',
            'ma_short': '#FF6B6B',
            'ma_long': '#4ECDC4'
        }
    
    def plot_stock_data(
        self,
        data: pd.DataFrame,
        columns: List[str] = None,
        title: str = "Stock Price History",
        save_path: str = None
    ):
        """
        Plot stock price history
        
        Args:
            data: DataFrame with stock data
            columns: Columns to plot (default: Close)
            title: Plot title
            save_path: Path to save figure
        """
        columns = columns or ['Close']
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        for col in columns:
            if col in data.columns:
                ax.plot(data['Date'], data[col], label=col, linewidth=1.5)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price (USD)', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def plot_predictions(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
        dates: Optional[np.ndarray] = None,
        title: str = "Actual vs Predicted Stock Prices",
        save_path: str = None
    ):
        """
        Plot actual vs predicted prices
        
        Args:
            actual: Actual price values
            predicted: Predicted price values
            dates: Date values for x-axis
            title: Plot title
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        x_axis = dates if dates is not None else np.arange(len(actual))
        
        ax.plot(x_axis, actual, color=self.colors['actual'], 
                label='Actual Price', linewidth=2)
        ax.plot(x_axis, predicted, color=self.colors['predicted'], 
                label='Predicted Price', linewidth=2, linestyle='--')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Time' if dates is None else 'Date', fontsize=12)
        ax.set_ylabel('Price (USD)', fontsize=12)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        if dates is not None:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def plot_train_test_split(
        self,
        train_data: np.ndarray,
        test_actual: np.ndarray,
        test_predicted: np.ndarray,
        title: str = "Train Data & Test Predictions",
        save_path: str = None
    ):
        """
        Plot training data along with test predictions
        
        Args:
            train_data: Training price data
            test_actual: Actual test prices
            test_predicted: Predicted test prices
            title: Plot title
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=(16, 8))
        
        # Create x-axis indices
        train_x = np.arange(len(train_data))
        test_x = np.arange(len(train_data), len(train_data) + len(test_actual))
        
        # Plot training data
        ax.plot(train_x, train_data, color=self.colors['train'], 
                label='Training Data', linewidth=1.5)
        
        # Plot actual test data
        ax.plot(test_x, test_actual, color=self.colors['actual'], 
                label='Actual Price (Test)', linewidth=2)
        
        # Plot predicted test data
        ax.plot(test_x, test_predicted, color=self.colors['predicted'], 
                label='Predicted Price (Test)', linewidth=2, linestyle='--')
        
        # Add vertical line to show train/test split
        ax.axvline(x=len(train_data), color='gray', linestyle=':', 
                   linewidth=2, label='Train/Test Split')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Time Steps', fontsize=12)
        ax.set_ylabel('Price (USD)', fontsize=12)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def plot_prediction_error(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
        title: str = "Prediction Error Analysis",
        save_path: str = None
    ):
        """
        Plot prediction error analysis
        
        Args:
            actual: Actual values
            predicted: Predicted values
            title: Plot title
            save_path: Path to save figure
        """
        errors = predicted.flatten() - actual.flatten()
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Error distribution
        axes[0, 0].hist(errors, bins=50, color=self.colors['predicted'], 
                        alpha=0.7, edgecolor='white')
        axes[0, 0].axvline(x=0, color='black', linestyle='--', linewidth=2)
        axes[0, 0].set_title('Error Distribution', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Prediction Error')
        axes[0, 0].set_ylabel('Frequency')
        
        # Error over time
        axes[0, 1].plot(errors, color=self.colors['predicted'], alpha=0.7)
        axes[0, 1].axhline(y=0, color='black', linestyle='--', linewidth=2)
        axes[0, 1].set_title('Error Over Time', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Time Steps')
        axes[0, 1].set_ylabel('Prediction Error')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Actual vs Predicted scatter
        axes[1, 0].scatter(actual, predicted.flatten(), alpha=0.5, 
                          color=self.colors['actual'], s=20)
        min_val = min(actual.min(), predicted.min())
        max_val = max(actual.max(), predicted.max())
        axes[1, 0].plot([min_val, max_val], [min_val, max_val], 
                        'r--', linewidth=2, label='Perfect Prediction')
        axes[1, 0].set_title('Actual vs Predicted', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Actual Price')
        axes[1, 0].set_ylabel('Predicted Price')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Percentage error
        pct_errors = (errors / actual) * 100
        axes[1, 1].plot(pct_errors, color=self.colors['test'], alpha=0.7)
        axes[1, 1].axhline(y=0, color='black', linestyle='--', linewidth=2)
        axes[1, 1].set_title('Percentage Error Over Time', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Time Steps')
        axes[1, 1].set_ylabel('Error (%)')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = (f"Mean Error: {errors.mean():.4f}\n"
                     f"Std Error: {errors.std():.4f}\n"
                     f"MAE: {np.abs(errors).mean():.4f}\n"
                     f"MAPE: {np.abs(pct_errors).mean():.2f}%")
        fig.text(0.02, 0.02, stats_text, fontsize=10, 
                verticalalignment='bottom', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def plot_candlestick(
        self,
        data: pd.DataFrame,
        title: str = "Candlestick Chart",
        save_path: str = None
    ):
        """
        Plot candlestick chart
        
        Args:
            data: DataFrame with Open, High, Low, Close columns
            title: Plot title
            save_path: Path to save figure
        """
        from mplfinance.original_flavor import candlestick_ohlc
        import matplotlib.dates as mdates
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Prepare data for candlestick
        ohlc = data[['Date', 'Open', 'High', 'Low', 'Close']].copy()
        ohlc['Date'] = mdates.date2num(pd.to_datetime(ohlc['Date']))
        
        candlestick_ohlc(ax, ohlc.values, width=0.6, 
                        colorup='green', colordown='red', alpha=0.8)
        
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def plot_with_moving_averages(
        self,
        data: pd.DataFrame,
        windows: List[int] = [7, 21, 50],
        title: str = "Stock Price with Moving Averages",
        save_path: str = None
    ):
        """
        Plot stock price with moving averages
        
        Args:
            data: DataFrame with stock data
            windows: List of moving average windows
            title: Plot title
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot close price
        ax.plot(data['Date'], data['Close'], label='Close Price', 
                color=self.colors['actual'], linewidth=1.5)
        
        # Calculate and plot moving averages
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(windows)))
        for window, color in zip(windows, colors):
            ma = data['Close'].rolling(window=window).mean()
            ax.plot(data['Date'], ma, label=f'MA {window}', 
                    color=color, linewidth=1.5, linestyle='--')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price (USD)', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def plot_volume(
        self,
        data: pd.DataFrame,
        title: str = "Trading Volume",
        save_path: str = None
    ):
        """
        Plot trading volume with price
        
        Args:
            data: DataFrame with stock data
            title: Plot title
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), 
                                  gridspec_kw={'height_ratios': [2, 1]})
        
        # Price plot
        axes[0].plot(data['Date'], data['Close'], color=self.colors['actual'], 
                     linewidth=1.5)
        axes[0].set_title(title, fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Price (USD)', fontsize=12)
        axes[0].grid(True, alpha=0.3)
        
        # Volume plot
        colors = ['green' if data['Close'].iloc[i] > data['Open'].iloc[i] 
                  else 'red' for i in range(len(data))]
        axes[1].bar(data['Date'], data['Volume'], color=colors, alpha=0.7)
        axes[1].set_ylabel('Volume', fontsize=12)
        axes[1].set_xlabel('Date', fontsize=12)
        axes[1].grid(True, alpha=0.3)
        
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def create_summary_report(
        self,
        data: pd.DataFrame,
        actual: np.ndarray,
        predicted: np.ndarray,
        metrics: dict,
        ticker: str,
        save_dir: str = "reports"
    ):
        """
        Create a comprehensive summary report with multiple plots
        
        Args:
            data: Original stock data
            actual: Actual test values
            predicted: Predicted test values
            metrics: Model metrics dictionary
            ticker: Stock ticker symbol
            save_dir: Directory to save report
        """
        os.makedirs(save_dir, exist_ok=True)
        
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Stock price history (top)
        ax1 = plt.subplot(3, 2, (1, 2))
        ax1.plot(data['Date'], data['Close'], color=self.colors['actual'], linewidth=1.5)
        ax1.set_title(f'{ticker} Stock Price History', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price (USD)')
        ax1.grid(True, alpha=0.3)
        
        # 2. Actual vs Predicted
        ax2 = plt.subplot(3, 2, 3)
        ax2.plot(actual, color=self.colors['actual'], label='Actual', linewidth=1.5)
        ax2.plot(predicted, color=self.colors['predicted'], 
                 label='Predicted', linewidth=1.5, linestyle='--')
        ax2.set_title('Actual vs Predicted Prices (Test Set)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Time Steps')
        ax2.set_ylabel('Price')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Error distribution
        ax3 = plt.subplot(3, 2, 4)
        errors = predicted.flatten() - actual.flatten()
        ax3.hist(errors, bins=30, color=self.colors['test'], alpha=0.7, edgecolor='white')
        ax3.axvline(x=0, color='black', linestyle='--', linewidth=2)
        ax3.set_title('Prediction Error Distribution', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Error')
        ax3.set_ylabel('Frequency')
        
        # 4. Scatter plot
        ax4 = plt.subplot(3, 2, 5)
        ax4.scatter(actual, predicted.flatten(), alpha=0.5, 
                   color=self.colors['actual'], s=20)
        min_val = min(actual.min(), predicted.min())
        max_val = max(actual.max(), predicted.max())
        ax4.plot([min_val, max_val], [min_val, max_val], 
                'r--', linewidth=2, label='Perfect Prediction')
        ax4.set_title('Actual vs Predicted (Scatter)', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Actual Price')
        ax4.set_ylabel('Predicted Price')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Metrics summary
        ax5 = plt.subplot(3, 2, 6)
        ax5.axis('off')
        metrics_text = f"""
        MODEL PERFORMANCE METRICS
        ========================
        
        Ticker: {ticker}
        
        Loss (MSE): {metrics.get('loss', 'N/A'):.6f}
        MAE: {metrics.get('mae', 'N/A'):.6f}
        RMSE: {metrics.get('rmse', 'N/A'):.6f}
        MAPE: {metrics.get('mape', 'N/A'):.2f}%
        
        Test Samples: {len(actual)}
        Mean Error: {errors.mean():.6f}
        Std Error: {errors.std():.6f}
        """
        ax5.text(0.1, 0.5, metrics_text, fontsize=14, 
                verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        
        plt.suptitle(f'Stock Price Prediction Report - {ticker}', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        report_path = os.path.join(save_dir, f'{ticker}_prediction_report.png')
        plt.savefig(report_path, dpi=150, bbox_inches='tight')
        print(f"Report saved to {report_path}")
        
        plt.show()


# Example usage
if __name__ == "__main__":
    # Create dummy data for demonstration
    dates = pd.date_range(start='2023-01-01', periods=252, freq='B')
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(252) * 2)
    
    data = pd.DataFrame({
        'Date': dates,
        'Open': prices + np.random.randn(252),
        'High': prices + np.abs(np.random.randn(252)) * 2,
        'Low': prices - np.abs(np.random.randn(252)) * 2,
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, 252)
    })
    
    # Create visualizer
    viz = StockVisualizer()
    
    # Plot stock data
    viz.plot_stock_data(data, title="Demo Stock Price")
