"""
Stock Prediction Web Dashboard
Simple Flask app to display predictions
"""

from flask import Flask, send_from_directory, jsonify, request
import sys
import os
sys.path.append('src')

import numpy as np
import pandas as pd
import json
from datetime import datetime

# Import logging
from logger import get_logger
logger = get_logger(__name__)

# Flask app - serve index.html directly from web folder
app = Flask(__name__, static_folder='web', static_url_path='/static')

# Import modules
from data_loader import DataLoader
from preprocessor import StockPreprocessor
from model import LSTMStockPredictor

# Import centralized config
from config import config, MODEL, TRAINING, DATA

# Cache for loaded models
loaded_models = {}
loaded_scalers = {}

def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index"""
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    
    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_sma(prices, period):
    """Calculate Simple Moving Average"""
    if len(prices) < period:
        return None
    return np.mean(prices[-period:])

def calculate_ema(prices, period):
    """Calculate Exponential Moving Average"""
    if len(prices) < period:
        return None
    multiplier = 2 / (period + 1)
    ema = prices[0]
    for price in prices[1:]:
        ema = (price - ema) * multiplier + ema
    return ema

def get_prediction(ticker, epochs=20, period='1y'):
    """Get comprehensive prediction and analysis for a stock"""
    loader = DataLoader(data_dir=DATA.data_dir)
    
    # Load or download data
    try:
        # Check if we need to force download for a specific period if not available
        data = loader.load_data(ticker)
        
        # If requested period is longer than available data, might need re-download
        # For simplicity, we'll use available data unless it's too short
        if len(data) < 60:
             data = loader.download_stock_data(ticker, period=DATA.default_period)
    except FileNotFoundError:
        try:
            download_period = period if period in ['1y', '2y', '5y', 'max'] else DATA.default_period
            data = loader.download_stock_data(ticker, period=download_period)
        except:
            return None, "Could not download data for " + ticker
            
    # Check for existing model using config helper
    model_path = config.get_model_path(ticker) + '_final.keras'
    scaler_path = config.get_scaler_path(ticker)
    
    preprocessor = StockPreprocessor(sequence_length=MODEL.sequence_length)
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        # Load existing model
        model = LSTMStockPredictor(sequence_length=MODEL.sequence_length, n_features=1)
        model.load_saved_model(model_path)
        preprocessor.load_scaler(scaler_path)
    else:
        # Train new model
        X_train, y_train, X_test, y_test = preprocessor.prepare_data(data)
        
        model = LSTMStockPredictor(
            sequence_length=MODEL.sequence_length,
            n_features=1,
            lstm_units=MODEL.lstm_units[:2],
            dropout_rate=MODEL.dropout_rate
        )
        model.build_model()
        model.train(X_train, y_train, X_val=X_test, y_val=y_test, 
                   epochs=epochs, batch_size=TRAINING.batch_size, patience=10,
                   model_path=model_path)
        preprocessor.save_scaler(scaler_path)
    
    # Make prediction
    last_sequence = preprocessor.prepare_prediction_data(data)
    next_day_scaled = model.predict(last_sequence)
    next_day_price = preprocessor.inverse_transform_predictions(next_day_scaled)
    
    # Current data
    current_price = float(data['Close'].iloc[-1])
    predicted_price = float(next_day_price[0][0])
    change_pct = (predicted_price - current_price) / current_price * 100
    
    # Get historical data for charts based on requested period
    # Define fast lookups for slicing
    period_map = {
        '1mo': 22,
        '3mo': 66,
        '6mo': 132,
        '1y': 252,
        'ytd': 0, # handled separately
        'all': 0 # all data
    }
    
    days_to_show = period_map.get(period, 252) # Default to 1y
    
    if period == 'ytd':
        current_year =  data['Date'].iloc[-1].year
        recent_data = data[data['Date'].dt.year == current_year]
    elif period == 'all':
        recent_data = data
    else:
        recent_data = data.tail(days_to_show)
        
    prices = data['Close'].values # Indicators use full data for accuracy
    
    # Calculate technical indicators
    rsi = calculate_rsi(prices)
    sma_20 = calculate_sma(prices, 20)
    sma_50 = calculate_sma(prices, 50)
    ema_20 = calculate_ema(prices, 20)
    
    # Calculate volatility (standard deviation of returns)
    returns = np.diff(prices) / prices[:-1]
    volatility = float(np.std(returns) * 100) if len(returns) > 0 else 0
    
    # 52-week high/low
    year_data = data.tail(252) if len(data) >= 252 else data
    week_52_high = float(year_data['High'].max())
    week_52_low = float(year_data['Low'].min())
    
    # Determine signal strength
    signal_strength = min(abs(change_pct) * 20, 100)
    
    # Build comprehensive result
    result = {
        'ticker': ticker,
        'current_price': current_price,
        'predicted_price': predicted_price,
        'change_percent': change_pct,
        'signal': 'BULLISH' if change_pct > 0 else 'BEARISH',
        'signal_strength': signal_strength,
        'last_update': data['Date'].iloc[-1].strftime('%Y-%m-%d'),
        
        # Stock info
        'stock_info': {
            'currency': config.get_currency(ticker),
            'is_indonesian': config.is_indonesian_stock(ticker),
            'exchange': 'IDX' if config.is_indonesian_stock(ticker) else 'NYSE/NASDAQ'
        },
        
        # Key statistics
        'statistics': {
            'open': float(data['Open'].iloc[-1]),
            'high': float(data['High'].iloc[-1]),
            'low': float(data['Low'].iloc[-1]),
            'prev_close': float(data['Close'].iloc[-2]) if len(data) > 1 else current_price,
            'volume': int(data['Volume'].iloc[-1]),
            'avg_volume': int(data['Volume'].tail(20).mean()),
            'week_52_high': week_52_high,
            'week_52_low': week_52_low
        },
        
        # Technical indicators
        'indicators': {
            'rsi_14': float(rsi) if rsi else None,
            'sma_20': float(sma_20) if sma_20 else None,
            'sma_50': float(sma_50) if sma_50 else None,
            'ema_20': float(ema_20) if ema_20 else None,
            'volatility': volatility
        },
        
        # Chart data
        'charts': {
            'dates': recent_data['Date'].dt.strftime('%Y-%m-%d').tolist(),
            'prices': recent_data['Close'].tolist(),
            'volumes': recent_data['Volume'].tolist(),
            'highs': recent_data['High'].tolist(),
            'lows': recent_data['Low'].tolist()
        }
    }
    
    return result, None

@app.route('/')
def index():
    """Serve the main dashboard page"""
    return send_from_directory('web', 'index.html')

@app.route('/api/predict/<ticker>')
def predict(ticker):
    """Predict stock price for given ticker"""
    logger.info(f"Prediction request for: {ticker}")
    epochs = request.args.get('epochs', 20, type=int)
    period = request.args.get('period', '1y')
    result, error = get_prediction(ticker.upper(), epochs, period)
    
    if error:
        logger.error(f"Prediction failed for {ticker}: {error}")
        return jsonify({'error': error}), 400
    
    logger.info(f"Prediction successful for {ticker}: {result['signal']} ({result['change_percent']:.2f}%)")
    return jsonify(result)

@app.route('/api/stocks')
def list_stocks():
    """List available stocks with saved models"""
    stocks = []
    models_dir = DATA.models_dir
    
    if os.path.exists(models_dir):
        for f in os.listdir(models_dir):
            if f.startswith('lstm_') and f.endswith('_final.keras'):
                ticker = f.replace('lstm_', '').replace('_final.keras', '').replace('_', '.')
                stocks.append(ticker)
    
    return jsonify({'stocks': stocks, 'available_tickers': {
        'us': config.us_tickers,
        'indonesian': config.indonesian_tickers
    }})

@app.route('/api/config')
def get_config():
    """Get current model configuration"""
    return jsonify({
        'model': {
            'sequence_length': MODEL.sequence_length,
            'lstm_units': MODEL.lstm_units,
            'dropout_rate': MODEL.dropout_rate
        },
        'training': {
            'epochs': TRAINING.epochs,
            'batch_size': TRAINING.batch_size
        }
    })

if __name__ == '__main__':
    os.makedirs('web/static', exist_ok=True)
    logger.info("=" * 50)
    logger.info("STOCK PREDICTION WEB DASHBOARD")
    logger.info("=" * 50)
    logger.info("Open http://localhost:5000 in your browser")
    app.run(debug=True, port=5000)
