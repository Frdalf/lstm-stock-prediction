"""
Predict Indonesian Stock - BBCA (Bank Central Asia)
"""

import sys
import os
sys.path.append('src')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')

print("=" * 60)
print("LSTM PREDICTION - BBCA.JK (Bank Central Asia)")
print("=" * 60)

from data_loader import DataLoader
from preprocessor import StockPreprocessor
from model import LSTMStockPredictor

# Import centralized config
from config import config, MODEL, TRAINING, DATA

# Configuration from config.py
TICKER = "BBCA.JK"
SEQUENCE_LENGTH = MODEL.sequence_length
EPOCHS = 30  # Shorter for demo
BATCH_SIZE = TRAINING.batch_size

print(f"\nTicker: {TICKER}")
print(f"Sequence Length: {SEQUENCE_LENGTH} days")
print(f"Epochs: {EPOCHS}")

# Download/Load data
print("\n[1/4] Downloading BBCA.JK Data...")
loader = DataLoader(data_dir='data/raw')

try:
    data = loader.load_data(TICKER)
    print(f"[OK] Loaded existing: {len(data)} rows")
except FileNotFoundError:
    try:
        data = loader.download_stock_data(TICKER, period="2y")
        print(f"[OK] Downloaded: {len(data)} rows")
    except Exception as e:
        print(f"[ERROR] {e}")
        print("[INFO] Using sample Indonesian stock data...")
        dates = pd.date_range(end=pd.Timestamp.today(), periods=500, freq='B')
        np.random.seed(123)
        prices = 9500 + np.cumsum(np.random.randn(500) * 100)
        data = pd.DataFrame({
            'Date': dates,
            'Open': prices + np.random.randn(500) * 50,
            'High': prices + np.abs(np.random.randn(500)) * 100,
            'Low': prices - np.abs(np.random.randn(500)) * 100,
            'Close': prices,
            'Volume': np.random.randint(10000000, 50000000, 500)
        })

print(f"  Date range: {data['Date'].min().strftime('%Y-%m-%d')} to {data['Date'].max().strftime('%Y-%m-%d')}")
print(f"  Latest Close: Rp {data['Close'].iloc[-1]:,.0f}")

# Preprocessing
print("\n[2/4] Preprocessing...")
preprocessor = StockPreprocessor(sequence_length=SEQUENCE_LENGTH)
X_train, y_train, X_test, y_test = preprocessor.prepare_data(
    data, target_column='Close', train_ratio=0.8
)

# Build and Train
print("\n[3/4] Training LSTM...")
model = LSTMStockPredictor(
    sequence_length=SEQUENCE_LENGTH,
    n_features=1,
    lstm_units=[50, 50],
    dropout_rate=0.2,
    learning_rate=0.001
)
model.build_model()

history = model.train(
    X_train, y_train,
    X_val=X_test, y_val=y_test,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    model_path='models/lstm_bbca.keras',
    patience=10
)

# Evaluate
print("\n[4/4] Evaluating...")
metrics = model.evaluate(X_test, y_test)

predictions_scaled = model.predict(X_test)
predictions = preprocessor.inverse_transform_predictions(predictions_scaled)
actual = preprocessor.inverse_transform_actual(y_test)

# Calculate metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error
mae = mean_absolute_error(actual, predictions)
rmse = np.sqrt(mean_squared_error(actual, predictions))
mape = np.mean(np.abs((actual.flatten() - predictions.flatten()) / actual.flatten())) * 100

# Create visualization
os.makedirs('results', exist_ok=True)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Price history
axes[0, 0].plot(data['Date'], data['Close'], color='#2E86AB', linewidth=1.5)
axes[0, 0].set_title(f'{TICKER} Stock Price History', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Date')
axes[0, 0].set_ylabel('Price (IDR)')
axes[0, 0].grid(True, alpha=0.3)

# Predictions
axes[0, 1].plot(actual.flatten(), color='#2E86AB', label='Actual', linewidth=2)
axes[0, 1].plot(predictions.flatten(), color='#E94F37', label='Predicted', 
                linewidth=2, linestyle='--')
axes[0, 1].set_title('Actual vs Predicted (Test Set)', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Time Steps')
axes[0, 1].set_ylabel('Price (IDR)')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Training history
axes[1, 0].plot(history['loss'], label='Training Loss', linewidth=2)
axes[1, 0].plot(history['val_loss'], label='Validation Loss', linewidth=2)
axes[1, 0].set_title('Training History', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Loss')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Error distribution
errors = predictions.flatten() - actual.flatten()
axes[1, 1].hist(errors, bins=25, color='#E94F37', alpha=0.7, edgecolor='white')
axes[1, 1].axvline(x=0, color='black', linestyle='--', linewidth=2)
axes[1, 1].set_title('Error Distribution', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Error (IDR)')
axes[1, 1].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('results/bbca_prediction.png', dpi=150, bbox_inches='tight')
print("[OK] Saved: results/bbca_prediction.png")
plt.close()

# Next day prediction
last_sequence = preprocessor.prepare_prediction_data(data)
next_day_scaled = model.predict(last_sequence)
next_day_price = preprocessor.inverse_transform_predictions(next_day_scaled)
current_price = data['Close'].iloc[-1]
predicted_change = (next_day_price[0][0] - current_price) / current_price * 100

# Save model
model.save_model('models/lstm_bbca_final.keras')
preprocessor.save_scaler('models/scaler_bbca.pkl')

# Results
print("\n" + "=" * 60)
print("BBCA.JK PREDICTION RESULTS")
print("=" * 60)
print(f"\nModel Performance:")
print(f"  MAE: Rp {mae:,.0f}")
print(f"  RMSE: Rp {rmse:,.0f}")
print(f"  MAPE: {mape:.2f}%")
print(f"\nNext Day Prediction:")
print(f"  Current Price: Rp {current_price:,.0f}")
print(f"  Predicted: Rp {next_day_price[0][0]:,.0f}")
print(f"  Expected Change: {predicted_change:+.2f}%")

if predicted_change > 0:
    print("\n[UP] Model predicts BULLISH movement")
else:
    print("\n[DOWN] Model predicts BEARISH movement")

print("\n" + "=" * 60)
print("Files saved:")
print("  - results/bbca_prediction.png")
print("  - models/lstm_bbca_final.keras")
print("  - models/scaler_bbca.pkl")
print("=" * 60)
