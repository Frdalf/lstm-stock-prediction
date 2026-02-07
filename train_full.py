"""
Full Training Script with Visualization
Run this for complete training with 50 epochs and save all plots
"""

import sys
import os
sys.path.append('src')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8-whitegrid')

print("=" * 60)
print("LSTM STOCK PREDICTION - FULL TRAINING")
print("=" * 60)

# Load modules
from data_loader import DataLoader
from preprocessor import StockPreprocessor
from model import LSTMStockPredictor
from visualizer import StockVisualizer

# Import centralized config
from config import config, MODEL, TRAINING, DATA

# Configuration from config.py
TICKER = config.default_ticker
SEQUENCE_LENGTH = MODEL.sequence_length
TRAIN_RATIO = TRAINING.train_ratio
EPOCHS = TRAINING.epochs
BATCH_SIZE = TRAINING.batch_size

print(f"\nConfiguration:")
print(f"  Ticker: {TICKER}")
print(f"  Sequence Length: {SEQUENCE_LENGTH} days")
print(f"  Train/Test Ratio: {TRAIN_RATIO*100:.0f}/{(1-TRAIN_RATIO)*100:.0f}")
print(f"  Epochs: {EPOCHS}")
print(f"  Batch Size: {BATCH_SIZE}")

# Load data
print("\n[1/5] Loading Data...")
loader = DataLoader(data_dir='data/raw')
try:
    data = loader.load_data(TICKER)
    print(f"[OK] Loaded {len(data)} rows")
except FileNotFoundError:
    data = loader.download_stock_data(TICKER, period="2y")
    print(f"[OK] Downloaded {len(data)} rows")

# Initialize visualizer
viz = StockVisualizer()
os.makedirs('results', exist_ok=True)

# Plot stock data
print("\n[2/5] Visualizing Stock Data...")
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Price plot
axes[0].plot(data['Date'], data['Close'], color='#2E86AB', linewidth=1.5)
axes[0].set_title(f'{TICKER} Stock Price History', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Date')
axes[0].set_ylabel('Price (USD)')
axes[0].grid(True, alpha=0.3)

# Volume plot
colors = ['green' if data['Close'].iloc[i] > data['Open'].iloc[i] else 'red' 
          for i in range(len(data))]
axes[1].bar(data['Date'], data['Volume'], color=colors, alpha=0.7)
axes[1].set_title('Trading Volume', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Date')
axes[1].set_ylabel('Volume')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/01_stock_data.png', dpi=150, bbox_inches='tight')
print("[OK] Saved: results/01_stock_data.png")
plt.close()

# Preprocessing
print("\n[3/5] Preprocessing Data...")
preprocessor = StockPreprocessor(sequence_length=SEQUENCE_LENGTH)
X_train, y_train, X_test, y_test = preprocessor.prepare_data(
    data, target_column='Close', train_ratio=TRAIN_RATIO
)
preprocessor.save_scaler('models/scaler.pkl')

# Train/Test split visualization
split_point = int(len(data) * TRAIN_RATIO) + SEQUENCE_LENGTH
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(data['Date'][:split_point], data['Close'][:split_point], 
        color='#1B998B', label='Training Data', linewidth=1.5)
ax.plot(data['Date'][split_point:], data['Close'][split_point:], 
        color='#F46036', label='Test Data', linewidth=1.5)
ax.axvline(x=data['Date'].iloc[split_point], color='gray', linestyle='--', linewidth=2)
ax.set_title(f'{TICKER} Train/Test Split', fontsize=14, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Price (USD)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/02_train_test_split.png', dpi=150, bbox_inches='tight')
print("[OK] Saved: results/02_train_test_split.png")
plt.close()

# Build and Train Model
print("\n[4/5] Training LSTM Model...")
model = LSTMStockPredictor(
    sequence_length=SEQUENCE_LENGTH,
    n_features=1,
    lstm_units=[50, 50, 50],  # 3 layers
    dropout_rate=0.2,
    learning_rate=0.001
)
model.build_model()

history = model.train(
    X_train, y_train,
    X_val=X_test, y_val=y_test,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    model_path='models/lstm_model.keras',
    patience=15
)

# Plot training history
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history['loss'], label='Training Loss', linewidth=2)
axes[0].plot(history['val_loss'], label='Validation Loss', linewidth=2)
axes[0].set_title('Model Loss Over Epochs', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss (MSE)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history['mae'], label='Training MAE', linewidth=2)
axes[1].plot(history['val_mae'], label='Validation MAE', linewidth=2)
axes[1].set_title('Model MAE Over Epochs', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('MAE')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/03_training_history.png', dpi=150, bbox_inches='tight')
print("[OK] Saved: results/03_training_history.png")
plt.close()

# Evaluate and Predict
print("\n[5/5] Evaluating Model...")
metrics = model.evaluate(X_test, y_test)

predictions_scaled = model.predict(X_test)
predictions = preprocessor.inverse_transform_predictions(predictions_scaled)
actual = preprocessor.inverse_transform_actual(y_test)

# Predictions plot
fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(actual.flatten(), color='#2E86AB', label='Actual Price', linewidth=2)
ax.plot(predictions.flatten(), color='#E94F37', label='Predicted Price', 
        linewidth=2, linestyle='--')
ax.set_title(f'{TICKER} LSTM Prediction - Actual vs Predicted', fontsize=14, fontweight='bold')
ax.set_xlabel('Time Steps')
ax.set_ylabel('Price (USD)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/04_predictions.png', dpi=150, bbox_inches='tight')
print("[OK] Saved: results/04_predictions.png")
plt.close()

# Full visualization with training data
train_prices = data['Close'].values[SEQUENCE_LENGTH:SEQUENCE_LENGTH + len(y_train)]
fig, ax = plt.subplots(figsize=(16, 8))
train_x = np.arange(len(train_prices))
test_x = np.arange(len(train_prices), len(train_prices) + len(actual))

ax.plot(train_x, train_prices, color='#1B998B', label='Training Data', linewidth=1.5)
ax.plot(test_x, actual.flatten(), color='#2E86AB', label='Actual (Test)', linewidth=2)
ax.plot(test_x, predictions.flatten(), color='#E94F37', 
        label='Predicted (Test)', linewidth=2, linestyle='--')
ax.axvline(x=len(train_prices), color='gray', linestyle=':', linewidth=2)
ax.set_title(f'{TICKER} Complete Prediction View', fontsize=14, fontweight='bold')
ax.set_xlabel('Time Steps')
ax.set_ylabel('Price (USD)')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/05_full_prediction.png', dpi=150, bbox_inches='tight')
print("[OK] Saved: results/05_full_prediction.png")
plt.close()

# Error analysis
errors = predictions.flatten() - actual.flatten()
pct_errors = (errors / actual.flatten()) * 100

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Error distribution
axes[0, 0].hist(errors, bins=30, color='#E94F37', alpha=0.7, edgecolor='white')
axes[0, 0].axvline(x=0, color='black', linestyle='--', linewidth=2)
axes[0, 0].set_title('Prediction Error Distribution', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Error (USD)')
axes[0, 0].set_ylabel('Frequency')

# Error over time
axes[0, 1].plot(errors, color='#E94F37', alpha=0.7)
axes[0, 1].axhline(y=0, color='black', linestyle='--', linewidth=2)
axes[0, 1].set_title('Error Over Time', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Time Steps')
axes[0, 1].set_ylabel('Error (USD)')
axes[0, 1].grid(True, alpha=0.3)

# Scatter plot
axes[1, 0].scatter(actual.flatten(), predictions.flatten(), alpha=0.5, color='#2E86AB')
min_val, max_val = actual.min(), actual.max()
axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
axes[1, 0].set_title('Actual vs Predicted Scatter', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Actual Price (USD)')
axes[1, 0].set_ylabel('Predicted Price (USD)')
axes[1, 0].grid(True, alpha=0.3)

# Percentage error
axes[1, 1].plot(pct_errors, color='#F46036', alpha=0.7)
axes[1, 1].axhline(y=0, color='black', linestyle='--', linewidth=2)
axes[1, 1].set_title('Percentage Error Over Time', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Time Steps')
axes[1, 1].set_ylabel('Error (%)')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/06_error_analysis.png', dpi=150, bbox_inches='tight')
print("[OK] Saved: results/06_error_analysis.png")
plt.close()

# Save model
model.save_model('models/lstm_final_model.keras')

# Next day prediction
last_sequence = preprocessor.prepare_prediction_data(data)
next_day_scaled = model.predict(last_sequence)
next_day_price = preprocessor.inverse_transform_predictions(next_day_scaled)
current_price = data['Close'].iloc[-1]
predicted_change = (next_day_price[0][0] - current_price) / current_price * 100

# Save summary report
from sklearn.metrics import mean_absolute_error, mean_squared_error
mae = mean_absolute_error(actual, predictions)
rmse = np.sqrt(mean_squared_error(actual, predictions))
mape = np.mean(np.abs(pct_errors))

report = f"""
================================================================================
                    LSTM STOCK PREDICTION - FINAL REPORT
================================================================================

CONFIGURATION
-------------
Ticker: {TICKER}
Sequence Length: {SEQUENCE_LENGTH} days
Train/Test Ratio: {TRAIN_RATIO*100:.0f}/{(1-TRAIN_RATIO)*100:.0f}
LSTM Layers: 3 (50-50-50 units)
Dropout Rate: 0.2
Epochs: {len(history['loss'])} (early stopped)

DATA SUMMARY
------------
Total Data Points: {len(data)}
Training Samples: {len(y_train)}
Testing Samples: {len(y_test)}
Date Range: {data['Date'].min().strftime('%Y-%m-%d')} to {data['Date'].max().strftime('%Y-%m-%d')}

MODEL PERFORMANCE
-----------------
MAE (Mean Absolute Error): ${mae:.2f}
RMSE (Root Mean Squared Error): ${rmse:.2f}
MAPE (Mean Absolute Percentage Error): {mape:.2f}%
Final Training Loss: {history['loss'][-1]:.6f}
Final Validation Loss: {history['val_loss'][-1]:.6f}

NEXT DAY PREDICTION
-------------------
Current Price: ${current_price:.2f}
Predicted Price: ${next_day_price[0][0]:.2f}
Expected Change: {predicted_change:+.2f}%
Signal: {'BULLISH (UP)' if predicted_change > 0 else 'BEARISH (DOWN)'}

FILES GENERATED
---------------
- results/01_stock_data.png
- results/02_train_test_split.png
- results/03_training_history.png
- results/04_predictions.png
- results/05_full_prediction.png
- results/06_error_analysis.png
- models/lstm_final_model.keras
- models/scaler.pkl

================================================================================
DISCLAIMER: This is for educational purposes only. Past performance does not
guarantee future results. Do not use for actual trading decisions.
================================================================================
"""

with open('results/report.txt', 'w') as f:
    f.write(report)

print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)
print(f"\nModel Performance:")
print(f"  MAE: ${mae:.2f}")
print(f"  RMSE: ${rmse:.2f}")
print(f"  MAPE: {mape:.2f}%")
print(f"\nNext Day Prediction:")
print(f"  Current: ${current_price:.2f}")
print(f"  Predicted: ${next_day_price[0][0]:.2f} ({predicted_change:+.2f}%)")
print(f"\nAll results saved to: results/")
print("=" * 60)
