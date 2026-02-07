"""
Predict Any Stock - Universal Prediction Script
Usage: python predict_any.py TICKER [EPOCHS]
Example: python predict_any.py GOOGL 30
"""

import sys
import os
sys.path.append('src')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Import centralized config
from config import config, MODEL, TRAINING, DATA

# Get ticker from command line
if len(sys.argv) < 2:
    print("Usage: python predict_any.py TICKER [EPOCHS]")
    print("Example: python predict_any.py GOOGL 30")
    print("\nAvailable tickers:")
    print(f"  US: {', '.join(config.us_tickers)}")
    print(f"  ID: {', '.join(config.indonesian_tickers)}")
    sys.exit(1)

TICKER = sys.argv[1].upper()
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 30
SEQUENCE_LENGTH = MODEL.sequence_length
BATCH_SIZE = TRAINING.batch_size

print("=" * 60)
print(f"LSTM STOCK PREDICTION - {TICKER}")
print("=" * 60)

from data_loader import DataLoader
from preprocessor import StockPreprocessor
from model import LSTMStockPredictor

# Download/Load data
print(f"\n[1/4] Loading {TICKER} data...")
loader = DataLoader(data_dir='data/raw')

try:
    data = loader.load_data(TICKER)
    print(f"[OK] Loaded existing: {len(data)} rows")
except FileNotFoundError:
    try:
        data = loader.download_stock_data(TICKER, period="2y")
        print(f"[OK] Downloaded: {len(data)} rows")
    except Exception as e:
        print(f"[ERROR] Could not download {TICKER}: {e}")
        sys.exit(1)

# Check if Indonesian stock using config
is_indonesian = config.is_indonesian_stock(TICKER)
currency = config.get_currency(TICKER)
currency_format = config.get_currency_format(TICKER)

print(f"  Date range: {data['Date'].min().strftime('%Y-%m-%d')} to {data['Date'].max().strftime('%Y-%m-%d')}")
current_price = data['Close'].iloc[-1]
print(f"  Current price: {currency} {currency_format.format(current_price)}")

# Preprocessing
print("\n[2/4] Preprocessing...")
preprocessor = StockPreprocessor(sequence_length=SEQUENCE_LENGTH)
X_train, y_train, X_test, y_test = preprocessor.prepare_data(
    data, target_column='Close', train_ratio=0.8
)

# Training
print(f"\n[3/4] Training LSTM ({EPOCHS} epochs)...")
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
    model_path=f'models/lstm_{TICKER.replace(".", "_")}.keras',
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
plt.style.use('seaborn-v0_8-whitegrid')

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Price history
axes[0, 0].plot(data['Date'], data['Close'], color='#2E86AB', linewidth=1.5)
axes[0, 0].set_title(f'{TICKER} Stock Price History', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Date')
axes[0, 0].set_ylabel(f'Price ({currency})')
axes[0, 0].grid(True, alpha=0.3)

# Predictions
axes[0, 1].plot(actual.flatten(), color='#2E86AB', label='Actual', linewidth=2)
axes[0, 1].plot(predictions.flatten(), color='#E94F37', label='Predicted', 
                linewidth=2, linestyle='--')
axes[0, 1].set_title('Actual vs Predicted (Test Set)', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Time Steps')
axes[0, 1].set_ylabel(f'Price ({currency})')
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

# Scatter plot
axes[1, 1].scatter(actual.flatten(), predictions.flatten(), alpha=0.5, color='#2E86AB')
min_val, max_val = actual.min(), actual.max()
axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
axes[1, 1].set_title('Actual vs Predicted Scatter', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel(f'Actual ({currency})')
axes[1, 1].set_ylabel(f'Predicted ({currency})')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
output_file = f'results/{TICKER.replace(".", "_")}_prediction.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"[OK] Saved: {output_file}")
plt.close()

# Next day prediction
last_sequence = preprocessor.prepare_prediction_data(data)
next_day_scaled = model.predict(last_sequence)
next_day_price = preprocessor.inverse_transform_predictions(next_day_scaled)
predicted_change = (next_day_price[0][0] - current_price) / current_price * 100

# Save model
model.save_model(f'models/lstm_{TICKER.replace(".", "_")}_final.keras')
preprocessor.save_scaler(f'models/scaler_{TICKER.replace(".", "_")}.pkl')

# Results
print("\n" + "=" * 60)
print(f"{TICKER} PREDICTION RESULTS")
print("=" * 60)
print(f"\nModel Performance:")
print(f"  MAE: {currency} {currency_format.format(mae)}")
print(f"  RMSE: {currency} {currency_format.format(rmse)}")
print(f"  MAPE: {mape:.2f}%")
print(f"\nNext Day Prediction:")
print(f"  Current: {currency} {currency_format.format(current_price)}")
print(f"  Predicted: {currency} {currency_format.format(next_day_price[0][0])}")
print(f"  Change: {predicted_change:+.2f}%")

if predicted_change > 0:
    print(f"\n[BULLISH] Model predicts UPWARD movement")
else:
    print(f"\n[BEARISH] Model predicts DOWNWARD movement")

print("\n" + "=" * 60)
print("Files saved:")
print(f"  - {output_file}")
print(f"  - models/lstm_{TICKER.replace('.', '_')}_final.keras")
print(f"  - models/scaler_{TICKER.replace('.', '_')}.pkl")
print("=" * 60)
