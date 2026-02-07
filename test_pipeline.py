"""
Quick Test Script for LSTM Stock Prediction Pipeline
Run this to verify the entire pipeline works correctly
"""

import sys
import os
sys.path.append('src')

import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("LSTM STOCK PREDICTION - PIPELINE TEST")
print("=" * 60)

# Step 1: Data Download
print("\n[1/4] Downloading Stock Data...")
from data_loader import DataLoader

loader = DataLoader(data_dir='data/raw')
ticker = "AAPL"

try:
    # Try to load existing data first
    data = loader.load_data(ticker)
    print(f"[OK] Loaded existing data: {len(data)} rows for {ticker}")
except FileNotFoundError:
    try:
        data = loader.download_stock_data(ticker, period="2y")
        print(f"[OK] Downloaded {len(data)} rows for {ticker}")
    except Exception as e:
        print(f"[ERROR] Error downloading data: {e}")
        print("[INFO] Using sample data for demonstration...")
        # Create sample data if download fails
        import pandas as pd
        dates = pd.date_range(end=pd.Timestamp.today(), periods=500, freq='B')
        np.random.seed(42)
        prices = 150 + np.cumsum(np.random.randn(500) * 2)
        data = pd.DataFrame({
            'Date': dates,
            'Open': prices + np.random.randn(500),
            'High': prices + np.abs(np.random.randn(500)) * 2,
            'Low': prices - np.abs(np.random.randn(500)) * 2,
            'Close': prices,
            'Volume': np.random.randint(10000000, 100000000, 500)
        })
        ticker = "SAMPLE"
print(f"  Date range: {data['Date'].min().strftime('%Y-%m-%d')} to {data['Date'].max().strftime('%Y-%m-%d')}")

# Step 2: Preprocessing
print("\n[2/4] Preprocessing Data...")
from preprocessor import StockPreprocessor

SEQUENCE_LENGTH = 60
TRAIN_RATIO = 0.8

preprocessor = StockPreprocessor(sequence_length=SEQUENCE_LENGTH)
X_train, y_train, X_test, y_test = preprocessor.prepare_data(
    data, target_column='Close', train_ratio=TRAIN_RATIO
)

print(f"[OK] Data preprocessed successfully")
print(f"  X_train shape: {X_train.shape}")
print(f"  X_test shape: {X_test.shape}")

# Save preprocessed data
os.makedirs('data/processed', exist_ok=True)
np.save('data/processed/X_train.npy', X_train)
np.save('data/processed/y_train.npy', y_train)
np.save('data/processed/X_test.npy', X_test)
np.save('data/processed/y_test.npy', y_test)
preprocessor.save_scaler('models/scaler.pkl')
print(f"[OK] Preprocessed data saved to data/processed/")

# Step 3: Build & Train Model
print("\n[3/4] Building & Training LSTM Model...")
from model import LSTMStockPredictor

model = LSTMStockPredictor(
    sequence_length=SEQUENCE_LENGTH,
    n_features=1,
    lstm_units=[50, 50],  # 2 layers for quick test
    dropout_rate=0.2,
    learning_rate=0.001
)

model.build_model()
print("[OK] Model built successfully")

# Train with fewer epochs for quick test
print("\nTraining (10 epochs for quick test)...")
history = model.train(
    X_train, y_train,
    X_val=X_test, y_val=y_test,
    epochs=10,  # Quick test
    batch_size=32,
    model_path='models/lstm_model.keras',
    patience=5
)

# Step 4: Evaluate
print("\n[4/4] Evaluating Model...")
metrics = model.evaluate(X_test, y_test)

# Make predictions
predictions_scaled = model.predict(X_test)
predictions = preprocessor.inverse_transform_predictions(predictions_scaled)
actual = preprocessor.inverse_transform_actual(y_test)

# Calculate final metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error
mae = mean_absolute_error(actual, predictions)
rmse = np.sqrt(mean_squared_error(actual, predictions))

print(f"\n[OK] Model evaluation complete")
print(f"  MAE: ${mae:.2f}")
print(f"  RMSE: ${rmse:.2f}")

# Save model
model.save_model('models/lstm_final_model.keras')

# Show sample predictions
print("\n" + "=" * 60)
print("SAMPLE PREDICTIONS")
print("=" * 60)
print(f"{'Actual':>12} | {'Predicted':>12} | {'Diff':>10}")
print("-" * 40)
for i in range(min(5, len(actual))):
    diff = predictions[i][0] - actual[i][0]
    print(f"${actual[i][0]:>10.2f} | ${predictions[i][0]:>10.2f} | {diff:>+9.2f}")

# Predict next day
print("\n" + "=" * 60)
print("NEXT DAY PREDICTION")
print("=" * 60)
last_sequence = preprocessor.prepare_prediction_data(data)
next_day_scaled = model.predict(last_sequence)
next_day_price = preprocessor.inverse_transform_predictions(next_day_scaled)

current_price = data['Close'].iloc[-1]
predicted_change = (next_day_price[0][0] - current_price) / current_price * 100

print(f"Current Price ({ticker}): ${current_price:.2f}")
print(f"Predicted Next Day: ${next_day_price[0][0]:.2f}")
print(f"Expected Change: {predicted_change:+.2f}%")

if predicted_change > 0:
    print("\n[UP] Model predicts UPWARD movement")
else:
    print("\n[DOWN] Model predicts DOWNWARD movement")

print("\n" + "=" * 60)
print("[SUCCESS] PIPELINE TEST COMPLETE!")
print("=" * 60)
print("\nFiles created:")
print("  - data/raw/AAPL_data.csv")
print("  - data/processed/X_train.npy, y_train.npy, X_test.npy, y_test.npy")
print("  - models/scaler.pkl")
print("  - models/lstm_model.keras")
print("  - models/lstm_final_model.keras")
print("\nYou can now run the Jupyter notebooks for detailed analysis!")
