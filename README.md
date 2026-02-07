# 📈 LSTM Stock Price Prediction

Neural network untuk memprediksi harga saham menggunakan LSTM (Long Short-Term Memory) dengan web dashboard interaktif.

![MIT License](https://img.shields.io/badge/license-MIT-green)
![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)

## ✨ Features

- 🌐 **Web Dashboard** - Analisis saham dengan UI modern dan interaktif
- 📊 **Technical Indicators** - RSI, SMA, EMA, Volatility
- 🔮 **AI Prediction** - LSTM 3-layer dengan signal strength meter
- 📈 **Multi-chart** - Price history, volume, 52-week range
- 🇮🇩 **Multi-market** - Support saham Indonesia (.JK) dan US market
- ⚡ **Auto-download** - Data otomatis dari Yahoo Finance
- 🎨 **Modern UI** - Glassmorphism design dengan skeleton loading

## 📸 Screenshots

### Web Dashboard
- Animated gradient background dengan floating particles
- Card-based layout dengan glow effects
- Real-time technical analysis

## 📊 Model Performance

| Stock | MAE | RMSE | MAPE |
|-------|-----|------|------|
| AAPL (Apple) | $11.10 | $12.72 | 4.15% |
| BBCA.JK (BCA) | Rp 181 | Rp 234 | 2.50% |

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Web Dashboard
```bash
python app.py
# Open http://localhost:5000
```

### 3. Run Quick Test
```bash
python test_pipeline.py
```

### 4. Full Training with Visualization
```bash
python train_full.py
```

### 5. Predict Any Stock
```bash
python predict_any.py GOOGL
python predict_any.py BBCA.JK
```

### 6. Interactive Notebooks
```bash
jupyter notebook
```

## 📁 Project Structure

```
stock-prediction/
├── app.py                # Flask web server
├── config.py             # Centralized configuration
├── web/
│   └── index.html        # Dashboard UI
├── src/
│   ├── data_loader.py    # Yahoo Finance downloader
│   ├── preprocessor.py   # MinMaxScaler & sequences
│   ├── model.py          # LSTM architecture
│   ├── visualizer.py     # Plotting functions
│   └── logger.py         # Centralized logging
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_model_training.ipynb
├── tests/                # Unit tests
├── models/               # Saved models & scalers
├── data/                 # Stock data
├── results/              # Generated charts
└── logs/                 # Application logs
```

## 🔧 Configuration

All configuration is centralized in `config.py`:

```python
from config import config, MODEL, TRAINING, DATA

# Model settings
MODEL.sequence_length    # 60 days lookback
MODEL.lstm_units         # [50, 50, 50] layers
MODEL.dropout_rate       # 0.2

# Training settings
TRAINING.epochs          # 50
TRAINING.batch_size      # 32
TRAINING.train_ratio     # 0.8

# Helper methods
config.get_model_path("AAPL")      # models/lstm_AAPL_final.keras
config.get_currency("BBCA.JK")     # "Rp"
config.is_indonesian_stock("TLKM.JK")  # True
```

## 🌐 Web Dashboard API

| Endpoint | Description |
|----------|-------------|
| `GET /` | Dashboard UI |
| `GET /api/predict/<ticker>` | Get prediction + indicators |
| `GET /api/stocks` | List available tickers |
| `GET /api/config` | Current model config |

### API Response Example
```json
{
  "ticker": "AAPL",
  "current_price": 185.50,
  "predicted_price": 187.20,
  "signal": "BULLISH",
  "signal_strength": 34.5,
  "indicators": {
    "rsi_14": 55.2,
    "sma_20": 182.30,
    "sma_50": 178.45,
    "volatility": 1.85
  },
  "statistics": {
    "week_52_high": 199.62,
    "week_52_low": 164.08
  }
}
```

## 📈 Ticker Format

| Exchange | Suffix | Example |
|----------|--------|---------|
| 🇺🇸 USA (NYSE/NASDAQ) | *none* | AAPL, GOOGL, MSFT |
| 🇮🇩 Indonesia (BEI) | .JK | BBCA.JK, TLKM.JK |
| 🇸🇬 Singapore | .SI | DBS.SI |
| 🇭🇰 Hong Kong | .HK | 0700.HK |

## 🧠 Model Architecture

```
Input Layer: (60, 1) - 60 days lookback
    ↓
LSTM Layer 1: 50 units + Dropout 0.2
    ↓
LSTM Layer 2: 50 units + Dropout 0.2
    ↓
LSTM Layer 3: 50 units + Dropout 0.2
    ↓
Dense Layer: 25 units (ReLU)
    ↓
Output Layer: 1 unit (Price)
```

## 📉 Training Features

- **EarlyStopping** - Stop when validation loss plateaus
- **ModelCheckpoint** - Auto-save best model
- **ReduceLROnPlateau** - Adaptive learning rate
- **Centralized Logging** - Console + file logging

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Skip slow tests
pytest tests/ -v -m "not slow"
```

## ⚠️ Disclaimer

This project is for **educational purposes only**. 

- Past performance does not guarantee future results
- Stock market prediction is inherently uncertain
- Do NOT use this for actual trading decisions
- Always do your own research before investing

## 📝 License

MIT License - See [LICENSE](LICENSE) file for details.

---

Made with ❤️ using TensorFlow + Flask
