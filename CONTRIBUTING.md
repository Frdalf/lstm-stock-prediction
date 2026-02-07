# Contributing to LSTM Stock Prediction

Terima kasih telah tertarik untuk berkontribusi! 🎉

## 🚀 Quick Start

1. **Fork** repository ini
2. **Clone** fork Anda:
   ```bash
   git clone https://github.com/YOUR_USERNAME/stock-prediction.git
   cd stock-prediction
   ```
3. **Install** dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. **Buat branch** baru:
   ```bash
   git checkout -b feature/nama-fitur
   ```

## 📝 Guidelines

### Code Style
- Gunakan **PEP 8** untuk Python code
- Tambahkan **docstrings** untuk semua functions dan classes
- Gunakan **type hints** bila memungkinkan

### Commit Messages
Format: `type: description`

Types:
- `feat`: Fitur baru
- `fix`: Bug fix
- `docs`: Dokumentasi
- `style`: Formatting (tanpa perubahan logic)
- `refactor`: Refactoring code
- `test`: Menambah/memperbaiki tests

Contoh:
```
feat: add multi-stock comparison feature
fix: resolve RSI calculation edge case
docs: update API documentation
```

### Pull Request
1. Pastikan semua tests pass: `pytest tests/ -v`
2. Update dokumentasi jika diperlukan
3. Buat PR dengan deskripsi yang jelas

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test
pytest tests/test_model.py -v
```

## 📁 Project Structure

```
src/
├── data_loader.py    # Data downloading from Yahoo Finance
├── preprocessor.py   # Data preprocessing & scaling
├── model.py          # LSTM model architecture
├── visualizer.py     # Plotting functions
└── logger.py         # Centralized logging
```

## ❓ Questions?

Buka **Issue** baru jika ada pertanyaan atau temukan bug.

---

Made with ❤️ by contributors
