# Options Screener

A clean, minimal options chain viewer built with Streamlit.

## Features

- Real-time stock price and metrics
- Options chain display (calls & puts)
- Key metrics: Strike, Bid/Ask, Volume, Open Interest, IV
- Clean, professional interface

## Quick Start
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Data Source

Market data provided by Yahoo Finance via yfinance.

---

*For educational purposes only. Not financial advice.*
```

---

**4. .gitignore**
```
__pycache__/
*.py[cod]
venv/
.venv/
.env
.vscode/
.idea/
.DS_Store
*.csv
*.parquet
*.db
.streamlit/secrets.toml