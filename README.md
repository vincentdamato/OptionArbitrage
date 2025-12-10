# 📈 Options Arbitrage Scanner

A sophisticated quantitative options screener for detecting **mispricing opportunities** with real-time Schwab API integration, statistical analysis, and comprehensive backtesting.

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🎯 Features

### Arbitrage Detection
- **Put-Call Parity** violations with synthetic pricing
- **Box Spread** arbitrage (risk-free rate arbitrage)
- **Butterfly** mispricing detection
- **Vertical Spread** arbitrage
- **Calendar Spread** anomalies
- **Volatility Skew** statistical arbitrage

### Real-Time Market Data
- **Schwab API** integration for live options chains
- Complete Greeks (Delta, Gamma, Theta, Vega, Rho)
- Second-order Greeks (Vanna, Charm, Vomma)
- IV surface analysis

### Statistical Analysis
- Z-score based opportunity scoring
- Monte Carlo simulations for P&L distribution
- VaR and CVaR risk metrics
- Volatility regime detection
- IV term structure analysis
- Put/Call skew quantification

### Backtesting Engine
- Walk-forward optimization
- Realistic transaction cost modeling
- Slippage simulation
- Comprehensive performance metrics (Sharpe, Sortino, Calmar)
- Win rate analysis by strategy type

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/vincentdamato/OptionArbitrage.git
cd OptionArbitrage

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create a `.env` file with your Schwab API credentials:

```env
SCHWAB_API_KEY=your_api_key_here
SCHWAB_APP_SECRET=your_app_secret_here
SCHWAB_CALLBACK_URL=https://127.0.0.1:8182
```

### Running the Dashboard

```bash
streamlit run app.py
```

### Using the CLI

```bash
# Scan for opportunities
python cli.py scan -s SPY QQQ AAPL NVDA

# Analyze a specific symbol
python cli.py analyze SPY --detailed

# Run backtest
python cli.py backtest -s 2024-01-01 -e 2024-12-01

# Continuous monitoring
python cli.py watch -i 60 --min-edge 50
```

## 📊 Architecture

```
OptionArbitrage/
├── api/                    # Schwab API client
│   └── schwab_client.py    # Real-time market data
├── analysis/               # Quantitative analysis
│   ├── greeks_engine.py    # Black-Scholes & Greeks
│   └── statistical_analyzer.py  # Statistical methods
├── strategies/             # Arbitrage detection
│   └── arbitrage_scanner.py  # Multi-strategy scanner
├── backtesting/            # Strategy validation
│   └── backtest_engine.py  # Walk-forward backtesting
├── core/                   # Orchestration
│   └── orchestrator.py     # Main coordinator
├── utils/                  # Utilities
│   └── helpers.py          # Helper functions
├── config/                 # Configuration
│   └── settings.py         # Pydantic settings
├── app.py                  # Streamlit dashboard
└── cli.py                  # Command-line interface
```

## 🔬 Arbitrage Types Explained

### 1. Put-Call Parity (Risk-Free)
Exploits pricing inconsistencies between calls, puts, and the underlying stock.

**Formula:** `C - P = S - K * e^(-rT)`

When violated:
- **Conversion**: Sell call + Buy put + Buy stock
- **Reversal**: Buy call + Sell put + Short stock

### 2. Box Spread (Risk-Free)
Combines a bull call spread with a bear put spread to lock in risk-free profit.

**Value at expiration:** `K₂ - K₁`
**Present value:** `(K₂ - K₁) * e^(-rT)`

### 3. Butterfly Spread
Exploits mispricing in three-strike combinations. A properly priced butterfly should never have negative value.

### 4. Volatility Skew (Statistical)
Identifies options with implied volatility significantly deviating from the surface norm (z-score > 2).

## ⚙️ Configuration Options

### Scan Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_edge_pct` | 0.5% | Minimum profit edge to flag |
| `commission_per_contract` | $0.65 | Trading commission |
| `slippage_pct` | 0.05% | Expected slippage |
| `min_open_interest` | 100 | Minimum OI filter |
| `min_volume` | 50 | Minimum volume filter |
| `max_bid_ask_spread_pct` | 5% | Max spread filter |

### Risk Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_position_size` | 10 | Max contracts per trade |
| `max_delta_exposure` | 0.10 | Max absolute delta |
| `max_total_exposure` | $10,000 | Max capital at risk |
| `daily_loss_limit` | $2,000 | Max daily loss |

## 📈 Greeks Calculation

The platform calculates comprehensive Greeks:

### First-Order Greeks
- **Delta (Δ)**: Rate of change of option price with underlying
- **Gamma (Γ)**: Rate of change of delta
- **Theta (Θ)**: Time decay (daily)
- **Vega (ν)**: Sensitivity to volatility
- **Rho (ρ)**: Sensitivity to interest rates

### Second-Order Greeks
- **Vanna**: d(Delta)/d(Vol) - delta's sensitivity to volatility
- **Charm**: d(Delta)/d(Time) - delta decay
- **Vomma**: d(Vega)/d(Vol) - vega convexity
- **Speed**: d(Gamma)/d(Spot) - gamma's sensitivity to price

## 🧪 Backtesting

The backtesting engine includes:

- **Walk-Forward Optimization**: Prevents overfitting
- **Realistic Costs**: Commission and slippage modeling
- **Monte Carlo**: P&L distribution simulation
- **Risk Metrics**: VaR, CVaR, max drawdown

```python
from backtesting import BacktestEngine

engine = BacktestEngine(
    initial_capital=100000,
    commission_per_contract=0.65,
    slippage_model="proportional"
)

result = engine.run_arbitrage_backtest(opportunities)
print(engine.generate_performance_report(result))
```

## 📚 API Reference

### SchwabClient

```python
from api import SchwabClient

client = SchwabClient(
    api_key="your_key",
    app_secret="your_secret"
)

# Get options chain
chain = client.get_option_chain(
    symbol="SPY",
    min_dte=1,
    max_dte=60
)

# Get real-time quote
quote = client.get_quote("SPY")
```

### ArbitrageScanner

```python
from strategies import ArbitrageScanner

scanner = ArbitrageScanner(
    min_edge_pct=0.5,
    commission_per_contract=0.65
)

opportunities = scanner.scan_chain(chain)
```

### GreeksCalculator

```python
from analysis import GreeksCalculator

calc = GreeksCalculator()

# Calculate all Greeks
greeks = calc.calculate_all_greeks(
    S=100,      # Spot price
    K=100,      # Strike
    T=0.25,     # Time to exp (years)
    r=0.05,     # Risk-free rate
    sigma=0.20, # Volatility
    option_type="call"
)

# Calculate implied volatility
iv = calc.implied_volatility(
    market_price=5.50,
    S=100, K=100, T=0.25, r=0.05
)
```

## 🛡️ Risk Management

The platform enforces multiple risk controls:

1. **Position Limits**: Maximum contracts per position
2. **Delta Limits**: Maximum directional exposure
3. **Daily Loss Limits**: Automatic halt on excessive losses
4. **Liquidity Filters**: Minimum volume/OI requirements
5. **Spread Filters**: Maximum bid-ask spread

## 🤝 Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests.

## ⚠️ Disclaimer

This software is for educational and research purposes only. Options trading involves significant risk of loss. Past performance does not guarantee future results. Always conduct your own analysis and consult with a qualified financial advisor before trading.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- [schwab-py](https://github.com/alexgolec/schwab-py) for Schwab API wrapper
- [QuantLib](https://www.quantlib.org/) for options pricing reference
- [Streamlit](https://streamlit.io/) for the dashboard framework