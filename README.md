# Options Grid Screener

Professional options chain viewer with real-time data support.

## Features

- **Professional Grid Layout**: Calls | Strike | Puts view
- **Real-time Data**: Charles Schwab or Tradier API support
- **Visual Indicators**: ITM/OTM highlighting, ATM row, high IV alerts
- **Greeks Display**: Delta, Gamma, Theta, Vega (when available)
- **Liquidity Filtering**: Auto-hides illiquid strikes
- **Auto-refresh**: Optional 5-second refresh cycle

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Data Providers

### Option 1: Charles Schwab (Recommended)

**Free with any Schwab brokerage account. Real-time quotes + Greeks.**

1. **Create Developer Account** at [developer.schwab.com](https://developer.schwab.com/)

2. **Register Your App**
   - Dashboard → Create App
   - Select "Market Data Production"
   - Set callback URL: `https://127.0.0.1:8182`
   - Wait 2-3 days for approval

3. **Initial Authentication** (one-time)
   ```bash
   export SCHWAB_APP_KEY="your-app-key"
   export SCHWAB_APP_SECRET="your-app-secret"
   python schwab_auth_setup.py
   ```

4. **Run the App**
   ```bash
   streamlit run app.py
   ```

### Option 2: Tradier

1. Sign up at [developer.tradier.com](https://developer.tradier.com/)
2. Run:
   ```bash
   export TRADIER_API_KEY="your-api-key"
   streamlit run app.py
   ```

### Option 3: Yahoo Finance (Default)

No setup required - just run `streamlit run app.py`

## Grid Legend

- **Green tint**: In-the-money
- **Blue row**: At-the-money
- **Yellow IV**: High implied volatility
- **Blue volume**: High relative volume

## License

MIT