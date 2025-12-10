"""
Options Screener - Professional Options Grid
Real-time options chain with Schwab/Tradier API + yfinance fallback
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import requests
from typing import Optional, Tuple, Dict, List
import os

# Load environment variables from .env file (if exists)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, use system env vars

# Page configuration
st.set_page_config(
    page_title="Options Grid",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Data Provider Priority: Schwab > Tradier > Yahoo Finance
# 
# SCHWAB API (Recommended - Free with Schwab account)
# Setup: https://developer.schwab.com/
# 1. Create developer account
# 2. Register app with "Market Data Production"
# 3. Set callback URL to: https://127.0.0.1:8182
# 4. Wait 2-3 days for approval
# 5. Run initial auth to get tokens (see setup instructions below)

SCHWAB_APP_KEY = os.environ.get("SCHWAB_APP_KEY", "")
SCHWAB_APP_SECRET = os.environ.get("SCHWAB_APP_SECRET", "")
SCHWAB_TOKEN_PATH = os.environ.get("SCHWAB_TOKEN_PATH", "schwab_tokens.json")

# Tradier API Configuration (Alternative)
# Get your free API key at: https://developer.tradier.com/
# For sandbox (delayed): use 'sandbox.tradier.com' 
# For live (real-time): use 'api.tradier.com'

TRADIER_API_KEY = os.environ.get("TRADIER_API_KEY", "")
TRADIER_ENDPOINT = os.environ.get("TRADIER_ENDPOINT", "https://sandbox.tradier.com/v1")

# Provider selection (auto-detect based on credentials)
USE_SCHWAB = bool(SCHWAB_APP_KEY and SCHWAB_APP_SECRET)
USE_TRADIER = bool(TRADIER_API_KEY) and not USE_SCHWAB

# ============================================================================
# STYLING
# ============================================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
    
    :root {
        --bg-primary: #0a0a0f;
        --bg-secondary: #12121a;
        --bg-tertiary: #1a1a24;
        --border-color: #2a2a3a;
        --text-primary: #e4e4e7;
        --text-secondary: #71717a;
        --text-muted: #52525b;
        --accent-green: #22c55e;
        --accent-green-dim: #166534;
        --accent-red: #ef4444;
        --accent-red-dim: #991b1b;
        --accent-blue: #3b82f6;
        --accent-yellow: #eab308;
        --accent-orange: #f97316;
        --itm-bg: rgba(34, 197, 94, 0.08);
        --otm-bg: transparent;
        --atm-bg: rgba(59, 130, 246, 0.12);
    }
    
    .stApp {
        background: var(--bg-primary);
    }
    
    html, body, [class*="css"] {
        font-family: 'JetBrains Mono', monospace;
        color: var(--text-primary);
    }
    
    #MainMenu, footer, header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    .terminal-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 1rem 0;
        border-bottom: 1px solid var(--border-color);
        margin-bottom: 1.5rem;
    }
    
    .terminal-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--text-primary);
        letter-spacing: -0.02em;
    }
    
    .terminal-subtitle {
        font-size: 0.75rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    
    .live-indicator {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.75rem;
        color: var(--accent-green);
    }
    
    .delayed-indicator {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.75rem;
        color: var(--accent-orange);
    }
    
    .live-dot {
        width: 8px;
        height: 8px;
        background: var(--accent-green);
        border-radius: 50%;
        animation: pulse 2s infinite;
    }
    
    .delayed-dot {
        width: 8px;
        height: 8px;
        background: var(--accent-orange);
        border-radius: 50%;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(34, 197, 94, 0.7); }
        50% { opacity: 0.8; box-shadow: 0 0 0 6px rgba(34, 197, 94, 0); }
    }
    
    .stock-bar {
        display: flex;
        align-items: center;
        gap: 2rem;
        padding: 1rem 1.5rem;
        background: var(--bg-secondary);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        margin-bottom: 1.5rem;
        flex-wrap: wrap;
    }
    
    .stock-symbol {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        color: var(--text-primary);
    }
    
    .stock-name {
        font-size: 0.875rem;
        color: var(--text-secondary);
    }
    
    .stock-price {
        font-size: 1.75rem;
        font-weight: 600;
    }
    
    .stock-change {
        font-size: 0.875rem;
        font-weight: 500;
    }
    
    .positive { color: var(--accent-green); }
    .negative { color: var(--accent-red); }
    
    .stock-stat {
        text-align: center;
    }
    
    .stock-stat-value {
        font-size: 1rem;
        font-weight: 600;
        color: var(--text-primary);
    }
    
    .stock-stat-label {
        font-size: 0.625rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .grid-container {
        background: var(--bg-secondary);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        overflow: hidden;
        overflow-x: auto;
    }
    
    .grid-header {
        display: grid;
        grid-template-columns: 1fr 100px 1fr;
        background: var(--bg-tertiary);
        border-bottom: 1px solid var(--border-color);
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        min-width: 900px;
    }
    
    .grid-header-calls, .grid-header-puts {
        display: grid;
        grid-template-columns: repeat(7, 1fr);
        padding: 0.75rem 0.5rem;
    }
    
    .grid-header-calls {
        background: rgba(34, 197, 94, 0.1);
        color: var(--accent-green);
    }
    
    .grid-header-puts {
        background: rgba(239, 68, 68, 0.1);
        color: var(--accent-red);
    }
    
    .grid-header-strike {
        padding: 0.75rem 0.5rem;
        text-align: center;
        color: var(--accent-blue);
        background: rgba(59, 130, 246, 0.1);
    }
    
    .grid-header span {
        text-align: right;
        padding: 0 0.25rem;
    }
    
    .grid-header span:first-child {
        text-align: left;
    }
    
    .grid-row {
        display: grid;
        grid-template-columns: 1fr 100px 1fr;
        border-bottom: 1px solid var(--border-color);
        font-size: 0.75rem;
        transition: background 0.15s ease;
        min-width: 900px;
    }
    
    .grid-row:hover {
        background: rgba(255, 255, 255, 0.02);
    }
    
    .grid-row:last-child {
        border-bottom: none;
    }
    
    .grid-row.itm-call .calls-data {
        background: var(--itm-bg);
    }
    
    .grid-row.itm-put .puts-data {
        background: var(--itm-bg);
    }
    
    .grid-row.atm {
        background: var(--atm-bg);
        border-top: 2px solid var(--accent-blue);
        border-bottom: 2px solid var(--accent-blue);
        font-weight: 600;
    }
    
    .grid-row.atm .strike-cell {
        background: var(--accent-blue);
        color: white;
    }
    
    .calls-data, .puts-data {
        display: grid;
        grid-template-columns: repeat(7, 1fr);
        padding: 0.4rem 0.5rem;
        align-items: center;
    }
    
    .calls-data span, .puts-data span {
        text-align: right;
        color: var(--text-secondary);
        padding: 0 0.25rem;
        white-space: nowrap;
    }
    
    .calls-data span:first-child, .puts-data span:first-child {
        text-align: left;
    }
    
    .strike-cell {
        padding: 0.4rem 0.5rem;
        text-align: center;
        font-weight: 600;
        color: var(--text-primary);
        background: var(--bg-tertiary);
    }
    
    .bid-ask {
        color: var(--text-primary) !important;
        font-weight: 500;
    }
    
    .iv-high {
        color: var(--accent-yellow) !important;
    }
    
    .volume-high {
        color: var(--accent-blue) !important;
    }
    
    .delta-col {
        color: var(--accent-blue) !important;
    }
    
    .stats-row {
        display: grid;
        grid-template-columns: repeat(6, 1fr);
        gap: 0.75rem;
        margin-bottom: 1.5rem;
    }
    
    .stat-card {
        background: var(--bg-secondary);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 0.875rem;
        text-align: center;
    }
    
    .stat-value {
        font-size: 1.125rem;
        font-weight: 600;
        color: var(--text-primary);
    }
    
    .stat-label {
        font-size: 0.625rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.25rem;
    }
    
    .timestamp {
        font-size: 0.75rem;
        color: var(--text-muted);
        text-align: right;
        margin-top: 0.5rem;
    }
    
    .data-source {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        background: var(--bg-tertiary);
        border-radius: 4px;
        font-size: 0.625rem;
        color: var(--text-muted);
        margin-left: 0.5rem;
    }
    
    .stTextInput > div > div > input {
        background: var(--bg-secondary) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 6px !important;
        color: var(--text-primary) !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 1rem !important;
        padding: 0.75rem 1rem !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: var(--accent-blue) !important;
        box-shadow: 0 0 0 1px var(--accent-blue) !important;
    }
    
    .stSelectbox > div > div {
        background: var(--bg-secondary) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 6px !important;
    }
    
    .stSelectbox > div > div > div {
        color: var(--text-primary) !important;
        font-family: 'JetBrains Mono', monospace !important;
    }
    
    .stCheckbox > label {
        color: var(--text-secondary) !important;
        font-size: 0.875rem !important;
    }
    
    .stSlider > div > div > div {
        background: var(--accent-blue) !important;
    }
    
    div[data-baseweb="select"] > div {
        background: var(--bg-secondary) !important;
        border-color: var(--border-color) !important;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# DATA PROVIDERS
# ============================================================================

class SchwabProvider:
    """Real-time data from Charles Schwab API"""
    
    def __init__(self, app_key: str, app_secret: str, token_path: str = "schwab_tokens.json"):
        self.app_key = app_key
        self.app_secret = app_secret
        self.token_path = token_path
        self.access_token = None
        self.refresh_token = None
        self.token_expiry = None
        self.base_url = "https://api.schwabapi.com/marketdata/v1"
        self._load_tokens()
    
    def _load_tokens(self):
        """Load tokens from file"""
        import json
        try:
            if os.path.exists(self.token_path):
                with open(self.token_path, 'r') as f:
                    tokens = json.load(f)
                    self.access_token = tokens.get('access_token')
                    self.refresh_token = tokens.get('refresh_token')
                    expiry = tokens.get('token_expiry')
                    if expiry:
                        self.token_expiry = datetime.fromisoformat(expiry)
        except Exception as e:
            st.warning(f"Could not load Schwab tokens: {e}")
    
    def _save_tokens(self):
        """Save tokens to file"""
        import json
        try:
            tokens = {
                'access_token': self.access_token,
                'refresh_token': self.refresh_token,
                'token_expiry': self.token_expiry.isoformat() if self.token_expiry else None
            }
            with open(self.token_path, 'w') as f:
                json.dump(tokens, f)
        except Exception as e:
            st.warning(f"Could not save Schwab tokens: {e}")
    
    def _refresh_access_token(self) -> bool:
        """Refresh the access token using refresh token"""
        if not self.refresh_token:
            return False
        
        try:
            import base64
            auth_string = f"{self.app_key}:{self.app_secret}"
            auth_bytes = base64.b64encode(auth_string.encode()).decode()
            
            response = requests.post(
                "https://api.schwabapi.com/v1/oauth/token",
                headers={
                    "Authorization": f"Basic {auth_bytes}",
                    "Content-Type": "application/x-www-form-urlencoded"
                },
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": self.refresh_token
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                self.access_token = data['access_token']
                self.refresh_token = data.get('refresh_token', self.refresh_token)
                self.token_expiry = datetime.now() + timedelta(seconds=data.get('expires_in', 1800))
                self._save_tokens()
                return True
            else:
                st.error(f"Token refresh failed: {response.status_code}")
                return False
        except Exception as e:
            st.error(f"Token refresh error: {e}")
            return False
    
    def _ensure_valid_token(self) -> bool:
        """Ensure we have a valid access token"""
        if not self.access_token:
            return False
        
        # Refresh if token expires in less than 5 minutes
        if self.token_expiry and datetime.now() > self.token_expiry - timedelta(minutes=5):
            return self._refresh_access_token()
        
        return True
    
    def _get_headers(self) -> Dict:
        """Get headers with current access token"""
        return {
            "Authorization": f"Bearer {self.access_token}",
            "Accept": "application/json"
        }
    
    def is_authenticated(self) -> bool:
        """Check if we have valid authentication"""
        return self._ensure_valid_token()
    
    def get_quote(self, symbol: str) -> Optional[Dict]:
        """Get real-time quote for a symbol"""
        if not self._ensure_valid_token():
            return None
        
        try:
            url = f"{self.base_url}/quotes"
            response = requests.get(
                url, 
                params={"symbols": symbol, "fields": "quote,reference"},
                headers=self._get_headers()
            )
            response.raise_for_status()
            data = response.json()
            
            if symbol in data:
                quote = data[symbol].get("quote", {})
                ref = data[symbol].get("reference", {})
                
                return {
                    "symbol": symbol,
                    "name": ref.get("description", symbol),
                    "price": quote.get("lastPrice"),
                    "bid": quote.get("bidPrice"),
                    "ask": quote.get("askPrice"),
                    "change": quote.get("netChange"),
                    "change_pct": quote.get("netPercentChangeInDouble"),
                    "volume": quote.get("totalVolume"),
                    "day_high": quote.get("highPrice"),
                    "day_low": quote.get("lowPrice"),
                    "prev_close": quote.get("closePrice"),
                }
            return None
        except Exception as e:
            st.error(f"Schwab quote error: {e}")
            return None
    
    def get_expirations(self, symbol: str) -> List[str]:
        """Get available expiration dates"""
        if not self._ensure_valid_token():
            return []
        
        try:
            url = f"{self.base_url}/expirationchain"
            response = requests.get(
                url,
                params={"symbol": symbol},
                headers=self._get_headers()
            )
            response.raise_for_status()
            data = response.json()
            
            expirations = []
            for exp in data.get("expirationList", []):
                exp_date = exp.get("expirationDate")
                if exp_date:
                    expirations.append(exp_date)
            
            return sorted(expirations)
        except Exception as e:
            st.error(f"Schwab expirations error: {e}")
            return []
    
    def get_options_chain(self, symbol: str, expiration: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Get options chain for a given expiration"""
        if not self._ensure_valid_token():
            return None, None
        
        try:
            url = f"{self.base_url}/chains"
            params = {
                "symbol": symbol,
                "contractType": "ALL",
                "strikeCount": 100,
                "includeUnderlyingQuote": "true",
                "fromDate": expiration,
                "toDate": expiration
            }
            response = requests.get(url, params=params, headers=self._get_headers())
            response.raise_for_status()
            data = response.json()
            
            calls_data = []
            puts_data = []
            
            # Process call options
            call_map = data.get("callExpDateMap", {})
            for exp_key, strikes in call_map.items():
                for strike_key, options in strikes.items():
                    for opt in options:
                        calls_data.append({
                            "strike": opt.get("strikePrice"),
                            "bid": opt.get("bid"),
                            "ask": opt.get("ask"),
                            "lastPrice": opt.get("last"),
                            "volume": opt.get("totalVolume", 0),
                            "openInterest": opt.get("openInterest", 0),
                            "impliedVolatility": opt.get("volatility"),
                            "delta": opt.get("delta"),
                            "gamma": opt.get("gamma"),
                            "theta": opt.get("theta"),
                            "vega": opt.get("vega"),
                        })
            
            # Process put options
            put_map = data.get("putExpDateMap", {})
            for exp_key, strikes in put_map.items():
                for strike_key, options in strikes.items():
                    for opt in options:
                        puts_data.append({
                            "strike": opt.get("strikePrice"),
                            "bid": opt.get("bid"),
                            "ask": opt.get("ask"),
                            "lastPrice": opt.get("last"),
                            "volume": opt.get("totalVolume", 0),
                            "openInterest": opt.get("openInterest", 0),
                            "impliedVolatility": opt.get("volatility"),
                            "delta": opt.get("delta"),
                            "gamma": opt.get("gamma"),
                            "theta": opt.get("theta"),
                            "vega": opt.get("vega"),
                        })
            
            calls_df = pd.DataFrame(calls_data) if calls_data else pd.DataFrame()
            puts_df = pd.DataFrame(puts_data) if puts_data else pd.DataFrame()
            
            return calls_df, puts_df
            
        except Exception as e:
            st.error(f"Schwab options chain error: {e}")
            return None, None


class TradierProvider:
    """Real-time data from Tradier API"""
    
    def __init__(self, api_key: str, endpoint: str = "https://sandbox.tradier.com/v1"):
        self.api_key = api_key
        self.endpoint = endpoint
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json"
        }
    
    def get_quote(self, symbol: str) -> Optional[Dict]:
        """Get real-time quote for a symbol"""
        try:
            url = f"{self.endpoint}/markets/quotes"
            response = requests.get(url, params={"symbols": symbol}, headers=self.headers)
            response.raise_for_status()
            data = response.json()
            
            if "quotes" in data and "quote" in data["quotes"]:
                quote = data["quotes"]["quote"]
                return {
                    "symbol": quote.get("symbol"),
                    "name": quote.get("description", symbol),
                    "price": quote.get("last"),
                    "bid": quote.get("bid"),
                    "ask": quote.get("ask"),
                    "change": quote.get("change"),
                    "change_pct": quote.get("change_percentage"),
                    "volume": quote.get("volume"),
                    "day_high": quote.get("high"),
                    "day_low": quote.get("low"),
                    "prev_close": quote.get("prevclose"),
                }
            return None
        except Exception as e:
            st.error(f"Tradier quote error: {e}")
            return None
    
    def get_expirations(self, symbol: str) -> List[str]:
        """Get available expiration dates"""
        try:
            url = f"{self.endpoint}/markets/options/expirations"
            response = requests.get(url, params={"symbol": symbol}, headers=self.headers)
            response.raise_for_status()
            data = response.json()
            
            if "expirations" in data and "date" in data["expirations"]:
                dates = data["expirations"]["date"]
                return dates if isinstance(dates, list) else [dates]
            return []
        except Exception as e:
            st.error(f"Tradier expirations error: {e}")
            return []
    
    def get_options_chain(self, symbol: str, expiration: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Get options chain for a given expiration"""
        try:
            url = f"{self.endpoint}/markets/options/chains"
            params = {
                "symbol": symbol,
                "expiration": expiration,
                "greeks": "true"
            }
            response = requests.get(url, params=params, headers=self.headers)
            response.raise_for_status()
            data = response.json()
            
            if "options" not in data or "option" not in data["options"]:
                return None, None
            
            options = data["options"]["option"]
            if not isinstance(options, list):
                options = [options]
            
            calls_data = []
            puts_data = []
            
            for opt in options:
                greeks = opt.get("greeks") or {}
                row = {
                    "strike": opt.get("strike"),
                    "bid": opt.get("bid"),
                    "ask": opt.get("ask"),
                    "lastPrice": opt.get("last"),
                    "volume": opt.get("volume", 0),
                    "openInterest": opt.get("open_interest", 0),
                    "impliedVolatility": greeks.get("mid_iv"),
                    "delta": greeks.get("delta"),
                    "gamma": greeks.get("gamma"),
                    "theta": greeks.get("theta"),
                    "vega": greeks.get("vega"),
                }
                
                if opt.get("option_type") == "call":
                    calls_data.append(row)
                else:
                    puts_data.append(row)
            
            calls_df = pd.DataFrame(calls_data) if calls_data else pd.DataFrame()
            puts_df = pd.DataFrame(puts_data) if puts_data else pd.DataFrame()
            
            return calls_df, puts_df
            
        except Exception as e:
            st.error(f"Tradier options chain error: {e}")
            return None, None


class YFinanceProvider:
    """Fallback data from Yahoo Finance (delayed ~15 min)"""
    
    def __init__(self):
        import yfinance as yf
        self.yf = yf
    
    def get_quote(self, symbol: str) -> Optional[Dict]:
        """Get quote for a symbol"""
        try:
            stock = self.yf.Ticker(symbol)
            info = stock.info
            
            current_price = info.get("regularMarketPrice") or info.get("currentPrice")
            prev_close = info.get("regularMarketPreviousClose")
            change = (current_price - prev_close) if current_price and prev_close else 0
            change_pct = (change / prev_close * 100) if prev_close else 0
            
            return {
                "symbol": symbol,
                "name": info.get("shortName", symbol),
                "price": current_price,
                "bid": info.get("bid"),
                "ask": info.get("ask"),
                "change": change,
                "change_pct": change_pct,
                "volume": info.get("regularMarketVolume", 0),
                "day_high": info.get("dayHigh"),
                "day_low": info.get("dayLow"),
                "prev_close": prev_close,
            }
        except Exception as e:
            st.error(f"YFinance quote error: {e}")
            return None
    
    def get_expirations(self, symbol: str) -> List[str]:
        """Get available expiration dates"""
        try:
            stock = self.yf.Ticker(symbol)
            return list(stock.options)
        except Exception:
            return []
    
    def get_options_chain(self, symbol: str, expiration: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Get options chain for a given expiration"""
        try:
            stock = self.yf.Ticker(symbol)
            chain = stock.option_chain(expiration)
            return chain.calls, chain.puts
        except Exception as e:
            st.error(f"YFinance options error: {e}")
            return None, None


# Initialize provider based on configuration
@st.cache_resource
def get_provider():
    """Get the appropriate data provider (Schwab > Tradier > Yahoo)"""
    if USE_SCHWAB and SCHWAB_APP_KEY and SCHWAB_APP_SECRET:
        provider = SchwabProvider(SCHWAB_APP_KEY, SCHWAB_APP_SECRET, SCHWAB_TOKEN_PATH)
        if provider.is_authenticated():
            return provider, "Schwab"
        else:
            st.warning("Schwab tokens not found or expired. Run initial auth setup.")
    
    if USE_TRADIER and TRADIER_API_KEY:
        return TradierProvider(TRADIER_API_KEY, TRADIER_ENDPOINT), "Tradier"
    
    return YFinanceProvider(), "Yahoo Finance"


# ============================================================================
# DATA FUNCTIONS WITH CACHING
# ============================================================================

# Determine cache TTL based on provider
CACHE_TTL = 5 if (USE_SCHWAB or USE_TRADIER) else 30

@st.cache_data(ttl=CACHE_TTL)
def get_stock_data(ticker: str) -> Optional[Dict]:
    """Fetch stock data with appropriate caching"""
    provider, _ = get_provider()
    return provider.get_quote(ticker)


@st.cache_data(ttl=CACHE_TTL)
def get_options_data(ticker: str, expiration: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Fetch options chain data"""
    provider, _ = get_provider()
    return provider.get_options_chain(ticker, expiration)


@st.cache_data(ttl=300)
def get_expirations(ticker: str) -> List[str]:
    """Get available expiration dates"""
    provider, _ = get_provider()
    return provider.get_expirations(ticker)


def calculate_days_to_expiry(exp_date: str) -> int:
    """Calculate days until expiration"""
    exp = datetime.strptime(exp_date, "%Y-%m-%d")
    today = datetime.now()
    return (exp - today).days


def build_options_grid(calls: pd.DataFrame, puts: pd.DataFrame, current_price: float) -> Optional[pd.DataFrame]:
    """Build unified options grid with calls and puts aligned by strike"""
    if calls is None or puts is None or calls.empty or puts.empty:
        return None
    
    all_strikes = sorted(set(calls['strike'].tolist() + puts['strike'].tolist()))
    grid_data = []
    
    # Find the closest strike to current price for ATM marking
    if current_price:
        atm_strike = min(all_strikes, key=lambda x: abs(x - current_price))
    else:
        atm_strike = None
    
    for strike in all_strikes:
        call_row = calls[calls['strike'] == strike]
        put_row = puts[puts['strike'] == strike]
        
        if current_price:
            if strike == atm_strike:
                moneyness = "ATM"
            elif strike < current_price:
                moneyness = "ITM_CALL"
            else:
                moneyness = "ITM_PUT"
        else:
            moneyness = "OTM"
        
        def safe_get(df, col):
            if len(df) > 0 and col in df.columns:
                val = df[col].iloc[0]
                return val if pd.notna(val) else None
            return None
        
        row = {
            "strike": strike,
            "moneyness": moneyness,
            "call_bid": safe_get(call_row, 'bid'),
            "call_ask": safe_get(call_row, 'ask'),
            "call_last": safe_get(call_row, 'lastPrice'),
            "call_volume": safe_get(call_row, 'volume'),
            "call_oi": safe_get(call_row, 'openInterest'),
            "call_iv": safe_get(call_row, 'impliedVolatility'),
            "call_delta": safe_get(call_row, 'delta'),
            "put_bid": safe_get(put_row, 'bid'),
            "put_ask": safe_get(put_row, 'ask'),
            "put_last": safe_get(put_row, 'lastPrice'),
            "put_volume": safe_get(put_row, 'volume'),
            "put_oi": safe_get(put_row, 'openInterest'),
            "put_iv": safe_get(put_row, 'impliedVolatility'),
            "put_delta": safe_get(put_row, 'delta'),
        }
        grid_data.append(row)
    
    return pd.DataFrame(grid_data)


def format_value(val, decimals=2, prefix="", suffix=""):
    """Format a value for display"""
    if pd.isna(val) or val is None:
        return "—"
    if isinstance(val, (int, float)):
        if abs(val) >= 1_000_000:
            return f"{prefix}{val/1_000_000:.1f}M{suffix}"
        elif abs(val) >= 10_000:
            return f"{prefix}{val/1_000:.1f}K{suffix}"
        elif decimals == 0:
            return f"{prefix}{int(val):,}{suffix}"
        else:
            return f"{prefix}{val:.{decimals}f}{suffix}"
    return str(val)


def render_options_grid(grid_df: pd.DataFrame, current_price: float, iv_threshold: float = 0.8) -> str:
    """Render the options grid as HTML"""
    if grid_df is None or grid_df.empty:
        return "<p style='color: var(--text-muted);'>No options data available</p>"
    
    max_call_vol = grid_df['call_volume'].max() if grid_df['call_volume'].notna().any() else 1
    max_put_vol = grid_df['put_volume'].max() if grid_df['put_volume'].notna().any() else 1
    
    has_delta = grid_df['call_delta'].notna().any()
    
    if has_delta:
        call_headers = "<span>Δ</span><span>IV</span><span>OI</span><span>Vol</span><span>Last</span><span>Bid</span><span>Ask</span>"
        put_headers = "<span>Bid</span><span>Ask</span><span>Last</span><span>Vol</span><span>OI</span><span>IV</span><span>Δ</span>"
    else:
        call_headers = "<span>IV</span><span>OI</span><span>Vol</span><span>Last</span><span>Bid</span><span>Ask</span><span></span>"
        put_headers = "<span></span><span>Bid</span><span>Ask</span><span>Last</span><span>Vol</span><span>OI</span><span>IV</span>"
    
    html = f"""
    <div class="grid-container">
        <div class="grid-header">
            <div class="grid-header-calls">
                {call_headers}
            </div>
            <div class="grid-header-strike">STRIKE</div>
            <div class="grid-header-puts">
                {put_headers}
            </div>
        </div>
    """
    
    for _, row in grid_df.iterrows():
        row_class = ""
        if row['moneyness'] == "ATM":
            row_class = "atm"
        elif row['moneyness'] == "ITM_CALL":
            row_class = "itm-call"
        elif row['moneyness'] == "ITM_PUT":
            row_class = "itm-put"
        
        call_iv = row['call_iv']
        call_iv_class = "iv-high" if call_iv and call_iv > iv_threshold else ""
        call_vol_class = "volume-high" if row['call_volume'] and row['call_volume'] > max_call_vol * 0.5 else ""
        
        put_iv = row['put_iv']
        put_iv_class = "iv-high" if put_iv and put_iv > iv_threshold else ""
        put_vol_class = "volume-high" if row['put_volume'] and row['put_volume'] > max_put_vol * 0.5 else ""
        
        call_delta = format_value(row['call_delta'], 2) if has_delta else ""
        put_delta = format_value(row['put_delta'], 2) if has_delta else ""
        
        if has_delta:
            call_data = f"""
                <span class="delta-col">{call_delta}</span>
                <span class="{call_iv_class}">{format_value(call_iv * 100 if call_iv else None, 1)}%</span>
                <span>{format_value(row['call_oi'], 0)}</span>
                <span class="{call_vol_class}">{format_value(row['call_volume'], 0)}</span>
                <span>{format_value(row['call_last'])}</span>
                <span class="bid-ask">{format_value(row['call_bid'])}</span>
                <span class="bid-ask">{format_value(row['call_ask'])}</span>
            """
            put_data = f"""
                <span class="bid-ask">{format_value(row['put_bid'])}</span>
                <span class="bid-ask">{format_value(row['put_ask'])}</span>
                <span>{format_value(row['put_last'])}</span>
                <span class="{put_vol_class}">{format_value(row['put_volume'], 0)}</span>
                <span>{format_value(row['put_oi'], 0)}</span>
                <span class="{put_iv_class}">{format_value(put_iv * 100 if put_iv else None, 1)}%</span>
                <span class="delta-col">{put_delta}</span>
            """
        else:
            call_data = f"""
                <span class="{call_iv_class}">{format_value(call_iv * 100 if call_iv else None, 1)}%</span>
                <span>{format_value(row['call_oi'], 0)}</span>
                <span class="{call_vol_class}">{format_value(row['call_volume'], 0)}</span>
                <span>{format_value(row['call_last'])}</span>
                <span class="bid-ask">{format_value(row['call_bid'])}</span>
                <span class="bid-ask">{format_value(row['call_ask'])}</span>
                <span></span>
            """
            put_data = f"""
                <span></span>
                <span class="bid-ask">{format_value(row['put_bid'])}</span>
                <span class="bid-ask">{format_value(row['put_ask'])}</span>
                <span>{format_value(row['put_last'])}</span>
                <span class="{put_vol_class}">{format_value(row['put_volume'], 0)}</span>
                <span>{format_value(row['put_oi'], 0)}</span>
                <span class="{put_iv_class}">{format_value(put_iv * 100 if put_iv else None, 1)}%</span>
            """
        
        html += f"""
        <div class="grid-row {row_class}">
            <div class="calls-data">{call_data}</div>
            <div class="strike-cell">{format_value(row['strike'])}</div>
            <div class="puts-data">{put_data}</div>
        </div>
        """
    
    html += "</div>"
    return html


# ============================================================================
# MAIN APP
# ============================================================================

provider, provider_name = get_provider()
is_realtime = (provider_name == "Schwab" or 
               (provider_name == "Tradier" and "api.tradier.com" in TRADIER_ENDPOINT))

if provider_name == "Schwab":
    indicator_html = '<div class="live-indicator"><div class="live-dot"></div><span>SCHWAB LIVE</span></div>'
elif is_realtime:
    indicator_html = '<div class="live-indicator"><div class="live-dot"></div><span>REAL-TIME</span></div>'
elif provider_name == "Tradier":
    indicator_html = '<div class="delayed-indicator"><div class="delayed-dot"></div><span>SANDBOX</span></div>'
else:
    indicator_html = '<div class="delayed-indicator"><div class="delayed-dot"></div><span>DELAYED</span></div>'

st.markdown(f"""
<div class="terminal-header">
    <div>
        <div class="terminal-title">Options Grid</div>
        <div class="terminal-subtitle">Professional Options Chain</div>
    </div>
    {indicator_html}
</div>
""", unsafe_allow_html=True)

if not USE_SCHWAB and not USE_TRADIER:
    with st.expander("🔑 Enable Real-Time Data", expanded=False):
        st.markdown("""
        ### Option 1: Charles Schwab (Recommended - Free with account)
        
        1. Create developer account at [developer.schwab.com](https://developer.schwab.com/)
        2. Register an app with "Market Data Production"
        3. Set callback URL: `https://127.0.0.1:8182`
        4. Wait 2-3 days for approval
        5. Run initial auth (first time only):
        ```bash
        pip install schwab-py
        python -c "
        import schwab
        schwab.auth.easy_client('YOUR_APP_KEY', 'YOUR_APP_SECRET', 
                                'https://127.0.0.1:8182', 'schwab_tokens.json')
        "
        ```
        6. Set environment variables:
        ```bash
        export SCHWAB_APP_KEY="your-app-key"
        export SCHWAB_APP_SECRET="your-app-secret"
        ```
        
        ---
        
        ### Option 2: Tradier (Simpler setup)
        
        1. Sign up free at [developer.tradier.com](https://developer.tradier.com/)
        2. For Streamlit Cloud, add to `.streamlit/secrets.toml`:
        ```toml
        TRADIER_API_KEY = "your-api-key"
        TRADIER_ENDPOINT = "https://api.tradier.com/v1"
        ```
        3. For local dev: `export TRADIER_API_KEY="your-key"`
        """)

col1, col2, col3, col4 = st.columns([2, 1, 1, 6])

with col1:
    ticker = st.text_input("Symbol", value="SPY", label_visibility="collapsed", key="ticker").upper().strip()

with col2:
    auto_refresh = st.checkbox("Auto-refresh", value=False)

with col3:
    if st.button("🔄"):
        st.cache_data.clear()

if auto_refresh:
    time.sleep(5)
    st.rerun()

if ticker:
    stock_data = get_stock_data(ticker)
    
    if stock_data and stock_data.get("price"):
        change = stock_data.get('change', 0) or 0
        change_pct = stock_data.get('change_pct', 0) or 0
        change_class = "positive" if change >= 0 else "negative"
        change_sign = "+" if change >= 0 else ""
        
        day_low = stock_data.get('day_low')
        day_high = stock_data.get('day_high')
        range_str = f"${day_low:.2f} - ${day_high:.2f}" if day_low and day_high else "—"
        
        st.markdown(f"""
        <div class="stock-bar">
            <div>
                <div class="stock-symbol">{stock_data['symbol']}</div>
                <div class="stock-name">{stock_data['name']}</div>
            </div>
            <div>
                <div class="stock-price {change_class}">${stock_data['price']:.2f}</div>
                <div class="stock-change {change_class}">{change_sign}{change:.2f} ({change_sign}{change_pct:.2f}%)</div>
            </div>
            <div class="stock-stat">
                <div class="stock-stat-value">{format_value(stock_data.get('bid'))}/{format_value(stock_data.get('ask'))}</div>
                <div class="stock-stat-label">Bid/Ask</div>
            </div>
            <div class="stock-stat">
                <div class="stock-stat-value">{format_value(stock_data['volume'], 0)}</div>
                <div class="stock-stat-label">Volume</div>
            </div>
            <div class="stock-stat">
                <div class="stock-stat-value">{range_str}</div>
                <div class="stock-stat-label">Day Range</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        expirations = get_expirations(ticker)
        
        if expirations:
            exp_options = [f"{exp} ({calculate_days_to_expiry(exp)}d)" for exp in expirations[:15]]
            
            col1, col2, col3 = st.columns([3, 2, 5])
            
            with col1:
                selected_exp_display = st.selectbox("Expiration", options=exp_options, label_visibility="collapsed")
                selected_exp = selected_exp_display.split(" (")[0]
            
            with col2:
                strike_range = st.slider("Range %", 5, 50, 8, 5, label_visibility="collapsed", help="Strike range from ATM")
            
            calls, puts = get_options_data(ticker, selected_exp)
            
            if calls is not None and puts is not None and not calls.empty and not puts.empty:
                grid_df = build_options_grid(calls, puts, stock_data['price'])
                
                if grid_df is not None and not grid_df.empty:
                    current_price = stock_data['price']
                    min_strike = current_price * (1 - strike_range / 100)
                    max_strike = current_price * (1 + strike_range / 100)
                    grid_df = grid_df[(grid_df['strike'] >= min_strike) & (grid_df['strike'] <= max_strike)]
                    
                    # Show applied range
                    st.caption(f"📊 Showing strikes ${min_strike:.0f} - ${max_strike:.0f} ({len(grid_df)} strikes)")
                    
                    # Filter: require a real two-sided market (bid > 0 AND ask > 0) on at least one side
                    def has_real_market(row):
                        call_liquid = (pd.notna(row['call_bid']) and row['call_bid'] > 0.01 and 
                                      pd.notna(row['call_ask']) and row['call_ask'] > 0)
                        put_liquid = (pd.notna(row['put_bid']) and row['put_bid'] > 0.01 and 
                                     pd.notna(row['put_ask']) and row['put_ask'] > 0)
                        return call_liquid or put_liquid
                    
                    grid_df = grid_df[grid_df.apply(has_real_market, axis=1)]
                    
                    total_call_oi = grid_df['call_oi'].sum() or 0
                    total_put_oi = grid_df['put_oi'].sum() or 0
                    total_call_vol = grid_df['call_volume'].sum() or 0
                    total_put_vol = grid_df['put_volume'].sum() or 0
                    put_call_ratio = total_put_oi / total_call_oi if total_call_oi > 0 else 0
                    
                    # Calculate dynamic IV threshold (1.5x median IV of the chain)
                    iv_vals = [v for v in grid_df[['call_iv', 'put_iv']].values.flatten() if pd.notna(v) and v > 0]
                    median_iv = np.median(iv_vals) if iv_vals else 0.3
                    iv_threshold = median_iv * 1.5  # Highlight IVs 50% above median
                    avg_iv = (sum(iv_vals) / len(iv_vals) * 100) if iv_vals else 0
                    
                    st.markdown(f"""
                    <div class="stats-row">
                        <div class="stat-card"><div class="stat-value">{format_value(total_call_oi, 0)}</div><div class="stat-label">Call OI</div></div>
                        <div class="stat-card"><div class="stat-value">{format_value(total_put_oi, 0)}</div><div class="stat-label">Put OI</div></div>
                        <div class="stat-card"><div class="stat-value">{put_call_ratio:.2f}</div><div class="stat-label">P/C Ratio</div></div>
                        <div class="stat-card"><div class="stat-value">{format_value(total_call_vol, 0)}</div><div class="stat-label">Call Vol</div></div>
                        <div class="stat-card"><div class="stat-value">{format_value(total_put_vol, 0)}</div><div class="stat-label">Put Vol</div></div>
                        <div class="stat-card"><div class="stat-value">{avg_iv:.1f}%</div><div class="stat-label">Avg IV (>{iv_threshold*100:.0f}% ⚠)</div></div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(render_options_grid(grid_df, current_price, iv_threshold), unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div class="timestamp">
                        {datetime.now().strftime("%H:%M:%S")} • {len(grid_df)} strikes
                        <span class="data-source">{provider_name}</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("Could not load options chain.")
        else:
            st.warning(f"No options for {ticker}")
    else:
        st.error(f"Could not find: {ticker}")