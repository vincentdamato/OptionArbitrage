"""
Options Screener - Professional Options Grid
Real-time options chain with calls/puts grid layout
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

# Page configuration
st.set_page_config(
    page_title="Options Grid",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Professional trading terminal CSS
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
    
    /* Hide Streamlit branding */
    #MainMenu, footer, header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Header styling */
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
    
    .live-dot {
        width: 8px;
        height: 8px;
        background: var(--accent-green);
        border-radius: 50%;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(34, 197, 94, 0.7); }
        50% { opacity: 0.8; box-shadow: 0 0 0 6px rgba(34, 197, 94, 0); }
    }
    
    /* Stock info bar */
    .stock-bar {
        display: flex;
        align-items: center;
        gap: 2rem;
        padding: 1rem 1.5rem;
        background: var(--bg-secondary);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        margin-bottom: 1.5rem;
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
    
    /* Options grid container */
    .grid-container {
        background: var(--bg-secondary);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        overflow: hidden;
    }
    
    .grid-header {
        display: grid;
        grid-template-columns: 1fr auto 1fr;
        background: var(--bg-tertiary);
        border-bottom: 1px solid var(--border-color);
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .grid-header-calls, .grid-header-puts {
        display: grid;
        grid-template-columns: repeat(6, 1fr);
        padding: 0.75rem 1rem;
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
        padding: 0.75rem 1rem;
        text-align: center;
        color: var(--accent-blue);
        background: rgba(59, 130, 246, 0.1);
        min-width: 100px;
    }
    
    .grid-header span {
        text-align: right;
    }
    
    .grid-header span:first-child {
        text-align: left;
    }
    
    /* Grid rows */
    .grid-row {
        display: grid;
        grid-template-columns: 1fr auto 1fr;
        border-bottom: 1px solid var(--border-color);
        font-size: 0.8125rem;
        transition: background 0.15s ease;
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
        border-top: 1px solid var(--accent-blue);
        border-bottom: 1px solid var(--accent-blue);
    }
    
    .calls-data, .puts-data {
        display: grid;
        grid-template-columns: repeat(6, 1fr);
        padding: 0.5rem 1rem;
        align-items: center;
    }
    
    .calls-data span, .puts-data span {
        text-align: right;
        color: var(--text-secondary);
    }
    
    .calls-data span:first-child, .puts-data span:first-child {
        text-align: left;
    }
    
    .strike-cell {
        padding: 0.5rem 1rem;
        text-align: center;
        font-weight: 600;
        color: var(--text-primary);
        background: var(--bg-tertiary);
        min-width: 100px;
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
    
    /* Expiration selector */
    .exp-container {
        display: flex;
        gap: 0.5rem;
        margin-bottom: 1rem;
        flex-wrap: wrap;
    }
    
    .exp-btn {
        padding: 0.5rem 1rem;
        background: var(--bg-secondary);
        border: 1px solid var(--border-color);
        border-radius: 6px;
        color: var(--text-secondary);
        font-size: 0.75rem;
        cursor: pointer;
        transition: all 0.15s ease;
    }
    
    .exp-btn:hover {
        border-color: var(--accent-blue);
        color: var(--text-primary);
    }
    
    .exp-btn.active {
        background: var(--accent-blue);
        border-color: var(--accent-blue);
        color: white;
    }
    
    /* Controls bar */
    .controls-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
        padding: 0.75rem 1rem;
        background: var(--bg-secondary);
        border: 1px solid var(--border-color);
        border-radius: 8px;
    }
    
    .filter-group {
        display: flex;
        gap: 1rem;
        align-items: center;
    }
    
    .filter-label {
        font-size: 0.75rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Streamlit overrides */
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
    
    /* Stats summary */
    .stats-row {
        display: grid;
        grid-template-columns: repeat(6, 1fr);
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    
    .stat-card {
        background: var(--bg-secondary);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    
    .stat-value {
        font-size: 1.25rem;
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
    
    /* Last update timestamp */
    .timestamp {
        font-size: 0.75rem;
        color: var(--text-muted);
        text-align: right;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# DATA FUNCTIONS
# ============================================================================

@st.cache_data(ttl=30)  # Cache for 30 seconds for near real-time
def get_stock_data(ticker: str) -> dict:
    """Fetch stock data with caching"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        hist = stock.history(period="2d")
        
        current_price = info.get("regularMarketPrice") or info.get("currentPrice")
        prev_close = info.get("regularMarketPreviousClose") or (hist['Close'].iloc[-2] if len(hist) > 1 else current_price)
        
        change = current_price - prev_close if current_price and prev_close else 0
        change_pct = (change / prev_close * 100) if prev_close else 0
        
        return {
            "symbol": ticker,
            "name": info.get("shortName", ticker),
            "price": current_price,
            "change": change,
            "change_pct": change_pct,
            "volume": info.get("regularMarketVolume", 0),
            "day_high": info.get("dayHigh"),
            "day_low": info.get("dayLow"),
            "iv_30": info.get("impliedVolatility"),  # May not always be available
            "market_cap": info.get("marketCap"),
        }
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
        return None


@st.cache_data(ttl=30)
def get_options_data(ticker: str, expiration: str) -> tuple:
    """Fetch options chain data"""
    try:
        stock = yf.Ticker(ticker)
        chain = stock.option_chain(expiration)
        return chain.calls, chain.puts
    except Exception as e:
        st.error(f"Error fetching options: {e}")
        return None, None


@st.cache_data(ttl=300)  # Cache expirations for 5 minutes
def get_expirations(ticker: str) -> list:
    """Get available expiration dates"""
    try:
        stock = yf.Ticker(ticker)
        return list(stock.options)
    except Exception:
        return []


def calculate_days_to_expiry(exp_date: str) -> int:
    """Calculate days until expiration"""
    exp = datetime.strptime(exp_date, "%Y-%m-%d")
    today = datetime.now()
    return (exp - today).days


def build_options_grid(calls: pd.DataFrame, puts: pd.DataFrame, current_price: float) -> pd.DataFrame:
    """Build unified options grid with calls and puts aligned by strike"""
    if calls is None or puts is None:
        return None
    
    # Get all unique strikes
    all_strikes = sorted(set(calls['strike'].tolist() + puts['strike'].tolist()))
    
    # Build grid data
    grid_data = []
    
    for strike in all_strikes:
        call_row = calls[calls['strike'] == strike]
        put_row = puts[puts['strike'] == strike]
        
        # Determine moneyness
        if current_price:
            pct_diff = (strike - current_price) / current_price
            if abs(pct_diff) < 0.005:
                moneyness = "ATM"
            elif strike < current_price:
                moneyness = "ITM_CALL"
            else:
                moneyness = "ITM_PUT"
        else:
            moneyness = "OTM"
        
        row = {
            "strike": strike,
            "moneyness": moneyness,
            # Call data
            "call_bid": call_row['bid'].iloc[0] if len(call_row) > 0 else None,
            "call_ask": call_row['ask'].iloc[0] if len(call_row) > 0 else None,
            "call_last": call_row['lastPrice'].iloc[0] if len(call_row) > 0 else None,
            "call_volume": call_row['volume'].iloc[0] if len(call_row) > 0 else None,
            "call_oi": call_row['openInterest'].iloc[0] if len(call_row) > 0 else None,
            "call_iv": call_row['impliedVolatility'].iloc[0] if len(call_row) > 0 else None,
            # Put data
            "put_bid": put_row['bid'].iloc[0] if len(put_row) > 0 else None,
            "put_ask": put_row['ask'].iloc[0] if len(put_row) > 0 else None,
            "put_last": put_row['lastPrice'].iloc[0] if len(put_row) > 0 else None,
            "put_volume": put_row['volume'].iloc[0] if len(put_row) > 0 else None,
            "put_oi": put_row['openInterest'].iloc[0] if len(put_row) > 0 else None,
            "put_iv": put_row['impliedVolatility'].iloc[0] if len(put_row) > 0 else None,
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


def render_options_grid(grid_df: pd.DataFrame, current_price: float, iv_threshold: float = 0.5):
    """Render the options grid as HTML"""
    if grid_df is None or grid_df.empty:
        return "<p>No options data available</p>"
    
    # Find max volume and OI for highlighting
    max_call_vol = grid_df['call_volume'].max() if grid_df['call_volume'].notna().any() else 1
    max_put_vol = grid_df['put_volume'].max() if grid_df['put_volume'].notna().any() else 1
    
    html = """
    <div class="grid-container">
        <div class="grid-header">
            <div class="grid-header-calls">
                <span>IV</span>
                <span>OI</span>
                <span>Vol</span>
                <span>Last</span>
                <span>Bid</span>
                <span>Ask</span>
            </div>
            <div class="grid-header-strike">STRIKE</div>
            <div class="grid-header-puts">
                <span>Bid</span>
                <span>Ask</span>
                <span>Last</span>
                <span>Vol</span>
                <span>OI</span>
                <span>IV</span>
            </div>
        </div>
    """
    
    for _, row in grid_df.iterrows():
        # Determine row class based on moneyness
        row_class = ""
        if row['moneyness'] == "ATM":
            row_class = "atm"
        elif row['moneyness'] == "ITM_CALL":
            row_class = "itm-call"
        elif row['moneyness'] == "ITM_PUT":
            row_class = "itm-put"
        
        # Format call values
        call_iv = row['call_iv']
        call_iv_class = "iv-high" if call_iv and call_iv > iv_threshold else ""
        call_vol_class = "volume-high" if row['call_volume'] and row['call_volume'] > max_call_vol * 0.5 else ""
        
        # Format put values
        put_iv = row['put_iv']
        put_iv_class = "iv-high" if put_iv and put_iv > iv_threshold else ""
        put_vol_class = "volume-high" if row['put_volume'] and row['put_volume'] > max_put_vol * 0.5 else ""
        
        html += f"""
        <div class="grid-row {row_class}">
            <div class="calls-data">
                <span class="{call_iv_class}">{format_value(call_iv * 100 if call_iv else None, 1)}%</span>
                <span>{format_value(row['call_oi'], 0)}</span>
                <span class="{call_vol_class}">{format_value(row['call_volume'], 0)}</span>
                <span>{format_value(row['call_last'])}</span>
                <span class="bid-ask">{format_value(row['call_bid'])}</span>
                <span class="bid-ask">{format_value(row['call_ask'])}</span>
            </div>
            <div class="strike-cell">{format_value(row['strike'])}</div>
            <div class="puts-data">
                <span class="bid-ask">{format_value(row['put_bid'])}</span>
                <span class="bid-ask">{format_value(row['put_ask'])}</span>
                <span>{format_value(row['put_last'])}</span>
                <span class="{put_vol_class}">{format_value(row['put_volume'], 0)}</span>
                <span>{format_value(row['put_oi'], 0)}</span>
                <span class="{put_iv_class}">{format_value(put_iv * 100 if put_iv else None, 1)}%</span>
            </div>
        </div>
        """
    
    html += "</div>"
    return html


# ============================================================================
# MAIN APP
# ============================================================================

# Header
st.markdown("""
<div class="terminal-header">
    <div>
        <div class="terminal-title">Options Grid</div>
        <div class="terminal-subtitle">Real-time Options Chain</div>
    </div>
    <div class="live-indicator">
        <div class="live-dot"></div>
        <span>LIVE</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Ticker input
col1, col2, col3 = st.columns([2, 2, 6])
with col1:
    ticker = st.text_input(
        "Symbol",
        value="SPY",
        placeholder="Enter symbol",
        label_visibility="collapsed",
        key="ticker_input"
    ).upper().strip()

# Auto-refresh toggle
with col2:
    auto_refresh = st.checkbox("Auto-refresh (30s)", value=False)

if auto_refresh:
    time.sleep(0.1)  # Small delay to prevent rapid refreshes
    st.rerun()

# Main content
if ticker:
    # Fetch stock data
    stock_data = get_stock_data(ticker)
    
    if stock_data and stock_data.get("price"):
        # Stock info bar
        change_class = "positive" if stock_data['change'] >= 0 else "negative"
        change_sign = "+" if stock_data['change'] >= 0 else ""
        
        st.markdown(f"""
        <div class="stock-bar">
            <div>
                <div class="stock-symbol">{stock_data['symbol']}</div>
                <div class="stock-name">{stock_data['name']}</div>
            </div>
            <div>
                <div class="stock-price {change_class}">${stock_data['price']:.2f}</div>
                <div class="stock-change {change_class}">{change_sign}{stock_data['change']:.2f} ({change_sign}{stock_data['change_pct']:.2f}%)</div>
            </div>
            <div class="stock-stat">
                <div class="stock-stat-value">{format_value(stock_data['volume'], 0)}</div>
                <div class="stock-stat-label">Volume</div>
            </div>
            <div class="stock-stat">
                <div class="stock-stat-value">${stock_data['day_low']:.2f} - ${stock_data['day_high']:.2f}</div>
                <div class="stock-stat-label">Day Range</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Get expirations
        expirations = get_expirations(ticker)
        
        if expirations:
            # Expiration selector with DTE
            exp_options = []
            for exp in expirations[:12]:  # Show first 12 expirations
                dte = calculate_days_to_expiry(exp)
                exp_options.append(f"{exp} ({dte}d)")
            
            col1, col2, col3 = st.columns([3, 2, 5])
            
            with col1:
                selected_exp_display = st.selectbox(
                    "Expiration",
                    options=exp_options,
                    label_visibility="collapsed"
                )
                selected_exp = selected_exp_display.split(" (")[0]
            
            with col2:
                strike_range = st.slider(
                    "Strike Range (%)",
                    min_value=5,
                    max_value=50,
                    value=15,
                    step=5,
                    label_visibility="collapsed",
                    help="Show strikes within X% of current price"
                )
            
            # Fetch options data
            calls, puts = get_options_data(ticker, selected_exp)
            
            if calls is not None and puts is not None:
                # Build and filter grid
                grid_df = build_options_grid(calls, puts, stock_data['price'])
                
                if grid_df is not None:
                    # Filter by strike range
                    current_price = stock_data['price']
                    min_strike = current_price * (1 - strike_range / 100)
                    max_strike = current_price * (1 + strike_range / 100)
                    grid_df = grid_df[(grid_df['strike'] >= min_strike) & (grid_df['strike'] <= max_strike)]
                    
                    # Summary stats
                    total_call_oi = grid_df['call_oi'].sum()
                    total_put_oi = grid_df['put_oi'].sum()
                    total_call_vol = grid_df['call_volume'].sum()
                    total_put_vol = grid_df['put_volume'].sum()
                    put_call_ratio = total_put_oi / total_call_oi if total_call_oi > 0 else 0
                    avg_iv = grid_df[['call_iv', 'put_iv']].mean().mean() * 100
                    
                    st.markdown(f"""
                    <div class="stats-row">
                        <div class="stat-card">
                            <div class="stat-value">{format_value(total_call_oi, 0)}</div>
                            <div class="stat-label">Call OI</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">{format_value(total_put_oi, 0)}</div>
                            <div class="stat-label">Put OI</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">{put_call_ratio:.2f}</div>
                            <div class="stat-label">P/C Ratio</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">{format_value(total_call_vol, 0)}</div>
                            <div class="stat-label">Call Vol</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">{format_value(total_put_vol, 0)}</div>
                            <div class="stat-label">Put Vol</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">{avg_iv:.1f}%</div>
                            <div class="stat-label">Avg IV</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Render options grid
                    grid_html = render_options_grid(grid_df, current_price)
                    st.markdown(grid_html, unsafe_allow_html=True)
                    
                    # Timestamp
                    st.markdown(f"""
                    <div class="timestamp">
                        Last updated: {datetime.now().strftime("%H:%M:%S")} • {len(grid_df)} strikes shown
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("Could not load options data for this expiration.")
        else:
            st.warning(f"No options available for {ticker}")
    else:
        st.error(f"Could not find data for: {ticker}")