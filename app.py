"""
Options Screener - A clean, minimal options chain viewer
"""

import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Options Screener",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    
    .sub-header {
        font-size: 1rem;
        color: #6b7280;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
    }
    
    .metric-value {
        font-size: 1.75rem;
        font-weight: 600;
        color: #1a1a2e;
    }
    
    .metric-label {
        font-size: 0.75rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.25rem;
    }
    
    .stDataFrame {
        border-radius: 8px;
    }
    
    div[data-testid="stDataFrame"] > div {
        border-radius: 8px;
    }
    
    .section-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #1a1a2e;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e2e8f0;
    }
    
    /* Clean up Streamlit defaults */
    .stButton > button {
        background: #1a1a2e;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 500;
        font-family: 'Inter', sans-serif;
    }
    
    .stButton > button:hover {
        background: #2d2d44;
    }
    
    .stSelectbox > div > div {
        border-radius: 8px;
    }
    
    .stTextInput > div > div > input {
        border-radius: 8px;
        font-family: 'Inter', sans-serif;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-family: 'Inter', sans-serif;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


def format_number(val, decimals=2):
    """Format numbers with proper handling of None/NaN"""
    if pd.isna(val) or val is None:
        return "—"
    if isinstance(val, (int, float)):
        if abs(val) >= 1_000_000:
            return f"{val/1_000_000:.1f}M"
        elif abs(val) >= 1_000:
            return f"{val/1_000:.1f}K"
        return f"{val:,.{decimals}f}"
    return str(val)


def format_percent(val):
    """Format as percentage"""
    if pd.isna(val) or val is None:
        return "—"
    return f"{val * 100:.1f}%"


def get_stock_info(ticker: str) -> dict:
    """Fetch basic stock information"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            "name": info.get("shortName", ticker),
            "price": info.get("regularMarketPrice") or info.get("currentPrice"),
            "change": info.get("regularMarketChangePercent"),
            "volume": info.get("regularMarketVolume"),
            "market_cap": info.get("marketCap"),
        }
    except Exception:
        return None


def get_options_chain(ticker: str, expiration: str) -> tuple:
    """Fetch options chain for given ticker and expiration"""
    try:
        stock = yf.Ticker(ticker)
        chain = stock.option_chain(expiration)
        return chain.calls, chain.puts
    except Exception as e:
        st.error(f"Error fetching options: {e}")
        return None, None


def get_expirations(ticker: str) -> list:
    """Get available expiration dates"""
    try:
        stock = yf.Ticker(ticker)
        return list(stock.options)
    except Exception:
        return []


def format_chain_df(df: pd.DataFrame, current_price: float) -> pd.DataFrame:
    """Format options chain DataFrame for display"""
    if df is None or df.empty:
        return pd.DataFrame()
    
    # Select and rename columns
    columns = {
        "strike": "Strike",
        "lastPrice": "Last",
        "bid": "Bid",
        "ask": "Ask",
        "volume": "Volume",
        "openInterest": "Open Int",
        "impliedVolatility": "IV",
    }
    
    available_cols = [c for c in columns.keys() if c in df.columns]
    result = df[available_cols].copy()
    result = result.rename(columns={k: columns[k] for k in available_cols})
    
    # Add ITM/OTM indicator
    if "Strike" in result.columns and current_price:
        result["Moneyness"] = result["Strike"].apply(
            lambda x: "ITM" if x < current_price else ("ATM" if abs(x - current_price) / current_price < 0.01 else "OTM")
        )
    
    return result


# Main UI
st.markdown('<p class="main-header">Options Screener</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">View options chains with key metrics</p>', unsafe_allow_html=True)

# Input section
col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    ticker = st.text_input(
        "Ticker Symbol",
        value="AAPL",
        placeholder="Enter ticker (e.g., AAPL, SPY)",
        label_visibility="collapsed"
    ).upper().strip()

# Fetch data when ticker is entered
if ticker:
    # Get stock info
    stock_info = get_stock_info(ticker)
    
    if stock_info and stock_info.get("price"):
        # Display stock metrics
        st.markdown('<p class="section-title">Stock Overview</p>', unsafe_allow_html=True)
        
        m1, m2, m3, m4 = st.columns(4)
        
        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">${format_number(stock_info['price'])}</div>
                <div class="metric-label">Current Price</div>
            </div>
            """, unsafe_allow_html=True)
        
        with m2:
            change = stock_info.get('change')
            change_color = "#10b981" if change and change >= 0 else "#ef4444"
            change_str = f"+{change:.2f}%" if change and change >= 0 else f"{change:.2f}%" if change else "—"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color: {change_color}">{change_str}</div>
                <div class="metric-label">Day Change</div>
            </div>
            """, unsafe_allow_html=True)
        
        with m3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{format_number(stock_info['volume'], 0)}</div>
                <div class="metric-label">Volume</div>
            </div>
            """, unsafe_allow_html=True)
        
        with m4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">${format_number(stock_info['market_cap'], 0)}</div>
                <div class="metric-label">Market Cap</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Get expirations
        expirations = get_expirations(ticker)
        
        if expirations:
            st.markdown('<p class="section-title">Options Chain</p>', unsafe_allow_html=True)
            
            # Expiration selector
            col1, col2 = st.columns([3, 1])
            with col1:
                selected_exp = st.selectbox(
                    "Expiration Date",
                    options=expirations,
                    format_func=lambda x: datetime.strptime(x, "%Y-%m-%d").strftime("%B %d, %Y"),
                    label_visibility="collapsed"
                )
            
            if selected_exp:
                calls, puts = get_options_chain(ticker, selected_exp)
                current_price = stock_info['price']
                
                # Tabs for calls/puts
                tab_calls, tab_puts = st.tabs(["📈 Calls", "📉 Puts"])
                
                with tab_calls:
                    if calls is not None and not calls.empty:
                        calls_formatted = format_chain_df(calls, current_price)
                        st.dataframe(
                            calls_formatted,
                            use_container_width=True,
                            hide_index=True,
                            height=400
                        )
                    else:
                        st.info("No call options available for this expiration.")
                
                with tab_puts:
                    if puts is not None and not puts.empty:
                        puts_formatted = format_chain_df(puts, current_price)
                        st.dataframe(
                            puts_formatted,
                            use_container_width=True,
                            hide_index=True,
                            height=400
                        )
                    else:
                        st.info("No put options available for this expiration.")
        else:
            st.warning(f"No options available for {ticker}")
    else:
        st.error(f"Could not find data for ticker: {ticker}")

# Footer
st.markdown("---")
st.markdown(
    '<p style="text-align: center; color: #9ca3af; font-size: 0.875rem;">'
    'Data provided by Yahoo Finance • For educational purposes only'
    '</p>',
    unsafe_allow_html=True
)