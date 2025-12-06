# app.py - Your Options Screener Dashboard

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="Options Screener Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2E86AB;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">📊 Options Screener Pro</p>', unsafe_allow_html=True)

# Sidebar - Screening Parameters
st.sidebar.header("🎯 Screening Parameters")

# Ticker input
ticker_input = st.sidebar.text_input(
    "Tickers (comma-separated)",
    "SPY,QQQ,IWM,AAPL,MSFT,NVDA"
)
tickers = [t.strip().upper() for t in ticker_input.split(',')]

# Strategy selection
strategy = st.sidebar.selectbox(
    "Strategy",
    ["Put Credit Spreads", "Call Credit Spreads", "Iron Condors", "Butterflies"]
)

# Spread parameters
st.sidebar.subheader("Spread Settings")
spread_width = st.sidebar.slider("Spread Width ($)", 1, 20, 5)
min_credit = st.sidebar.slider("Min Credit ($)", 0.1, 5.0, 0.35, 0.05)

# Time parameters
st.sidebar.subheader("Time Settings")
dte_min = st.sidebar.slider("Min DTE", 7, 60, 30)
dte_max = st.sidebar.slider("Max DTE", 7, 90, 45)

# Quality filters
st.sidebar.subheader("Quality Filters")
min_prob_profit = st.sidebar.slider("Min Prob of Profit", 0.5, 0.95, 0.70, 0.05)
min_expected_value = st.sidebar.slider("Min Expected Value ($)", 0.0, 1.0, 0.15, 0.05)
min_volume = st.sidebar.slider("Min Option Volume", 0, 500, 100, 50)

# Run screener button
run_button = st.sidebar.button("🔍 Run Screener", type="primary", use_container_width=True)

# Main content area
if run_button:
    with st.spinner('🔍 Scanning options chains...'):
        # Import your screening functions here
        # results = screen_multiple_tickers(tickers, ...)
        
        # For demo, create sample data
        results = pd.DataFrame({
            'ticker': ['SPY', 'QQQ', 'AAPL', 'MSFT', 'NVDA'] * 3,
            'expiration': ['2024-01-19'] * 15,
            'dte': [45] * 15,
            'short_strike': [445, 430, 185, 390, 520] * 3,
            'long_strike': [440, 425, 180, 385, 515] * 3,
            'credit': np.random.uniform(0.3, 0.8, 15),
            'prob_profit': np.random.uniform(0.65, 0.85, 15),
            'expected_value': np.random.uniform(0.1, 0.4, 15),
            'return_on_risk': np.random.uniform(15, 35, 15)
        })
    
    # Display results
    st.success(f"✅ Found {len(results)} opportunities!")
    
    # Top metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Opportunities", len(results))
    with col2:
        st.metric("Avg Expected Value", f"${results['expected_value'].mean():.2f}")
    with col3:
        st.metric("Avg Prob of Profit", f"{results['prob_profit'].mean():.1%}")
    with col4:
        st.metric("Avg Return on Risk", f"{results['return_on_risk'].mean():.1f}%")
    
    # Results table
    st.subheader("📋 Top Opportunities")
    
    # Format the dataframe
    display_df = results.copy()
    display_df['credit'] = display_df['credit'].apply(lambda x: f"${x:.2f}")
    display_df['prob_profit'] = display_df['prob_profit'].apply(lambda x: f"{x:.1%}")
    display_df['expected_value'] = display_df['expected_value'].apply(lambda x: f"${x:.2f}")
    display_df['return_on_risk'] = display_df['return_on_risk'].apply(lambda x: f"{x:.1f}%")
    
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "ticker": st.column_config.TextColumn("Ticker", width="small"),
            "expiration": st.column_config.DateColumn("Expiration"),
            "dte": st.column_config.NumberColumn("DTE", width="small"),
            "short_strike": st.column_config.NumberColumn("Short Strike", format="$%.2f"),
            "long_strike": st.column_config.NumberColumn("Long Strike", format="$%.2f"),
            "credit": "Credit",
            "prob_profit": "Prob Profit",
            "expected_value": "Expected Value",
            "return_on_risk": "Return on Risk"
        }
    )
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Expected Value by Ticker
        fig = px.bar(
            results.groupby('ticker')['expected_value'].mean().reset_index(),
            x='ticker',
            y='expected_value',
            title='Average Expected Value by Ticker',
            color='expected_value',
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Prob of Profit Distribution
        fig = px.histogram(
            results,
            x='prob_profit',
            title='Probability of Profit Distribution',
            nbins=20,
            color_discrete_sequence=['#2E86AB']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Download button
    csv = results.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Results (CSV)",
        data=csv,
        file_name=f'screener_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
        mime='text/csv',
    )

else:
    # Welcome screen
    st.info("👈 Configure your screening parameters in the sidebar and click 'Run Screener'")
    
    st.markdown("""
    ### 🎯 How to Use This Screener
    
    1. **Enter tickers** in the sidebar (comma-separated)
    2. **Select strategy** (Put Credit Spreads, Iron Condors, etc.)
    3. **Adjust parameters** to match your risk tolerance
    4. **Click 'Run Screener'** to find opportunities
    5. **Review results** and download to Excel
    
    ### 📊 What This Screener Does
    
    - Scans real-time options chains
    - Calculates Greeks (Delta, Theta, Vega)
    - Estimates probability of profit
    - Calculates expected value
    - Ranks opportunities by risk-adjusted returns
    
    ### 🚀 Features Coming Soon
    
    - Monte Carlo simulation
    - Historical backtesting
    - Live trade execution
    - Position tracking
    """)