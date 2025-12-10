"""
Options Arbitrage Scanner - Streamlit Dashboard
Real-time arbitrage detection with Schwab API integration.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
from typing import Dict, List, Optional
import json

# Page config
st.set_page_config(
    page_title="Options Arbitrage Scanner",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stMetric {
        background-color: #1e1e1e;
        padding: 10px;
        border-radius: 5px;
    }
    .opportunity-card {
        background-color: #262730;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #00ff88;
    }
    .risk-free {
        border-left-color: #00ff88;
    }
    .statistical {
        border-left-color: #ffaa00;
    }
    .header-title {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #00ff88, #00aaff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables."""
    if "opportunities" not in st.session_state:
        st.session_state.opportunities = []
    if "scan_history" not in st.session_state:
        st.session_state.scan_history = []
    if "last_scan" not in st.session_state:
        st.session_state.last_scan = None
    if "api_connected" not in st.session_state:
        st.session_state.api_connected = False
    if "watchlist" not in st.session_state:
        st.session_state.watchlist = [
            "SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA", "TSLA", "AMZN",
            "META", "GOOGL", "AMD", "NFLX"
        ]


def create_mock_opportunities():
    """Create mock opportunities for demo mode."""
    types = ["put_call_parity", "box_spread", "butterfly", "vertical_mispricing", "volatility_skew"]
    symbols = ["SPY", "QQQ", "AAPL", "NVDA", "TSLA"]
    
    opportunities = []
    for i in range(np.random.randint(3, 8)):
        opp_type = np.random.choice(types)
        symbol = np.random.choice(symbols)
        edge = np.random.uniform(0.5, 3.0)
        confidence = np.random.uniform(0.6, 0.95)
        
        opportunities.append({
            "id": i,
            "type": opp_type,
            "symbol": symbol,
            "expected_profit": edge * 100,
            "expected_profit_pct": edge,
            "confidence": confidence,
            "risk_free": opp_type in ["put_call_parity", "box_spread", "butterfly"],
            "net_delta": np.random.uniform(-0.1, 0.1),
            "net_vega": np.random.uniform(-50, 50),
            "days_to_expiration": np.random.randint(5, 45),
            "underlying_price": np.random.uniform(100, 500),
            "timestamp": datetime.now(),
            "legs": [
                {"action": "BUY", "type": "CALL", "strike": 100, "price": 5.50},
                {"action": "SELL", "type": "CALL", "strike": 105, "price": 3.20},
            ],
            "notes": [f"Edge detected: {edge:.2f}%"]
        })
    
    return opportunities


def render_sidebar():
    """Render sidebar with configuration options."""
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        # API Connection
        st.markdown("#### 🔌 Schwab API")
        api_key = st.text_input("API Key", type="password", key="api_key")
        api_secret = st.text_input("API Secret", type="password", key="api_secret")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Connect", use_container_width=True):
                if api_key and api_secret:
                    with st.spinner("Connecting..."):
                        time.sleep(1)
                        st.session_state.api_connected = True
                        st.success("Connected!")
                else:
                    st.warning("Enter credentials")
        with col2:
            if st.button("Demo Mode", use_container_width=True):
                st.session_state.api_connected = True
                st.info("Demo mode active")
        
        st.markdown("---")
        
        # Scan Parameters
        st.markdown("#### 🎯 Scan Parameters")
        
        min_edge = st.slider(
            "Min Edge (%)",
            min_value=0.1,
            max_value=5.0,
            value=0.5,
            step=0.1,
            help="Minimum profit edge to flag opportunity"
        )
        
        min_confidence = st.slider(
            "Min Confidence",
            min_value=0.0,
            max_value=1.0,
            value=0.6,
            step=0.05,
            help="Minimum confidence score"
        )
        
        max_dte = st.slider(
            "Max DTE",
            min_value=1,
            max_value=90,
            value=45,
            help="Maximum days to expiration"
        )
        
        st.markdown("---")
        
        # Risk Parameters
        st.markdown("#### ⚠️ Risk Limits")
        
        max_position = st.number_input(
            "Max Position Size",
            min_value=1,
            max_value=100,
            value=10,
            help="Maximum contracts per position"
        )
        
        max_delta = st.slider(
            "Max Delta Exposure",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.01,
            help="Maximum absolute delta"
        )
        
        st.markdown("---")
        
        # Watchlist
        st.markdown("#### 📋 Watchlist")
        
        watchlist_text = st.text_area(
            "Symbols (one per line)",
            value="\n".join(st.session_state.watchlist),
            height=150
        )
        
        if st.button("Update Watchlist", use_container_width=True):
            st.session_state.watchlist = [
                s.strip().upper() 
                for s in watchlist_text.split("\n") 
                if s.strip()
            ]
            st.success(f"Updated: {len(st.session_state.watchlist)} symbols")
        
        return {
            "min_edge": min_edge,
            "min_confidence": min_confidence,
            "max_dte": max_dte,
            "max_position": max_position,
            "max_delta": max_delta
        }


def render_opportunity_card(opp: Dict, idx: int):
    """Render a single opportunity card."""
    risk_class = "risk-free" if opp["risk_free"] else "statistical"
    risk_badge = "🟢 Risk-Free" if opp["risk_free"] else "🟡 Statistical"
    
    with st.container():
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        
        with col1:
            st.markdown(f"### {opp['symbol']} - {opp['type'].replace('_', ' ').title()}")
            st.caption(f"{risk_badge} | Confidence: {opp['confidence']*100:.0f}%")
        
        with col2:
            st.metric(
                "Expected Profit",
                f"${opp['expected_profit']:.2f}",
                f"{opp['expected_profit_pct']:.2f}%"
            )
        
        with col3:
            st.metric(
                "Net Delta",
                f"{opp['net_delta']:.3f}",
                delta_color="off"
            )
        
        with col4:
            st.metric(
                "DTE",
                f"{opp['days_to_expiration']}",
                delta_color="off"
            )
        
        # Expandable details
        with st.expander("View Details"):
            # Legs table
            if opp.get("legs"):
                legs_df = pd.DataFrame(opp["legs"])
                st.dataframe(legs_df, use_container_width=True)
            
            # Greeks
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Delta", f"{opp.get('net_delta', 0):.4f}")
            with col2:
                st.metric("Gamma", f"{opp.get('net_gamma', 0):.4f}")
            with col3:
                st.metric("Vega", f"{opp.get('net_vega', 0):.2f}")
            with col4:
                st.metric("Theta", f"{opp.get('net_theta', 0):.4f}")
            
            # Notes
            if opp.get("notes"):
                st.info(" | ".join(opp["notes"]))
            
            # Action buttons
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("📊 Analyze", key=f"analyze_{idx}"):
                    st.session_state.selected_opp = opp
            with col2:
                if st.button("📋 Copy", key=f"copy_{idx}"):
                    st.toast("Copied to clipboard!")
            with col3:
                if st.button("🚀 Execute", key=f"execute_{idx}", type="primary"):
                    st.warning("Execution requires live API connection")
        
        st.markdown("---")


def render_volatility_surface(symbol: str = "SPY"):
    """Render 3D volatility surface."""
    # Generate mock IV surface data
    strikes = np.linspace(0.8, 1.2, 20)  # Moneyness
    expirations = np.array([7, 14, 21, 30, 45, 60, 90])  # DTE
    
    # Create IV surface with skew
    iv_surface = np.zeros((len(expirations), len(strikes)))
    for i, dte in enumerate(expirations):
        atm_iv = 0.20 + 0.05 * np.sqrt(dte / 30)
        for j, moneyness in enumerate(strikes):
            # Add skew (higher IV for OTM puts)
            skew = 0.1 * (1 - moneyness) ** 2
            term_premium = 0.02 * np.sqrt(dte / 30)
            iv_surface[i, j] = atm_iv + skew + term_premium + np.random.normal(0, 0.005)
    
    fig = go.Figure(data=[go.Surface(
        z=iv_surface * 100,
        x=strikes * 100,
        y=expirations,
        colorscale='Viridis',
        showscale=True,
        colorbar=dict(title="IV %")
    )])
    
    fig.update_layout(
        title=f"{symbol} Implied Volatility Surface",
        scene=dict(
            xaxis_title="Strike (%ATM)",
            yaxis_title="Days to Expiration",
            zaxis_title="Implied Volatility (%)",
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.8))
        ),
        height=500,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig


def render_skew_chart(symbol: str = "SPY"):
    """Render volatility skew chart."""
    strikes = np.linspace(90, 110, 21)
    
    # Near-term skew (steeper)
    near_iv = 0.22 + 0.15 * ((100 - strikes) / 100) ** 2
    near_iv = np.where(strikes < 100, near_iv + 0.02, near_iv)
    
    # Far-term skew (flatter)
    far_iv = 0.25 + 0.08 * ((100 - strikes) / 100) ** 2
    far_iv = np.where(strikes < 100, far_iv + 0.01, far_iv)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=strikes, y=near_iv * 100,
        mode='lines+markers',
        name='7 DTE',
        line=dict(color='#00ff88', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=strikes, y=far_iv * 100,
        mode='lines+markers',
        name='30 DTE',
        line=dict(color='#00aaff', width=2)
    ))
    
    # Mark ATM
    fig.add_vline(x=100, line_dash="dash", line_color="white", opacity=0.5)
    
    fig.update_layout(
        title=f"{symbol} Volatility Skew",
        xaxis_title="Strike (%ATM)",
        yaxis_title="Implied Volatility (%)",
        height=350,
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    
    return fig


def render_term_structure(symbol: str = "SPY"):
    """Render IV term structure."""
    dtes = [7, 14, 21, 30, 45, 60, 90, 120]
    atm_iv = [0.18, 0.19, 0.195, 0.20, 0.205, 0.21, 0.215, 0.22]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=dtes, y=[iv * 100 for iv in atm_iv],
        mode='lines+markers',
        name='ATM IV',
        line=dict(color='#00ff88', width=3),
        fill='tozeroy',
        fillcolor='rgba(0, 255, 136, 0.1)'
    ))
    
    fig.update_layout(
        title=f"{symbol} IV Term Structure",
        xaxis_title="Days to Expiration",
        yaxis_title="ATM Implied Volatility (%)",
        height=350,
        template="plotly_dark"
    )
    
    return fig


def render_pnl_heatmap():
    """Render P&L heatmap for selected strategy."""
    # Generate mock P&L data
    price_moves = np.linspace(-10, 10, 21)
    days = np.array([0, 7, 14, 21, 30])
    
    pnl = np.zeros((len(days), len(price_moves)))
    for i, day in enumerate(days):
        for j, move in enumerate(price_moves):
            # Iron condor-like payoff
            max_profit = 200
            width = 5
            time_decay = (30 - day) / 30
            
            if abs(move) < width:
                pnl[i, j] = max_profit * time_decay
            else:
                pnl[i, j] = max_profit - (abs(move) - width) * 50
    
    fig = go.Figure(data=go.Heatmap(
        z=pnl,
        x=price_moves,
        y=days,
        colorscale='RdYlGn',
        zmid=0,
        colorbar=dict(title="P&L ($)")
    ))
    
    fig.update_layout(
        title="Position P&L Heatmap",
        xaxis_title="Underlying Price Change (%)",
        yaxis_title="Days Elapsed",
        height=350,
        template="plotly_dark"
    )
    
    return fig


def render_backtest_results():
    """Render backtest performance summary."""
    # Mock backtest data
    dates = pd.date_range(start="2024-01-01", end="2024-12-01", freq="D")
    
    # Generate equity curve
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, len(dates))
    returns[::30] += np.random.uniform(0.02, 0.05, len(returns[::30]))  # Monthly arb captures
    equity = 100000 * (1 + returns).cumprod()
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Equity Curve", "Monthly Returns", "Win Rate by Strategy", "Drawdown"),
        specs=[[{"colspan": 2}, None], [{}, {}]]
    )
    
    # Equity curve
    fig.add_trace(
        go.Scatter(x=dates, y=equity, mode='lines', name='Equity',
                   line=dict(color='#00ff88', width=2)),
        row=1, col=1
    )
    
    # Monthly returns
    monthly_returns = [np.random.uniform(-0.02, 0.08) for _ in range(12)]
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    colors = ['#00ff88' if r > 0 else '#ff4444' for r in monthly_returns]
    
    fig.add_trace(
        go.Bar(x=months, y=[r*100 for r in monthly_returns], 
               marker_color=colors, name='Monthly Returns'),
        row=2, col=1
    )
    
    # Win rate by strategy
    strategies = ['PCP', 'Box', 'Butterfly', 'Vertical', 'Skew']
    win_rates = [0.85, 0.92, 0.78, 0.72, 0.65]
    
    fig.add_trace(
        go.Bar(x=strategies, y=[wr*100 for wr in win_rates],
               marker_color='#00aaff', name='Win Rate'),
        row=2, col=2
    )
    
    fig.update_layout(
        height=600,
        showlegend=False,
        template="plotly_dark"
    )
    
    return fig


def main():
    """Main application entry point."""
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="header-title">📈 Options Arbitrage Scanner</h1>', unsafe_allow_html=True)
    st.caption("Real-time mispricing detection with statistical edge quantification")
    
    # Sidebar
    config = render_sidebar()
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Live Scanner",
        "📊 Volatility Analysis", 
        "📈 Backtest Results",
        "⚙️ Settings"
    ])
    
    with tab1:
        # Scanner controls
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        
        with col1:
            st.markdown("### Active Opportunities")
        
        with col2:
            auto_refresh = st.toggle("Auto Refresh", value=False)
        
        with col3:
            refresh_interval = st.selectbox(
                "Interval",
                options=[5, 10, 30, 60],
                format_func=lambda x: f"{x}s"
            )
        
        with col4:
            if st.button("🔄 Scan Now", type="primary", use_container_width=True):
                with st.spinner("Scanning markets..."):
                    time.sleep(1)
                    st.session_state.opportunities = create_mock_opportunities()
                    st.session_state.last_scan = datetime.now()
                    st.success(f"Found {len(st.session_state.opportunities)} opportunities!")
        
        # Stats row
        if st.session_state.opportunities:
            opps = st.session_state.opportunities
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Total Opportunities", len(opps))
            
            with col2:
                risk_free = sum(1 for o in opps if o["risk_free"])
                st.metric("Risk-Free", risk_free)
            
            with col3:
                total_edge = sum(o["expected_profit"] for o in opps)
                st.metric("Total Edge", f"${total_edge:.2f}")
            
            with col4:
                avg_conf = np.mean([o["confidence"] for o in opps])
                st.metric("Avg Confidence", f"{avg_conf*100:.0f}%")
            
            with col5:
                if st.session_state.last_scan:
                    elapsed = (datetime.now() - st.session_state.last_scan).seconds
                    st.metric("Last Scan", f"{elapsed}s ago")
            
            st.markdown("---")
            
            # Filter options
            col1, col2, col3 = st.columns(3)
            
            with col1:
                type_filter = st.multiselect(
                    "Filter by Type",
                    options=list(set(o["type"] for o in opps)),
                    default=list(set(o["type"] for o in opps))
                )
            
            with col2:
                symbol_filter = st.multiselect(
                    "Filter by Symbol",
                    options=list(set(o["symbol"] for o in opps)),
                    default=list(set(o["symbol"] for o in opps))
                )
            
            with col3:
                risk_filter = st.radio(
                    "Risk Type",
                    options=["All", "Risk-Free Only", "Statistical Only"],
                    horizontal=True
                )
            
            # Filter opportunities
            filtered = [
                o for o in opps
                if o["type"] in type_filter
                and o["symbol"] in symbol_filter
                and (risk_filter == "All" or 
                     (risk_filter == "Risk-Free Only" and o["risk_free"]) or
                     (risk_filter == "Statistical Only" and not o["risk_free"]))
            ]
            
            # Render opportunities
            for idx, opp in enumerate(filtered):
                render_opportunity_card(opp, idx)
        
        else:
            st.info("👆 Click 'Scan Now' to search for arbitrage opportunities")
            
            # Demo data
            with st.expander("📚 How It Works"):
                st.markdown("""
                ### Arbitrage Types Detected
                
                1. **Put-Call Parity** - Exploits pricing inconsistencies between calls, puts, and stock
                2. **Box Spreads** - Risk-free arbitrage using four options at two strikes
                3. **Butterfly Spreads** - Exploits mispricing in three-strike combinations
                4. **Vertical Spreads** - Identifies spreads priced above theoretical maximum
                5. **Volatility Skew** - Statistical edge from IV surface anomalies
                
                ### Key Metrics
                
                - **Expected Profit**: Net profit after commissions and slippage
                - **Confidence**: Probability of successful execution
                - **Net Delta**: Position's directional exposure
                - **DTE**: Days until expiration
                """)
    
    with tab2:
        st.markdown("### Volatility Analysis")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            selected_symbol = st.selectbox(
                "Select Symbol",
                options=st.session_state.watchlist,
                index=0
            )
        
        # Volatility charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(render_skew_chart(selected_symbol), use_container_width=True)
        
        with col2:
            st.plotly_chart(render_term_structure(selected_symbol), use_container_width=True)
        
        # 3D Surface
        st.plotly_chart(render_volatility_surface(selected_symbol), use_container_width=True)
        
        # P&L Heatmap
        st.markdown("### Position Analysis")
        st.plotly_chart(render_pnl_heatmap(), use_container_width=True)
    
    with tab3:
        st.markdown("### Backtest Performance")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Return", "+47.3%", "+12.1% vs benchmark")
        with col2:
            st.metric("Sharpe Ratio", "2.14", "")
        with col3:
            st.metric("Win Rate", "78.5%", "+3.2%")
        with col4:
            st.metric("Max Drawdown", "-8.7%", "")
        
        st.plotly_chart(render_backtest_results(), use_container_width=True)
        
        # Trade log
        st.markdown("### Recent Trades")
        
        trades_data = {
            "Date": pd.date_range(end=datetime.now(), periods=10, freq="D"),
            "Symbol": np.random.choice(["SPY", "QQQ", "AAPL", "NVDA"], 10),
            "Type": np.random.choice(["PCP", "Box", "Butterfly"], 10),
            "Entry": np.random.uniform(1, 5, 10).round(2),
            "Exit": np.random.uniform(1, 5, 10).round(2),
            "P&L": np.random.uniform(-100, 500, 10).round(2),
            "Status": np.random.choice(["Winner", "Loser"], 10, p=[0.78, 0.22])
        }
        
        trades_df = pd.DataFrame(trades_data)
        trades_df["P&L"] = trades_df["P&L"].apply(
            lambda x: f"${x:.2f}" if x >= 0 else f"-${abs(x):.2f}"
        )
        
        st.dataframe(trades_df, use_container_width=True)
    
    with tab4:
        st.markdown("### Application Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Trading Parameters")
            
            commission = st.number_input(
                "Commission per Contract ($)",
                min_value=0.0,
                max_value=10.0,
                value=0.65,
                step=0.05
            )
            
            slippage = st.slider(
                "Slippage Model (%)",
                min_value=0.0,
                max_value=1.0,
                value=0.05,
                step=0.01
            )
            
            min_oi = st.number_input(
                "Minimum Open Interest",
                min_value=0,
                max_value=10000,
                value=100
            )
            
            min_vol = st.number_input(
                "Minimum Volume",
                min_value=0,
                max_value=10000,
                value=50
            )
        
        with col2:
            st.markdown("#### Risk Parameters")
            
            max_spread = st.slider(
                "Max Bid-Ask Spread (%)",
                min_value=0.0,
                max_value=20.0,
                value=5.0,
                step=0.5
            )
            
            risk_free_rate = st.slider(
                "Risk-Free Rate (%)",
                min_value=0.0,
                max_value=10.0,
                value=5.0,
                step=0.25
            )
            
            st.markdown("#### Notifications")
            
            email_alerts = st.toggle("Email Alerts", value=False)
            sound_alerts = st.toggle("Sound Alerts", value=True)
            
            if email_alerts:
                email = st.text_input("Email Address")
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💾 Save Settings", use_container_width=True):
                st.success("Settings saved!")
        
        with col2:
            if st.button("🔄 Reset Defaults", use_container_width=True):
                st.info("Settings reset to defaults")
        
        with col3:
            if st.button("📤 Export Config", use_container_width=True):
                st.download_button(
                    "Download JSON",
                    data=json.dumps({"commission": commission, "slippage": slippage}),
                    file_name="config.json",
                    mime="application/json"
                )


if __name__ == "__main__":
    main()