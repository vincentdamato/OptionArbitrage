"""
Configuration module for Options Arbitrage Platform.
Handles Schwab API credentials, trading parameters, and risk limits.
"""

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings
from typing import Optional, List
from pathlib import Path
import os


class SchwabAPIConfig(BaseSettings):
    """Schwab API configuration."""
    
    api_key: str = Field(
        default="",
        description="Schwab API Key from developer portal"
    )
    app_secret: str = Field(
        default="",
        description="Schwab App Secret from developer portal"
    )
    callback_url: str = Field(
        default="https://127.0.0.1:8182",
        description="OAuth callback URL"
    )
    token_path: Path = Field(
        default=Path("~/.schwab/token.json").expanduser(),
        description="Path to store OAuth tokens"
    )
    account_hash: Optional[str] = Field(
        default=None,
        description="Account hash for trading operations"
    )
    
    class Config:
        env_prefix = "SCHWAB_"
        env_file = ".env"


class ArbitrageConfig(BaseSettings):
    """Arbitrage detection parameters."""
    
    # Minimum edge requirements (in percentage)
    min_edge_pct: float = Field(
        default=0.5,
        ge=0.1,
        le=5.0,
        description="Minimum profit edge to flag opportunity"
    )
    
    # Transaction costs
    commission_per_contract: float = Field(
        default=0.65,
        description="Commission per options contract"
    )
    slippage_pct: float = Field(
        default=0.05,
        description="Expected slippage as percentage of spread"
    )
    
    # Liquidity filters
    min_open_interest: int = Field(
        default=100,
        description="Minimum open interest for inclusion"
    )
    min_volume: int = Field(
        default=50,
        description="Minimum daily volume"
    )
    max_bid_ask_spread_pct: float = Field(
        default=5.0,
        description="Maximum bid-ask spread as % of mid"
    )
    
    # Greeks thresholds
    max_delta_exposure: float = Field(
        default=0.10,
        description="Max absolute delta for 'neutral' positions"
    )
    max_gamma_exposure: float = Field(
        default=0.05,
        description="Max gamma exposure"
    )
    max_vega_exposure: float = Field(
        default=0.50,
        description="Max vega exposure per $1 IV move"
    )
    
    # Time filters
    min_dte: int = Field(
        default=1,
        description="Minimum days to expiration"
    )
    max_dte: int = Field(
        default=60,
        description="Maximum days to expiration"
    )
    
    class Config:
        env_prefix = "ARB_"


class RiskConfig(BaseSettings):
    """Risk management parameters."""
    
    max_position_size: int = Field(
        default=10,
        description="Maximum contracts per position"
    )
    max_total_exposure: float = Field(
        default=10000.0,
        description="Maximum total capital at risk"
    )
    max_loss_per_trade: float = Field(
        default=500.0,
        description="Maximum loss per trade"
    )
    daily_loss_limit: float = Field(
        default=2000.0,
        description="Maximum daily loss before halting"
    )
    
    # Position limits
    max_positions_per_underlying: int = Field(
        default=3,
        description="Max concurrent positions per ticker"
    )
    max_total_positions: int = Field(
        default=20,
        description="Max total concurrent positions"
    )
    
    class Config:
        env_prefix = "RISK_"


class BacktestConfig(BaseSettings):
    """Backtesting configuration."""
    
    # Data settings
    lookback_days: int = Field(
        default=252,
        description="Historical lookback period"
    )
    
    # Walk-forward optimization
    train_window: int = Field(
        default=126,
        description="Training window in days"
    )
    test_window: int = Field(
        default=21,
        description="Testing window in days"
    )
    step_size: int = Field(
        default=21,
        description="Step size for walk-forward"
    )
    
    # Monte Carlo settings
    mc_simulations: int = Field(
        default=10000,
        description="Number of Monte Carlo simulations"
    )
    confidence_level: float = Field(
        default=0.95,
        description="Confidence level for VaR/CVaR"
    )
    
    class Config:
        env_prefix = "BACKTEST_"


class StatisticalConfig(BaseSettings):
    """Statistical analysis parameters."""
    
    # Z-score thresholds
    zscore_entry_threshold: float = Field(
        default=2.0,
        description="Z-score for entry signal"
    )
    zscore_exit_threshold: float = Field(
        default=0.5,
        description="Z-score for exit signal"
    )
    
    # Moving windows
    iv_lookback: int = Field(
        default=20,
        description="IV calculation lookback"
    )
    hv_lookback: int = Field(
        default=20,
        description="Historical volatility lookback"
    )
    
    # Regime detection
    regime_lookback: int = Field(
        default=60,
        description="Lookback for regime detection"
    )
    
    class Config:
        env_prefix = "STAT_"


class AppConfig(BaseSettings):
    """Main application configuration."""
    
    schwab: SchwabAPIConfig = Field(default_factory=SchwabAPIConfig)
    arbitrage: ArbitrageConfig = Field(default_factory=ArbitrageConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    statistical: StatisticalConfig = Field(default_factory=StatisticalConfig)
    
    # Application settings
    debug: bool = Field(default=False)
    log_level: str = Field(default="INFO")
    data_dir: Path = Field(
        default=Path("data"),
        description="Directory for data storage"
    )
    
    # Watchlist - default high-liquidity options underlyings
    default_watchlist: List[str] = Field(
        default=[
            "SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA", "TSLA", "AMZN",
            "META", "GOOGL", "AMD", "NFLX", "JPM", "BAC", "XLF", "GLD",
            "SLV", "TLT", "VIX", "UVXY"
        ]
    )
    
    class Config:
        env_file = ".env"
        env_nested_delimiter = "__"


def get_config() -> AppConfig:
    """Get application configuration singleton."""
    return AppConfig()


# Export for easy access
config = get_config()