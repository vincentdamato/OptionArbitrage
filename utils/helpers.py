"""
Utility functions for Options Arbitrage Platform.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple, Any, Union
import pytz
import holidays
from loguru import logger


# Market hours (Eastern Time)
MARKET_OPEN = (9, 30)   # 9:30 AM ET
MARKET_CLOSE = (16, 0)  # 4:00 PM ET
EXTENDED_OPEN = (4, 0)  # 4:00 AM ET
EXTENDED_CLOSE = (20, 0)  # 8:00 PM ET


def get_eastern_time() -> datetime:
    """Get current time in Eastern timezone."""
    eastern = pytz.timezone('US/Eastern')
    return datetime.now(eastern)


def is_market_open(include_extended: bool = False) -> bool:
    """
    Check if US stock market is currently open.
    
    Args:
        include_extended: Include extended hours
    
    Returns:
        True if market is open
    """
    now = get_eastern_time()
    
    # Check if weekend
    if now.weekday() >= 5:
        return False
    
    # Check if holiday
    us_holidays = holidays.US(years=now.year)
    if now.date() in us_holidays:
        return False
    
    # Check time
    current_time = (now.hour, now.minute)
    
    if include_extended:
        open_time = EXTENDED_OPEN
        close_time = EXTENDED_CLOSE
    else:
        open_time = MARKET_OPEN
        close_time = MARKET_CLOSE
    
    return open_time <= current_time < close_time


def get_next_market_open() -> datetime:
    """Get datetime of next market open."""
    eastern = pytz.timezone('US/Eastern')
    now = get_eastern_time()
    
    # Start with today's open
    next_open = now.replace(
        hour=MARKET_OPEN[0],
        minute=MARKET_OPEN[1],
        second=0,
        microsecond=0
    )
    
    # If past today's open, move to tomorrow
    if now >= next_open:
        next_open += timedelta(days=1)
    
    # Skip weekends
    while next_open.weekday() >= 5:
        next_open += timedelta(days=1)
    
    # Skip holidays
    us_holidays = holidays.US(years=[next_open.year, next_open.year + 1])
    while next_open.date() in us_holidays:
        next_open += timedelta(days=1)
        while next_open.weekday() >= 5:
            next_open += timedelta(days=1)
    
    return next_open


def calculate_days_to_expiration(expiration: Union[datetime, date, str]) -> int:
    """
    Calculate trading days to expiration.
    
    Args:
        expiration: Expiration date
    
    Returns:
        Number of trading days
    """
    if isinstance(expiration, str):
        expiration = datetime.strptime(expiration, "%Y-%m-%d").date()
    elif isinstance(expiration, datetime):
        expiration = expiration.date()
    
    today = datetime.now().date()
    
    if expiration <= today:
        return 0
    
    # Count trading days
    trading_days = 0
    current = today + timedelta(days=1)
    us_holidays = holidays.US(years=[today.year, expiration.year])
    
    while current <= expiration:
        if current.weekday() < 5 and current not in us_holidays:
            trading_days += 1
        current += timedelta(days=1)
    
    return trading_days


def annualize_return(
    total_return: float,
    days_held: int,
    trading_days_per_year: int = 252
) -> float:
    """
    Annualize a return.
    
    Args:
        total_return: Total return (e.g., 0.05 for 5%)
        days_held: Number of days position was held
        trading_days_per_year: Trading days in a year
    
    Returns:
        Annualized return
    """
    if days_held <= 0:
        return 0.0
    
    years = days_held / trading_days_per_year
    return (1 + total_return) ** (1 / years) - 1


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.05,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Sharpe ratio.
    
    Args:
        returns: Series of returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year
    
    Returns:
        Sharpe ratio
    """
    if len(returns) < 2:
        return 0.0
    
    excess_returns = returns - risk_free_rate / periods_per_year
    
    if excess_returns.std() == 0:
        return 0.0
    
    return np.sqrt(periods_per_year) * excess_returns.mean() / excess_returns.std()


def calculate_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.05,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Sortino ratio (using downside deviation).
    
    Args:
        returns: Series of returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year
    
    Returns:
        Sortino ratio
    """
    if len(returns) < 2:
        return 0.0
    
    excess_returns = returns - risk_free_rate / periods_per_year
    negative_returns = returns[returns < 0]
    
    if len(negative_returns) < 2:
        return float('inf') if excess_returns.mean() > 0 else 0.0
    
    downside_std = negative_returns.std()
    
    if downside_std == 0:
        return float('inf') if excess_returns.mean() > 0 else 0.0
    
    return np.sqrt(periods_per_year) * excess_returns.mean() / downside_std


def calculate_max_drawdown(equity_curve: pd.Series) -> Tuple[float, int, int]:
    """
    Calculate maximum drawdown and its duration.
    
    Args:
        equity_curve: Series of equity values
    
    Returns:
        (max_drawdown, start_index, end_index)
    """
    if len(equity_curve) < 2:
        return 0.0, 0, 0
    
    # Calculate running maximum
    running_max = equity_curve.expanding().max()
    
    # Calculate drawdown series
    drawdown = (equity_curve - running_max) / running_max
    
    # Find maximum drawdown
    max_dd = drawdown.min()
    end_idx = drawdown.idxmin()
    
    # Find start of drawdown
    start_idx = equity_curve[:end_idx].idxmax()
    
    return abs(max_dd), start_idx, end_idx


def format_currency(value: float, decimals: int = 2) -> str:
    """Format value as currency string."""
    if value >= 0:
        return f"${value:,.{decimals}f}"
    return f"-${abs(value):,.{decimals}f}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format value as percentage string."""
    return f"{value * 100:.{decimals}f}%"


def format_number(value: float, decimals: int = 2) -> str:
    """Format number with thousand separators."""
    return f"{value:,.{decimals}f}"


def parse_option_symbol(symbol: str) -> Dict[str, Any]:
    """
    Parse OCC option symbol.
    
    Format: AAPL  240119C00150000
           |     |     ||
           |     |     |+-- Strike price ($150.00)
           |     |     +--- C=Call, P=Put
           |     +--------- Expiration (YYMMDD)
           +--------------- Underlying (padded to 6 chars)
    
    Args:
        symbol: OCC option symbol
    
    Returns:
        Parsed components
    """
    try:
        # Remove spaces
        symbol = symbol.replace(" ", "")
        
        # Extract components
        underlying = symbol[:6].strip()
        exp_str = symbol[6:12]
        option_type = symbol[12]
        strike_str = symbol[13:]
        
        # Parse expiration
        exp_year = 2000 + int(exp_str[:2])
        exp_month = int(exp_str[2:4])
        exp_day = int(exp_str[4:6])
        expiration = datetime(exp_year, exp_month, exp_day)
        
        # Parse strike (divide by 1000)
        strike = int(strike_str) / 1000
        
        return {
            "underlying": underlying,
            "expiration": expiration,
            "type": "call" if option_type == "C" else "put",
            "strike": strike,
            "raw_symbol": symbol
        }
    except Exception as e:
        logger.error(f"Error parsing option symbol {symbol}: {e}")
        return {}


def build_option_symbol(
    underlying: str,
    expiration: datetime,
    option_type: str,
    strike: float
) -> str:
    """
    Build OCC option symbol.
    
    Args:
        underlying: Underlying symbol
        expiration: Expiration date
        option_type: 'call' or 'put'
        strike: Strike price
    
    Returns:
        OCC option symbol
    """
    # Pad underlying to 6 characters
    underlying_padded = underlying.ljust(6)
    
    # Format expiration
    exp_str = expiration.strftime("%y%m%d")
    
    # Option type
    type_char = "C" if option_type.lower() == "call" else "P"
    
    # Strike (multiply by 1000 and pad to 8 digits)
    strike_int = int(strike * 1000)
    strike_str = str(strike_int).zfill(8)
    
    return f"{underlying_padded}{exp_str}{type_char}{strike_str}"


def round_to_tick(price: float, tick_size: float = 0.01) -> float:
    """
    Round price to nearest tick size.
    
    Args:
        price: Price to round
        tick_size: Minimum tick size
    
    Returns:
        Rounded price
    """
    return round(price / tick_size) * tick_size


def calculate_kelly_criterion(
    win_rate: float,
    avg_win: float,
    avg_loss: float
) -> float:
    """
    Calculate Kelly criterion for position sizing.
    
    Args:
        win_rate: Probability of winning (0-1)
        avg_win: Average winning trade
        avg_loss: Average losing trade (positive number)
    
    Returns:
        Kelly fraction (fraction of capital to risk)
    """
    if avg_loss <= 0 or win_rate <= 0:
        return 0.0
    
    win_loss_ratio = avg_win / avg_loss
    kelly = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
    
    return max(0, min(kelly, 1))


def calculate_position_size(
    capital: float,
    risk_per_trade: float,
    max_loss_per_contract: float,
    kelly_fraction: Optional[float] = None
) -> int:
    """
    Calculate optimal position size.
    
    Args:
        capital: Total capital
        risk_per_trade: Maximum risk per trade (fraction)
        max_loss_per_contract: Maximum loss per contract
        kelly_fraction: Optional Kelly criterion fraction
    
    Returns:
        Number of contracts
    """
    if max_loss_per_contract <= 0:
        return 0
    
    # Calculate based on risk
    risk_amount = capital * risk_per_trade
    position_from_risk = int(risk_amount / max_loss_per_contract)
    
    # Apply Kelly if provided
    if kelly_fraction is not None:
        kelly_amount = capital * kelly_fraction
        position_from_kelly = int(kelly_amount / max_loss_per_contract)
        return min(position_from_risk, position_from_kelly)
    
    return position_from_risk


def calculate_breakeven(
    entry_price: float,
    commission_per_contract: float,
    num_contracts: int,
    direction: str = "long"
) -> float:
    """
    Calculate breakeven price including commissions.
    
    Args:
        entry_price: Entry price per contract
        commission_per_contract: Commission per contract
        num_contracts: Number of contracts
        direction: 'long' or 'short'
    
    Returns:
        Breakeven price
    """
    total_commission = commission_per_contract * num_contracts * 2  # Entry and exit
    commission_per_share = total_commission / (num_contracts * 100)
    
    if direction == "long":
        return entry_price + commission_per_share
    else:
        return entry_price - commission_per_share


class RateLimiter:
    """Simple rate limiter for API calls."""
    
    def __init__(self, calls_per_second: float = 2.0):
        self.calls_per_second = calls_per_second
        self.min_interval = 1.0 / calls_per_second
        self.last_call = 0.0
    
    def wait(self):
        """Wait if necessary to respect rate limit."""
        import time
        
        now = time.time()
        elapsed = now - self.last_call
        
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        
        self.last_call = time.time()


class CircuitBreaker:
    """Circuit breaker for error handling."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failures = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = "closed"  # closed, open, half-open
    
    def record_failure(self):
        """Record a failure."""
        self.failures += 1
        self.last_failure_time = datetime.now()
        
        if self.failures >= self.failure_threshold:
            self.state = "open"
            logger.warning(f"Circuit breaker opened after {self.failures} failures")
    
    def record_success(self):
        """Record a success."""
        self.failures = 0
        self.state = "closed"
    
    def can_proceed(self) -> bool:
        """Check if operation can proceed."""
        if self.state == "closed":
            return True
        
        if self.state == "open":
            # Check if recovery timeout has passed
            if self.last_failure_time:
                elapsed = (datetime.now() - self.last_failure_time).seconds
                if elapsed >= self.recovery_timeout:
                    self.state = "half-open"
                    return True
            return False
        
        # half-open state - allow one attempt
        return True