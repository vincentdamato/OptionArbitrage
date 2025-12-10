"""
Statistical Analysis Module for Options Arbitrage.
Implements z-score analysis, regime detection, and statistical edge quantification.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm, skew, kurtosis
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from loguru import logger

from analysis.greeks_engine import GreeksCalculator


@dataclass
class VolatilityMetrics:
    """Container for volatility analysis results."""
    historical_vol: float
    implied_vol: float
    vol_premium: float  # IV - HV
    vol_premium_zscore: float
    vol_percentile: float  # Where current IV sits in historical range
    vol_regime: str  # low, normal, high, extreme
    
    # Term structure
    near_term_iv: Optional[float] = None
    far_term_iv: Optional[float] = None
    term_structure_slope: Optional[float] = None  # Contango/Backwardation
    
    # Skew metrics
    put_call_skew: Optional[float] = None
    skew_zscore: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class StatisticalEdge:
    """Quantified statistical edge for a trade."""
    expected_value: float
    win_probability: float
    expected_win: float
    expected_loss: float
    risk_reward_ratio: float
    kelly_fraction: float
    sharpe_estimate: float
    zscore: float
    confidence_interval: Tuple[float, float]
    
    def to_dict(self) -> Dict:
        return self.__dict__


@dataclass
class RegimeState:
    """Market regime detection results."""
    regime: str  # trending_up, trending_down, ranging, volatile
    confidence: float
    trend_strength: float
    volatility_state: str
    momentum_score: float
    mean_reversion_score: float


class StatisticalAnalyzer:
    """
    Statistical analysis engine for options arbitrage.
    
    Features:
    - Z-score calculation for IV and prices
    - Volatility regime detection
    - Historical volatility calculation
    - IV term structure analysis
    - Put/Call skew analysis
    - Statistical edge quantification
    - Monte Carlo probability estimation
    """
    
    def __init__(
        self,
        iv_lookback: int = 20,
        hv_lookback: int = 20,
        zscore_threshold: float = 2.0,
        regime_lookback: int = 60
    ):
        self.iv_lookback = iv_lookback
        self.hv_lookback = hv_lookback
        self.zscore_threshold = zscore_threshold
        self.regime_lookback = regime_lookback
        self.greeks_calc = GreeksCalculator()
    
    def calculate_historical_volatility(
        self,
        prices: pd.Series,
        window: Optional[int] = None,
        annualize: bool = True
    ) -> pd.Series:
        """
        Calculate historical volatility using close-to-close method.
        
        Args:
            prices: Series of prices
            window: Rolling window size
            annualize: Whether to annualize (multiply by sqrt(252))
        
        Returns:
            Series of historical volatility values
        """
        window = window or self.hv_lookback
        
        # Calculate log returns
        log_returns = np.log(prices / prices.shift(1))
        
        # Rolling standard deviation
        hv = log_returns.rolling(window=window).std()
        
        if annualize:
            hv = hv * np.sqrt(252)
        
        return hv
    
    def calculate_parkinson_volatility(
        self,
        high: pd.Series,
        low: pd.Series,
        window: Optional[int] = None,
        annualize: bool = True
    ) -> pd.Series:
        """
        Calculate Parkinson volatility using high-low range.
        More efficient estimator than close-to-close.
        """
        window = window or self.hv_lookback
        
        # Parkinson formula
        log_hl = np.log(high / low)
        parkinson = log_hl ** 2 / (4 * np.log(2))
        
        vol = np.sqrt(parkinson.rolling(window=window).mean())
        
        if annualize:
            vol = vol * np.sqrt(252)
        
        return vol
    
    def calculate_garman_klass_volatility(
        self,
        open_: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: Optional[int] = None,
        annualize: bool = True
    ) -> pd.Series:
        """
        Calculate Garman-Klass volatility.
        Most efficient estimator using OHLC data.
        """
        window = window or self.hv_lookback
        
        log_hl = np.log(high / low)
        log_co = np.log(close / open_)
        
        # Garman-Klass formula
        term1 = 0.5 * log_hl ** 2
        term2 = (2 * np.log(2) - 1) * log_co ** 2
        
        gk = np.sqrt((term1 - term2).rolling(window=window).mean())
        
        if annualize:
            gk = gk * np.sqrt(252)
        
        return gk
    
    def calculate_iv_zscore(
        self,
        current_iv: float,
        iv_history: pd.Series,
        lookback: Optional[int] = None
    ) -> Tuple[float, float, float]:
        """
        Calculate z-score of current IV vs historical IV.
        
        Returns:
            (zscore, mean, std)
        """
        lookback = lookback or self.iv_lookback
        
        recent_iv = iv_history.tail(lookback)
        mean_iv = recent_iv.mean()
        std_iv = recent_iv.std()
        
        if std_iv < 0.001:
            return 0.0, mean_iv, std_iv
        
        zscore = (current_iv - mean_iv) / std_iv
        
        return zscore, mean_iv, std_iv
    
    def calculate_iv_percentile(
        self,
        current_iv: float,
        iv_history: pd.Series,
        lookback: int = 252
    ) -> float:
        """
        Calculate percentile rank of current IV.
        
        Returns:
            Percentile (0-100)
        """
        recent_iv = iv_history.tail(lookback)
        percentile = stats.percentileofscore(recent_iv.dropna(), current_iv)
        return percentile
    
    def analyze_volatility(
        self,
        current_iv: float,
        price_history: pd.DataFrame,
        iv_history: Optional[pd.Series] = None
    ) -> VolatilityMetrics:
        """
        Comprehensive volatility analysis.
        
        Args:
            current_iv: Current implied volatility
            price_history: DataFrame with OHLC columns
            iv_history: Historical IV series (optional)
        
        Returns:
            VolatilityMetrics with full analysis
        """
        # Calculate historical volatility
        if "close" in price_history.columns:
            hv = self.calculate_historical_volatility(price_history["close"]).iloc[-1]
        elif isinstance(price_history, pd.Series):
            hv = self.calculate_historical_volatility(price_history).iloc[-1]
        else:
            hv = 0.2  # Default
        
        # Volatility premium
        vol_premium = current_iv - hv
        
        # Z-score and percentile
        if iv_history is not None and len(iv_history) > self.iv_lookback:
            zscore, _, _ = self.calculate_iv_zscore(current_iv, iv_history)
            percentile = self.calculate_iv_percentile(current_iv, iv_history)
        else:
            zscore = 0.0
            percentile = 50.0
        
        # Determine regime
        if percentile > 90:
            regime = "extreme_high"
        elif percentile > 75:
            regime = "high"
        elif percentile < 10:
            regime = "extreme_low"
        elif percentile < 25:
            regime = "low"
        else:
            regime = "normal"
        
        return VolatilityMetrics(
            historical_vol=hv,
            implied_vol=current_iv,
            vol_premium=vol_premium,
            vol_premium_zscore=zscore,
            vol_percentile=percentile,
            vol_regime=regime
        )
    
    def analyze_iv_term_structure(
        self,
        iv_by_expiration: Dict[int, float]  # {days_to_exp: iv}
    ) -> Dict[str, Any]:
        """
        Analyze IV term structure.
        
        Args:
            iv_by_expiration: Dict mapping DTE to IV
        
        Returns:
            Term structure analysis
        """
        if len(iv_by_expiration) < 2:
            return {"structure": "insufficient_data"}
        
        dtes = sorted(iv_by_expiration.keys())
        ivs = [iv_by_expiration[d] for d in dtes]
        
        # Calculate slope
        if len(dtes) >= 2:
            slope, intercept, r_value, p_value, std_err = stats.linregress(dtes, ivs)
        else:
            slope = 0
            r_value = 0
        
        # Determine structure type
        near_iv = iv_by_expiration[dtes[0]]
        far_iv = iv_by_expiration[dtes[-1]]
        
        if near_iv > far_iv * 1.05:
            structure = "backwardation"  # Near > Far (often before events)
        elif far_iv > near_iv * 1.05:
            structure = "contango"  # Far > Near (normal)
        else:
            structure = "flat"
        
        return {
            "structure": structure,
            "slope": slope,
            "r_squared": r_value ** 2,
            "near_iv": near_iv,
            "far_iv": far_iv,
            "iv_differential": far_iv - near_iv,
            "dtes": dtes,
            "ivs": ivs
        }
    
    def analyze_put_call_skew(
        self,
        calls: List[Dict],  # [{strike, iv}]
        puts: List[Dict],
        underlying_price: float
    ) -> Dict[str, Any]:
        """
        Analyze put/call IV skew.
        
        Args:
            calls: List of call options with strike and IV
            puts: List of put options with strike and IV
            underlying_price: Current underlying price
        
        Returns:
            Skew analysis
        """
        if not calls or not puts:
            return {"skew": "insufficient_data"}
        
        # Find ATM options
        atm_call = min(calls, key=lambda x: abs(x["strike"] - underlying_price))
        atm_put = min(puts, key=lambda x: abs(x["strike"] - underlying_price))
        
        # Find OTM options (25 delta equivalent, roughly 5% OTM)
        otm_threshold = underlying_price * 0.05
        
        otm_calls = [c for c in calls if c["strike"] > underlying_price + otm_threshold]
        otm_puts = [p for p in puts if p["strike"] < underlying_price - otm_threshold]
        
        if not otm_calls or not otm_puts:
            return {"skew": "insufficient_otm_options"}
        
        # Average OTM IVs
        avg_otm_call_iv = np.mean([c["iv"] for c in otm_calls[:3]])
        avg_otm_put_iv = np.mean([p["iv"] for p in otm_puts[-3:]])
        atm_iv = (atm_call["iv"] + atm_put["iv"]) / 2
        
        # Skew metrics
        put_skew = avg_otm_put_iv - atm_iv  # Usually positive (put skew)
        call_skew = avg_otm_call_iv - atm_iv  # Usually negative or flat
        total_skew = avg_otm_put_iv - avg_otm_call_iv
        
        return {
            "put_skew": put_skew,
            "call_skew": call_skew,
            "total_skew": total_skew,
            "atm_iv": atm_iv,
            "otm_put_iv": avg_otm_put_iv,
            "otm_call_iv": avg_otm_call_iv,
            "skew_direction": "put_heavy" if total_skew > 0.02 else "call_heavy" if total_skew < -0.02 else "neutral"
        }
    
    def calculate_statistical_edge(
        self,
        entry_price: float,
        expected_exit: float,
        stop_loss: float,
        win_rate_estimate: float,
        volatility: float,
        days_held: int = 1
    ) -> StatisticalEdge:
        """
        Calculate statistical edge metrics for a trade.
        
        Args:
            entry_price: Trade entry price
            expected_exit: Target exit price
            stop_loss: Stop loss price
            win_rate_estimate: Estimated win probability (0-1)
            volatility: Annualized volatility
            days_held: Expected holding period
        
        Returns:
            StatisticalEdge with all metrics
        """
        # Expected win/loss
        expected_win = abs(expected_exit - entry_price)
        expected_loss = abs(entry_price - stop_loss)
        
        # Expected value
        ev = win_rate_estimate * expected_win - (1 - win_rate_estimate) * expected_loss
        
        # Risk/reward
        risk_reward = expected_win / expected_loss if expected_loss > 0 else float('inf')
        
        # Kelly fraction
        if expected_loss > 0:
            kelly = (win_rate_estimate * risk_reward - (1 - win_rate_estimate)) / risk_reward
            kelly = max(0, min(1, kelly))  # Clamp to [0, 1]
        else:
            kelly = 0
        
        # Estimate Sharpe
        daily_vol = volatility / np.sqrt(252)
        expected_daily_return = ev / entry_price
        sharpe = expected_daily_return / daily_vol * np.sqrt(252) if daily_vol > 0 else 0
        
        # Z-score of edge
        edge_std = np.sqrt(
            win_rate_estimate * expected_win**2 + 
            (1 - win_rate_estimate) * expected_loss**2 -
            ev**2
        )
        zscore = ev / edge_std if edge_std > 0 else 0
        
        # Confidence interval (95%)
        ci_width = 1.96 * edge_std
        ci = (ev - ci_width, ev + ci_width)
        
        return StatisticalEdge(
            expected_value=ev,
            win_probability=win_rate_estimate,
            expected_win=expected_win,
            expected_loss=expected_loss,
            risk_reward_ratio=risk_reward,
            kelly_fraction=kelly,
            sharpe_estimate=sharpe,
            zscore=zscore,
            confidence_interval=ci
        )
    
    def detect_regime(
        self,
        prices: pd.Series,
        lookback: Optional[int] = None
    ) -> RegimeState:
        """
        Detect market regime using multiple indicators.
        
        Args:
            prices: Price series
            lookback: Analysis period
        
        Returns:
            RegimeState with regime classification
        """
        lookback = lookback or self.regime_lookback
        recent = prices.tail(lookback)
        
        # Calculate returns
        returns = recent.pct_change().dropna()
        
        # Trend analysis
        ma_short = recent.rolling(10).mean().iloc[-1]
        ma_long = recent.rolling(30).mean().iloc[-1]
        current = recent.iloc[-1]
        
        trend_strength = (current - ma_long) / ma_long if ma_long > 0 else 0
        
        # Momentum
        momentum = returns.tail(5).mean() * 252  # Annualized
        
        # Volatility state
        vol = returns.std() * np.sqrt(252)
        vol_percentile = stats.percentileofscore(
            self.calculate_historical_volatility(prices).dropna(),
            vol
        )
        
        if vol_percentile > 80:
            vol_state = "high"
        elif vol_percentile < 20:
            vol_state = "low"
        else:
            vol_state = "normal"
        
        # Mean reversion indicator (distance from moving average)
        mr_score = (current - ma_long) / (recent.std() * np.sqrt(lookback))
        
        # Classify regime
        if abs(trend_strength) > 0.1:
            if trend_strength > 0:
                regime = "trending_up"
            else:
                regime = "trending_down"
            confidence = min(0.9, abs(trend_strength) * 5)
        elif vol_state == "high":
            regime = "volatile"
            confidence = vol_percentile / 100
        else:
            regime = "ranging"
            confidence = 1 - abs(trend_strength) * 5
        
        return RegimeState(
            regime=regime,
            confidence=confidence,
            trend_strength=trend_strength,
            volatility_state=vol_state,
            momentum_score=momentum,
            mean_reversion_score=mr_score
        )
    
    def monte_carlo_option_price(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = "call",
        num_simulations: int = 10000,
        num_steps: int = 252
    ) -> Dict[str, float]:
        """
        Monte Carlo simulation for option pricing.
        
        Args:
            S: Spot price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free rate
            sigma: Volatility
            option_type: 'call' or 'put'
            num_simulations: Number of MC paths
            num_steps: Time steps per path
        
        Returns:
            Pricing results with confidence intervals
        """
        dt = T / num_steps
        
        # Generate paths
        np.random.seed(42)  # For reproducibility
        Z = np.random.standard_normal((num_simulations, num_steps))
        
        # GBM simulation
        drift = (r - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt) * Z
        
        log_returns = drift + diffusion
        log_prices = np.log(S) + np.cumsum(log_returns, axis=1)
        final_prices = np.exp(log_prices[:, -1])
        
        # Calculate payoffs
        if option_type.lower() == "call":
            payoffs = np.maximum(final_prices - K, 0)
        else:
            payoffs = np.maximum(K - final_prices, 0)
        
        # Discount payoffs
        discounted = payoffs * np.exp(-r * T)
        
        # Statistics
        price = np.mean(discounted)
        std_error = np.std(discounted) / np.sqrt(num_simulations)
        ci_95 = (price - 1.96 * std_error, price + 1.96 * std_error)
        
        # Additional statistics
        prob_itm = np.mean(payoffs > 0)
        expected_payoff_if_itm = np.mean(payoffs[payoffs > 0]) if prob_itm > 0 else 0
        
        return {
            "price": price,
            "std_error": std_error,
            "ci_lower": ci_95[0],
            "ci_upper": ci_95[1],
            "prob_itm": prob_itm,
            "expected_payoff_if_itm": expected_payoff_if_itm,
            "num_simulations": num_simulations
        }
    
    def calculate_var_cvar(
        self,
        returns: pd.Series,
        confidence_level: float = 0.95,
        position_value: float = 10000
    ) -> Dict[str, float]:
        """
        Calculate Value at Risk and Conditional VaR.
        
        Args:
            returns: Series of returns
            confidence_level: Confidence level (e.g., 0.95)
            position_value: Position value in dollars
        
        Returns:
            VaR and CVaR metrics
        """
        # Historical VaR
        var_pct = np.percentile(returns, (1 - confidence_level) * 100)
        var_dollar = abs(var_pct * position_value)
        
        # Conditional VaR (Expected Shortfall)
        cvar_returns = returns[returns <= var_pct]
        cvar_pct = cvar_returns.mean() if len(cvar_returns) > 0 else var_pct
        cvar_dollar = abs(cvar_pct * position_value)
        
        # Parametric VaR (assuming normal distribution)
        mu = returns.mean()
        sigma = returns.std()
        parametric_var = norm.ppf(1 - confidence_level, mu, sigma)
        parametric_var_dollar = abs(parametric_var * position_value)
        
        return {
            "var_pct": var_pct,
            "var_dollar": var_dollar,
            "cvar_pct": cvar_pct,
            "cvar_dollar": cvar_dollar,
            "parametric_var_pct": parametric_var,
            "parametric_var_dollar": parametric_var_dollar,
            "confidence_level": confidence_level,
            "returns_skewness": skew(returns),
            "returns_kurtosis": kurtosis(returns)
        }