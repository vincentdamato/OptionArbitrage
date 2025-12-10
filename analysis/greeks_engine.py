"""
Greeks Calculation Engine.
Implements Black-Scholes and advanced Greeks calculations.
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq, newton
from typing import Dict, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import math
from loguru import logger


class OptionType(Enum):
    CALL = "call"
    PUT = "put"


@dataclass
class Greeks:
    """Container for option Greeks."""
    delta: float
    gamma: float
    theta: float  # Daily theta (negative for long positions)
    vega: float   # Per 1% IV move
    rho: float    # Per 1% rate move
    
    # Second-order Greeks
    vanna: Optional[float] = None      # d(delta)/d(vol)
    charm: Optional[float] = None      # d(delta)/d(time)
    vomma: Optional[float] = None      # d(vega)/d(vol)
    speed: Optional[float] = None      # d(gamma)/d(spot)
    color: Optional[float] = None      # d(gamma)/d(time)
    
    # Third-order Greeks
    ultima: Optional[float] = None     # d(vomma)/d(vol)
    
    def to_dict(self) -> Dict[str, float]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class OptionPrice:
    """Container for option pricing results."""
    price: float
    intrinsic: float
    extrinsic: float
    greeks: Greeks
    
    @property
    def time_value(self) -> float:
        return self.extrinsic


class GreeksCalculator:
    """
    Black-Scholes based Greeks calculator with extensions.
    
    Supports:
    - Standard first-order Greeks (delta, gamma, theta, vega, rho)
    - Second-order Greeks (vanna, charm, vomma, speed, color)
    - Implied volatility calculation
    - American option approximations
    """
    
    def __init__(self, dividend_yield: float = 0.0):
        self.dividend_yield = dividend_yield
    
    def _d1(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate d1 parameter."""
        if T <= 0 or sigma <= 0:
            return 0.0
        q = self.dividend_yield
        return (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    
    def _d2(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate d2 parameter."""
        if T <= 0 or sigma <= 0:
            return 0.0
        return self._d1(S, K, T, r, sigma) - sigma * np.sqrt(T)
    
    def black_scholes_price(
        self,
        S: float,       # Spot price
        K: float,       # Strike price
        T: float,       # Time to expiration (years)
        r: float,       # Risk-free rate
        sigma: float,   # Volatility
        option_type: Union[str, OptionType] = "call"
    ) -> float:
        """
        Calculate Black-Scholes option price.
        
        Args:
            S: Current underlying price
            K: Strike price
            T: Time to expiration in years
            r: Risk-free interest rate (annualized)
            sigma: Implied volatility (annualized)
            option_type: 'call' or 'put'
        
        Returns:
            Theoretical option price
        """
        if isinstance(option_type, str):
            option_type = OptionType(option_type.lower())
        
        if T <= 0:
            # At expiration
            if option_type == OptionType.CALL:
                return max(0, S - K)
            return max(0, K - S)
        
        q = self.dividend_yield
        d1 = self._d1(S, K, T, r, sigma)
        d2 = self._d2(S, K, T, r, sigma)
        
        if option_type == OptionType.CALL:
            price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
        
        return max(0, price)
    
    def calculate_delta(
        self, S: float, K: float, T: float, r: float, sigma: float,
        option_type: Union[str, OptionType] = "call"
    ) -> float:
        """Calculate option delta."""
        if isinstance(option_type, str):
            option_type = OptionType(option_type.lower())
        
        if T <= 0:
            if option_type == OptionType.CALL:
                return 1.0 if S > K else 0.0
            return -1.0 if S < K else 0.0
        
        q = self.dividend_yield
        d1 = self._d1(S, K, T, r, sigma)
        
        if option_type == OptionType.CALL:
            return np.exp(-q * T) * norm.cdf(d1)
        return np.exp(-q * T) * (norm.cdf(d1) - 1)
    
    def calculate_gamma(
        self, S: float, K: float, T: float, r: float, sigma: float
    ) -> float:
        """Calculate option gamma (same for calls and puts)."""
        if T <= 0 or sigma <= 0 or S <= 0:
            return 0.0
        
        q = self.dividend_yield
        d1 = self._d1(S, K, T, r, sigma)
        return np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    def calculate_theta(
        self, S: float, K: float, T: float, r: float, sigma: float,
        option_type: Union[str, OptionType] = "call"
    ) -> float:
        """Calculate option theta (daily decay)."""
        if isinstance(option_type, str):
            option_type = OptionType(option_type.lower())
        
        if T <= 0:
            return 0.0
        
        q = self.dividend_yield
        d1 = self._d1(S, K, T, r, sigma)
        d2 = self._d2(S, K, T, r, sigma)
        
        # First term: time decay of gamma
        term1 = -S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
        
        if option_type == OptionType.CALL:
            term2 = q * S * np.exp(-q * T) * norm.cdf(d1)
            term3 = -r * K * np.exp(-r * T) * norm.cdf(d2)
            theta_annual = term1 - term2 + term3
        else:
            term2 = -q * S * np.exp(-q * T) * norm.cdf(-d1)
            term3 = r * K * np.exp(-r * T) * norm.cdf(-d2)
            theta_annual = term1 - term2 + term3
        
        # Convert to daily theta
        return theta_annual / 365
    
    def calculate_vega(
        self, S: float, K: float, T: float, r: float, sigma: float
    ) -> float:
        """Calculate option vega (per 1% vol move)."""
        if T <= 0:
            return 0.0
        
        q = self.dividend_yield
        d1 = self._d1(S, K, T, r, sigma)
        vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)
        
        # Return vega per 1% move (not per 100%)
        return vega / 100
    
    def calculate_rho(
        self, S: float, K: float, T: float, r: float, sigma: float,
        option_type: Union[str, OptionType] = "call"
    ) -> float:
        """Calculate option rho (per 1% rate move)."""
        if isinstance(option_type, str):
            option_type = OptionType(option_type.lower())
        
        if T <= 0:
            return 0.0
        
        d2 = self._d2(S, K, T, r, sigma)
        
        if option_type == OptionType.CALL:
            rho = K * T * np.exp(-r * T) * norm.cdf(d2)
        else:
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
        
        # Return rho per 1% move
        return rho / 100
    
    def calculate_vanna(
        self, S: float, K: float, T: float, r: float, sigma: float
    ) -> float:
        """Calculate vanna (d(delta)/d(vol) or d(vega)/d(spot))."""
        if T <= 0 or sigma <= 0 or S <= 0:
            return 0.0
        
        q = self.dividend_yield
        d1 = self._d1(S, K, T, r, sigma)
        d2 = self._d2(S, K, T, r, sigma)
        
        vanna = -np.exp(-q * T) * norm.pdf(d1) * d2 / sigma
        return vanna / 100  # Per 1% vol move
    
    def calculate_charm(
        self, S: float, K: float, T: float, r: float, sigma: float,
        option_type: Union[str, OptionType] = "call"
    ) -> float:
        """Calculate charm (delta decay / d(delta)/d(time))."""
        if isinstance(option_type, str):
            option_type = OptionType(option_type.lower())
        
        if T <= 0 or sigma <= 0:
            return 0.0
        
        q = self.dividend_yield
        d1 = self._d1(S, K, T, r, sigma)
        d2 = self._d2(S, K, T, r, sigma)
        
        term1 = q * np.exp(-q * T) * norm.cdf(d1 if option_type == OptionType.CALL else -d1)
        term2_num = 2 * (r - q) * T - d2 * sigma * np.sqrt(T)
        term2 = np.exp(-q * T) * norm.pdf(d1) * term2_num / (2 * T * sigma * np.sqrt(T))
        
        if option_type == OptionType.CALL:
            return -(term1 + term2) / 365  # Daily charm
        return (term1 - term2) / 365
    
    def calculate_vomma(
        self, S: float, K: float, T: float, r: float, sigma: float
    ) -> float:
        """Calculate vomma (d(vega)/d(vol))."""
        if T <= 0 or sigma <= 0 or S <= 0:
            return 0.0
        
        q = self.dividend_yield
        d1 = self._d1(S, K, T, r, sigma)
        d2 = self._d2(S, K, T, r, sigma)
        
        vega = self.calculate_vega(S, K, T, r, sigma) * 100  # Get full vega
        vomma = vega * d1 * d2 / sigma
        return vomma / 100  # Per 1% vol move
    
    def calculate_speed(
        self, S: float, K: float, T: float, r: float, sigma: float
    ) -> float:
        """Calculate speed (d(gamma)/d(spot))."""
        if T <= 0 or sigma <= 0 or S <= 0:
            return 0.0
        
        q = self.dividend_yield
        d1 = self._d1(S, K, T, r, sigma)
        gamma = self.calculate_gamma(S, K, T, r, sigma)
        
        return -gamma / S * (d1 / (sigma * np.sqrt(T)) + 1)
    
    def calculate_all_greeks(
        self, S: float, K: float, T: float, r: float, sigma: float,
        option_type: Union[str, OptionType] = "call",
        include_second_order: bool = True
    ) -> Dict[str, float]:
        """Calculate all Greeks for an option."""
        if isinstance(option_type, str):
            option_type = OptionType(option_type.lower())
        
        price = self.black_scholes_price(S, K, T, r, sigma, option_type)
        
        # Calculate intrinsic value
        if option_type == OptionType.CALL:
            intrinsic = max(0, S - K)
        else:
            intrinsic = max(0, K - S)
        
        result = {
            "price": price,
            "intrinsic": intrinsic,
            "extrinsic": price - intrinsic,
            "delta": self.calculate_delta(S, K, T, r, sigma, option_type),
            "gamma": self.calculate_gamma(S, K, T, r, sigma),
            "theta": self.calculate_theta(S, K, T, r, sigma, option_type),
            "vega": self.calculate_vega(S, K, T, r, sigma),
            "rho": self.calculate_rho(S, K, T, r, sigma, option_type),
        }
        
        if include_second_order:
            result.update({
                "vanna": self.calculate_vanna(S, K, T, r, sigma),
                "charm": self.calculate_charm(S, K, T, r, sigma, option_type),
                "vomma": self.calculate_vomma(S, K, T, r, sigma),
                "speed": self.calculate_speed(S, K, T, r, sigma),
            })
        
        return result
    
    def implied_volatility(
        self,
        market_price: float,
        S: float,
        K: float,
        T: float,
        r: float,
        option_type: Union[str, OptionType] = "call",
        precision: float = 1e-6,
        max_iterations: int = 100
    ) -> Optional[float]:
        """
        Calculate implied volatility from market price using Newton-Raphson.
        
        Args:
            market_price: Observed market price
            S, K, T, r: Standard BS parameters
            option_type: 'call' or 'put'
            precision: Convergence threshold
            max_iterations: Maximum iterations
        
        Returns:
            Implied volatility or None if not found
        """
        if isinstance(option_type, str):
            option_type = OptionType(option_type.lower())
        
        # Check for arbitrage / invalid prices
        if option_type == OptionType.CALL:
            intrinsic = max(0, S - K * np.exp(-r * T))
            if market_price < intrinsic:
                return None
        else:
            intrinsic = max(0, K * np.exp(-r * T) - S)
            if market_price < intrinsic:
                return None
        
        # Initial guess using Brenner-Subrahmanyam approximation
        sigma_init = np.sqrt(2 * np.pi / T) * market_price / S
        sigma_init = max(0.01, min(sigma_init, 5.0))
        
        def objective(sigma):
            return self.black_scholes_price(S, K, T, r, sigma, option_type) - market_price
        
        def vega(sigma):
            if T <= 0:
                return 0.0
            d1 = self._d1(S, K, T, r, sigma)
            return S * np.exp(-self.dividend_yield * T) * norm.pdf(d1) * np.sqrt(T)
        
        # Newton-Raphson iteration
        sigma = sigma_init
        for i in range(max_iterations):
            price_diff = objective(sigma)
            v = vega(sigma)
            
            if abs(price_diff) < precision:
                return sigma
            
            if v < 1e-10:
                # Vega too small, try bisection
                break
            
            sigma_new = sigma - price_diff / v
            sigma_new = max(0.001, min(sigma_new, 10.0))
            
            if abs(sigma_new - sigma) < precision:
                return sigma_new
            
            sigma = sigma_new
        
        # Fall back to Brent's method
        try:
            return brentq(objective, 0.001, 10.0, xtol=precision)
        except ValueError:
            return None
    
    def put_call_parity_price(
        self,
        call_price: Optional[float],
        put_price: Optional[float],
        S: float,
        K: float,
        T: float,
        r: float
    ) -> Dict[str, float]:
        """
        Check put-call parity and calculate synthetic prices.
        
        Put-Call Parity: C - P = S - K * e^(-rT)
        
        Returns:
            Dictionary with synthetic prices and parity violation if any
        """
        q = self.dividend_yield
        forward = S * np.exp(-q * T)
        pv_strike = K * np.exp(-r * T)
        parity_diff = forward - pv_strike
        
        result = {
            "forward": forward,
            "pv_strike": pv_strike,
            "parity_rhs": parity_diff,  # C - P should equal this
        }
        
        if call_price is not None:
            result["synthetic_put"] = call_price - parity_diff
        
        if put_price is not None:
            result["synthetic_call"] = put_price + parity_diff
        
        if call_price is not None and put_price is not None:
            actual_diff = call_price - put_price
            result["parity_violation"] = actual_diff - parity_diff
            result["parity_violation_pct"] = (
                abs(result["parity_violation"]) / max(call_price, put_price) * 100
                if max(call_price, put_price) > 0 else 0
            )
        
        return result


class PositionGreeksCalculator:
    """Calculate aggregate Greeks for multi-leg positions."""
    
    def __init__(self, calculator: Optional[GreeksCalculator] = None):
        self.calculator = calculator or GreeksCalculator()
    
    def calculate_position_greeks(
        self,
        legs: list,  # List of dicts with option params and quantity
        underlying_price: float,
        risk_free_rate: float = 0.05
    ) -> Dict[str, float]:
        """
        Calculate aggregate Greeks for a multi-leg position.
        
        Args:
            legs: List of leg definitions, each containing:
                  {strike, expiration_years, option_type, iv, quantity}
            underlying_price: Current underlying price
            risk_free_rate: Risk-free rate
        
        Returns:
            Aggregate position Greeks
        """
        totals = {
            "delta": 0.0,
            "gamma": 0.0,
            "theta": 0.0,
            "vega": 0.0,
            "rho": 0.0,
            "vanna": 0.0,
            "charm": 0.0,
            "vomma": 0.0,
            "net_premium": 0.0,
        }
        
        for leg in legs:
            qty = leg.get("quantity", 1)
            multiplier = 100  # Standard option contract
            
            greeks = self.calculator.calculate_all_greeks(
                S=underlying_price,
                K=leg["strike"],
                T=leg["expiration_years"],
                r=risk_free_rate,
                sigma=leg["iv"],
                option_type=leg["option_type"],
                include_second_order=True
            )
            
            # Aggregate
            totals["delta"] += greeks["delta"] * qty * multiplier
            totals["gamma"] += greeks["gamma"] * qty * multiplier
            totals["theta"] += greeks["theta"] * qty * multiplier
            totals["vega"] += greeks["vega"] * qty * multiplier
            totals["rho"] += greeks["rho"] * qty * multiplier
            totals["vanna"] += greeks.get("vanna", 0) * qty * multiplier
            totals["charm"] += greeks.get("charm", 0) * qty * multiplier
            totals["vomma"] += greeks.get("vomma", 0) * qty * multiplier
            totals["net_premium"] += greeks["price"] * qty * multiplier
        
        return totals
    
    def calculate_hedge_ratio(
        self,
        position_delta: float,
        position_gamma: float,
        hedge_type: str = "delta"
    ) -> Dict[str, float]:
        """
        Calculate hedge ratios for delta/gamma neutrality.
        
        Args:
            position_delta: Current position delta
            position_gamma: Current position gamma
            hedge_type: 'delta', 'gamma', or 'both'
        
        Returns:
            Hedge quantities needed
        """
        result = {}
        
        if hedge_type in ["delta", "both"]:
            # Shares needed to delta hedge
            result["shares_to_hedge_delta"] = -position_delta
        
        if hedge_type in ["gamma", "both"]:
            # For gamma hedging, you need options
            result["gamma_exposure"] = position_gamma
            result["note"] = "Gamma hedge requires options with known gamma"
        
        return result