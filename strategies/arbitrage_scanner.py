"""
Options Arbitrage Scanner.
Detects mispricing opportunities across various arbitrage strategies.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger

from api.schwab_client import OptionsChain, OptionQuote, ContractType
from analysis.greeks_engine import GreeksCalculator


class ArbitrageType(Enum):
    PUT_CALL_PARITY = "put_call_parity"
    BOX_SPREAD = "box_spread"
    CONVERSION = "conversion"
    REVERSAL = "reversal"
    BUTTERFLY = "butterfly"
    CALENDAR_SPREAD = "calendar_spread"
    VERTICAL_MISPRICING = "vertical_mispricing"
    DIVIDEND_ARBITRAGE = "dividend_arbitrage"
    VOLATILITY_SKEW = "volatility_skew"


@dataclass
class ArbitrageOpportunity:
    """Represents a detected arbitrage opportunity."""
    type: ArbitrageType
    symbol: str
    expected_profit: float
    expected_profit_pct: float
    max_loss: float
    risk_free: bool
    confidence: float  # 0-1 confidence score
    
    # Leg details
    legs: List[Dict[str, Any]] = field(default_factory=list)
    
    # Costs
    gross_edge: float = 0.0
    commission_cost: float = 0.0
    slippage_estimate: float = 0.0
    net_edge: float = 0.0
    
    # Greeks
    net_delta: float = 0.0
    net_gamma: float = 0.0
    net_vega: float = 0.0
    net_theta: float = 0.0
    
    # Market data
    underlying_price: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    expiration: Optional[datetime] = None
    days_to_expiration: int = 0
    
    # Liquidity metrics
    min_volume: int = 0
    min_open_interest: int = 0
    max_spread_pct: float = 0.0
    
    # Execution hints
    execution_priority: str = "normal"  # urgent, normal, low
    notes: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "type": self.type.value,
            "symbol": self.symbol,
            "expected_profit": self.expected_profit,
            "expected_profit_pct": self.expected_profit_pct,
            "max_loss": self.max_loss,
            "risk_free": self.risk_free,
            "confidence": self.confidence,
            "legs": self.legs,
            "net_edge": self.net_edge,
            "net_delta": self.net_delta,
            "underlying_price": self.underlying_price,
            "days_to_expiration": self.days_to_expiration,
            "timestamp": self.timestamp.isoformat(),
        }


class ArbitrageScanner:
    """
    Scans options chains for arbitrage opportunities.
    
    Strategies implemented:
    1. Put-Call Parity violations
    2. Box Spread arbitrage
    3. Conversion/Reversal arbitrage
    4. Butterfly mispricing
    5. Calendar spread anomalies
    6. Vertical spread mispricing
    7. Volatility skew arbitrage
    """
    
    def __init__(
        self,
        min_edge_pct: float = 0.5,
        commission_per_contract: float = 0.65,
        slippage_pct: float = 0.05,
        min_open_interest: int = 100,
        min_volume: int = 50,
        max_bid_ask_spread_pct: float = 5.0,
        risk_free_rate: float = 0.05
    ):
        self.min_edge_pct = min_edge_pct
        self.commission_per_contract = commission_per_contract
        self.slippage_pct = slippage_pct
        self.min_open_interest = min_open_interest
        self.min_volume = min_volume
        self.max_bid_ask_spread_pct = max_bid_ask_spread_pct
        self.risk_free_rate = risk_free_rate
        self.greeks_calc = GreeksCalculator()
    
    def scan_chain(self, chain: OptionsChain) -> List[ArbitrageOpportunity]:
        """
        Scan entire options chain for all types of arbitrage.
        
        Args:
            chain: Complete options chain with calls and puts
        
        Returns:
            List of detected arbitrage opportunities
        """
        opportunities = []
        
        # 1. Put-Call Parity
        pcp_opps = self.scan_put_call_parity(chain)
        opportunities.extend(pcp_opps)
        
        # 2. Box Spreads
        box_opps = self.scan_box_spreads(chain)
        opportunities.extend(box_opps)
        
        # 3. Conversions and Reversals
        conv_opps = self.scan_conversions_reversals(chain)
        opportunities.extend(conv_opps)
        
        # 4. Butterfly mispricing
        butterfly_opps = self.scan_butterfly_mispricing(chain)
        opportunities.extend(butterfly_opps)
        
        # 5. Vertical spread mispricing
        vertical_opps = self.scan_vertical_mispricing(chain)
        opportunities.extend(vertical_opps)
        
        # 6. Calendar spread anomalies
        calendar_opps = self.scan_calendar_anomalies(chain)
        opportunities.extend(calendar_opps)
        
        # 7. Volatility skew
        skew_opps = self.scan_volatility_skew(chain)
        opportunities.extend(skew_opps)
        
        # Sort by expected profit
        opportunities.sort(key=lambda x: x.expected_profit, reverse=True)
        
        logger.info(f"Found {len(opportunities)} arbitrage opportunities for {chain.underlying.symbol}")
        
        return opportunities
    
    def scan_put_call_parity(self, chain: OptionsChain) -> List[ArbitrageOpportunity]:
        """
        Scan for put-call parity violations.
        
        Put-Call Parity: C - P = S * e^(-qT) - K * e^(-rT)
        
        Violation creates arbitrage:
        - If C - P > RHS: Sell call, buy put, buy stock (conversion)
        - If C - P < RHS: Buy call, sell put, sell stock (reversal)
        """
        opportunities = []
        S = chain.underlying.mark
        r = self.risk_free_rate
        
        for exp_key, calls in chain.calls.items():
            puts = chain.puts.get(exp_key, [])
            if not puts:
                continue
            
            put_by_strike = {p.strike: p for p in puts}
            
            for call in calls:
                put = put_by_strike.get(call.strike)
                if not put:
                    continue
                
                # Check liquidity
                if not self._check_liquidity(call) or not self._check_liquidity(put):
                    continue
                
                K = call.strike
                T = call.days_to_expiration / 365
                
                if T <= 0:
                    continue
                
                # Calculate theoretical difference
                pv_strike = K * np.exp(-r * T)
                theoretical_diff = S - pv_strike
                
                # Market difference using mid prices
                market_diff = call.mid_price - put.mid_price
                
                # Violation
                violation = market_diff - theoretical_diff
                violation_pct = abs(violation) / S * 100
                
                if violation_pct < self.min_edge_pct:
                    continue
                
                # Calculate after costs
                num_contracts = 4  # 2 legs x 2 (buy/sell)
                commission = num_contracts * self.commission_per_contract
                slippage = (call.spread + put.spread) * 0.5 * self.slippage_pct * 100
                net_edge = abs(violation) * 100 - commission - slippage
                
                if net_edge <= 0:
                    continue
                
                # Determine direction
                if violation > 0:
                    # Call overpriced: Sell call, buy put, buy stock
                    arb_type = ArbitrageType.CONVERSION
                    legs = [
                        {"action": "SELL", "type": "CALL", "strike": K, "price": call.bid},
                        {"action": "BUY", "type": "PUT", "strike": K, "price": put.ask},
                        {"action": "BUY", "type": "STOCK", "shares": 100, "price": S},
                    ]
                else:
                    # Put overpriced: Buy call, sell put, sell stock
                    arb_type = ArbitrageType.REVERSAL
                    legs = [
                        {"action": "BUY", "type": "CALL", "strike": K, "price": call.ask},
                        {"action": "SELL", "type": "PUT", "strike": K, "price": put.bid},
                        {"action": "SELL", "type": "STOCK", "shares": 100, "price": S},
                    ]
                
                opp = ArbitrageOpportunity(
                    type=ArbitrageType.PUT_CALL_PARITY,
                    symbol=chain.underlying.symbol,
                    expected_profit=net_edge,
                    expected_profit_pct=net_edge / (S * 100) * 100,
                    max_loss=0,  # Risk-free if executed properly
                    risk_free=True,
                    confidence=min(0.9, violation_pct / 2),
                    legs=legs,
                    gross_edge=abs(violation) * 100,
                    commission_cost=commission,
                    slippage_estimate=slippage,
                    net_edge=net_edge,
                    net_delta=0,  # Delta neutral
                    underlying_price=S,
                    expiration=call.expiration,
                    days_to_expiration=call.days_to_expiration,
                    min_volume=min(call.volume, put.volume),
                    min_open_interest=min(call.open_interest, put.open_interest),
                    max_spread_pct=max(call.spread_pct, put.spread_pct),
                    notes=[f"Parity violation: ${violation:.4f} ({violation_pct:.2f}%)"]
                )
                opportunities.append(opp)
        
        return opportunities
    
    def scan_box_spreads(self, chain: OptionsChain) -> List[ArbitrageOpportunity]:
        """
        Scan for box spread arbitrage.
        
        Box Spread = Bull Call Spread + Bear Put Spread
        - Buy Call K1, Sell Call K2 (K1 < K2)
        - Buy Put K2, Sell Put K1
        
        At expiration, value = K2 - K1 (guaranteed)
        Present value should be (K2 - K1) * e^(-rT)
        """
        opportunities = []
        S = chain.underlying.mark
        r = self.risk_free_rate
        
        for exp_key, calls in chain.calls.items():
            puts = chain.puts.get(exp_key, [])
            if not puts:
                continue
            
            # Sort by strike
            calls_sorted = sorted(calls, key=lambda x: x.strike)
            puts_sorted = sorted(puts, key=lambda x: x.strike)
            
            put_by_strike = {p.strike: p for p in puts_sorted}
            
            # Check all strike pairs
            for i, call_low in enumerate(calls_sorted[:-1]):
                if not self._check_liquidity(call_low):
                    continue
                
                put_low = put_by_strike.get(call_low.strike)
                if not put_low or not self._check_liquidity(put_low):
                    continue
                
                for call_high in calls_sorted[i+1:]:
                    if not self._check_liquidity(call_high):
                        continue
                    
                    put_high = put_by_strike.get(call_high.strike)
                    if not put_high or not self._check_liquidity(put_high):
                        continue
                    
                    K1 = call_low.strike
                    K2 = call_high.strike
                    T = call_low.days_to_expiration / 365
                    
                    if T <= 0:
                        continue
                    
                    # Theoretical box value
                    box_value = K2 - K1
                    pv_box = box_value * np.exp(-r * T)
                    
                    # Market cost of box (long box)
                    # Buy call K1, sell call K2, buy put K2, sell put K1
                    long_box_cost = (
                        call_low.ask - call_high.bid +
                        put_high.ask - put_low.bid
                    )
                    
                    # Short box cost
                    short_box_credit = (
                        call_low.bid - call_high.ask +
                        put_high.bid - put_low.ask
                    )
                    
                    # Check for arbitrage
                    long_edge = pv_box - long_box_cost
                    short_edge = -short_box_credit - pv_box
                    
                    edge = max(long_edge, short_edge)
                    edge_pct = edge / box_value * 100
                    
                    if edge_pct < self.min_edge_pct:
                        continue
                    
                    # Calculate costs
                    num_contracts = 4 * 2  # 4 legs
                    commission = num_contracts * self.commission_per_contract
                    total_spread = sum([
                        call_low.spread, call_high.spread,
                        put_low.spread, put_high.spread
                    ])
                    slippage = total_spread * 0.5 * self.slippage_pct * 100
                    net_edge = edge * 100 - commission - slippage
                    
                    if net_edge <= 0:
                        continue
                    
                    if long_edge > short_edge:
                        direction = "LONG"
                        legs = [
                            {"action": "BUY", "type": "CALL", "strike": K1, "price": call_low.ask},
                            {"action": "SELL", "type": "CALL", "strike": K2, "price": call_high.bid},
                            {"action": "BUY", "type": "PUT", "strike": K2, "price": put_high.ask},
                            {"action": "SELL", "type": "PUT", "strike": K1, "price": put_low.bid},
                        ]
                    else:
                        direction = "SHORT"
                        legs = [
                            {"action": "SELL", "type": "CALL", "strike": K1, "price": call_low.bid},
                            {"action": "BUY", "type": "CALL", "strike": K2, "price": call_high.ask},
                            {"action": "SELL", "type": "PUT", "strike": K2, "price": put_high.bid},
                            {"action": "BUY", "type": "PUT", "strike": K1, "price": put_low.ask},
                        ]
                    
                    opp = ArbitrageOpportunity(
                        type=ArbitrageType.BOX_SPREAD,
                        symbol=chain.underlying.symbol,
                        expected_profit=net_edge,
                        expected_profit_pct=edge_pct,
                        max_loss=0,
                        risk_free=True,
                        confidence=min(0.95, edge_pct / 1.5),
                        legs=legs,
                        gross_edge=edge * 100,
                        commission_cost=commission,
                        slippage_estimate=slippage,
                        net_edge=net_edge,
                        net_delta=0,
                        net_gamma=0,
                        net_vega=0,
                        underlying_price=S,
                        expiration=call_low.expiration,
                        days_to_expiration=call_low.days_to_expiration,
                        notes=[f"{direction} box: K1={K1}, K2={K2}, Edge=${edge:.4f}"]
                    )
                    opportunities.append(opp)
        
        return opportunities
    
    def scan_conversions_reversals(self, chain: OptionsChain) -> List[ArbitrageOpportunity]:
        """
        Scan for conversion and reversal arbitrage.
        
        Conversion: Long stock + Long put + Short call (same strike)
        Reversal: Short stock + Short put + Long call (same strike)
        
        Should yield risk-free rate; any deviation is arbitrage.
        """
        # Already covered in put-call parity scan
        return []
    
    def scan_butterfly_mispricing(self, chain: OptionsChain) -> List[ArbitrageOpportunity]:
        """
        Scan for butterfly spread mispricing.
        
        Butterfly: Buy 1 low strike, sell 2 middle, buy 1 high strike
        Should never have negative value (free money if it does).
        """
        opportunities = []
        S = chain.underlying.mark
        
        for exp_key, options in chain.calls.items():
            opps = self._check_butterfly_strikes(options, "CALL", chain.underlying.symbol, S)
            opportunities.extend(opps)
        
        for exp_key, options in chain.puts.items():
            opps = self._check_butterfly_strikes(options, "PUT", chain.underlying.symbol, S)
            opportunities.extend(opps)
        
        return opportunities
    
    def _check_butterfly_strikes(
        self,
        options: List[OptionQuote],
        option_type: str,
        symbol: str,
        underlying_price: float
    ) -> List[ArbitrageOpportunity]:
        """Check for butterfly mispricing across strikes."""
        opportunities = []
        options_sorted = sorted(options, key=lambda x: x.strike)
        
        for i in range(len(options_sorted) - 2):
            low = options_sorted[i]
            
            # Find middle strikes with equal spacing
            for j in range(i + 1, len(options_sorted) - 1):
                mid = options_sorted[j]
                
                # Look for high strike with same spacing
                target_high = 2 * mid.strike - low.strike
                high = next(
                    (o for o in options_sorted[j+1:] 
                     if abs(o.strike - target_high) < 0.01),
                    None
                )
                
                if not high:
                    continue
                
                # Check liquidity
                if not all(self._check_liquidity(o) for o in [low, mid, high]):
                    continue
                
                # Long butterfly cost (should be positive)
                # Buy low, sell 2 mid, buy high
                cost = low.ask - 2 * mid.bid + high.ask
                
                # If cost is negative, free money
                if cost < -0.05:  # Allow small threshold
                    edge = abs(cost)
                    num_contracts = 4
                    commission = num_contracts * self.commission_per_contract
                    slippage = (low.spread + 2 * mid.spread + high.spread) * 0.5 * self.slippage_pct * 100
                    net_edge = edge * 100 - commission - slippage
                    
                    if net_edge > 0:
                        opp = ArbitrageOpportunity(
                            type=ArbitrageType.BUTTERFLY,
                            symbol=symbol,
                            expected_profit=net_edge,
                            expected_profit_pct=edge / mid.strike * 100,
                            max_loss=0,
                            risk_free=True,
                            confidence=0.85,
                            legs=[
                                {"action": "BUY", "type": option_type, "strike": low.strike, "price": low.ask},
                                {"action": "SELL", "type": option_type, "strike": mid.strike, "price": mid.bid, "qty": 2},
                                {"action": "BUY", "type": option_type, "strike": high.strike, "price": high.ask},
                            ],
                            gross_edge=edge * 100,
                            commission_cost=commission,
                            slippage_estimate=slippage,
                            net_edge=net_edge,
                            underlying_price=underlying_price,
                            expiration=low.expiration,
                            days_to_expiration=low.days_to_expiration,
                            notes=[f"Butterfly credit: {option_type} K={low.strike}/{mid.strike}/{high.strike}"]
                        )
                        opportunities.append(opp)
        
        return opportunities
    
    def scan_vertical_mispricing(self, chain: OptionsChain) -> List[ArbitrageOpportunity]:
        """
        Scan for vertical spread mispricing.
        
        Bull call spread should cost less than strike difference.
        Bear put spread should cost less than strike difference.
        Credit spreads should not exceed strike difference.
        """
        opportunities = []
        
        for exp_key, calls in chain.calls.items():
            opps = self._check_vertical_spreads(calls, "CALL", chain.underlying.symbol, chain.underlying.mark)
            opportunities.extend(opps)
        
        for exp_key, puts in chain.puts.items():
            opps = self._check_vertical_spreads(puts, "PUT", chain.underlying.symbol, chain.underlying.mark)
            opportunities.extend(opps)
        
        return opportunities
    
    def _check_vertical_spreads(
        self,
        options: List[OptionQuote],
        option_type: str,
        symbol: str,
        underlying_price: float
    ) -> List[ArbitrageOpportunity]:
        """Check for vertical spread arbitrage."""
        opportunities = []
        options_sorted = sorted(options, key=lambda x: x.strike)
        
        for i, low in enumerate(options_sorted[:-1]):
            for high in options_sorted[i+1:]:
                if not self._check_liquidity(low) or not self._check_liquidity(high):
                    continue
                
                strike_diff = high.strike - low.strike
                
                if option_type == "CALL":
                    # Bull call spread: buy low, sell high
                    debit = low.ask - high.bid
                    
                    # Should not cost more than strike difference
                    if debit > strike_diff:
                        edge = debit - strike_diff
                        # Sell the spread for credit
                        legs = [
                            {"action": "SELL", "type": "CALL", "strike": low.strike, "price": low.bid},
                            {"action": "BUY", "type": "CALL", "strike": high.strike, "price": high.ask},
                        ]
                    else:
                        continue
                else:  # PUT
                    # Bear put spread: buy high, sell low
                    debit = high.ask - low.bid
                    
                    if debit > strike_diff:
                        edge = debit - strike_diff
                        legs = [
                            {"action": "SELL", "type": "PUT", "strike": high.strike, "price": high.bid},
                            {"action": "BUY", "type": "PUT", "strike": low.strike, "price": low.ask},
                        ]
                    else:
                        continue
                
                edge_pct = edge / strike_diff * 100
                if edge_pct < self.min_edge_pct:
                    continue
                
                num_contracts = 2
                commission = num_contracts * self.commission_per_contract
                slippage = (low.spread + high.spread) * 0.5 * self.slippage_pct * 100
                net_edge = edge * 100 - commission - slippage
                
                if net_edge > 0:
                    opp = ArbitrageOpportunity(
                        type=ArbitrageType.VERTICAL_MISPRICING,
                        symbol=symbol,
                        expected_profit=net_edge,
                        expected_profit_pct=edge_pct,
                        max_loss=0,
                        risk_free=True,
                        confidence=0.8,
                        legs=legs,
                        gross_edge=edge * 100,
                        commission_cost=commission,
                        slippage_estimate=slippage,
                        net_edge=net_edge,
                        underlying_price=underlying_price,
                        expiration=low.expiration,
                        days_to_expiration=low.days_to_expiration,
                        notes=[f"Vertical {option_type} overpriced: K={low.strike}/{high.strike}"]
                    )
                    opportunities.append(opp)
        
        return opportunities
    
    def scan_calendar_anomalies(self, chain: OptionsChain) -> List[ArbitrageOpportunity]:
        """
        Scan for calendar spread anomalies.
        
        Near-term options should not be more expensive than far-term
        at the same strike (time value should increase with time).
        """
        opportunities = []
        
        # Group options by strike
        call_by_strike: Dict[float, List[OptionQuote]] = {}
        put_by_strike: Dict[float, List[OptionQuote]] = {}
        
        for exp_key, calls in chain.calls.items():
            for opt in calls:
                if opt.strike not in call_by_strike:
                    call_by_strike[opt.strike] = []
                call_by_strike[opt.strike].append(opt)
        
        for exp_key, puts in chain.puts.items():
            for opt in puts:
                if opt.strike not in put_by_strike:
                    put_by_strike[opt.strike] = []
                put_by_strike[opt.strike].append(opt)
        
        # Check calls
        for strike, options in call_by_strike.items():
            opps = self._check_calendar_at_strike(
                options, "CALL", chain.underlying.symbol, chain.underlying.mark
            )
            opportunities.extend(opps)
        
        # Check puts
        for strike, options in put_by_strike.items():
            opps = self._check_calendar_at_strike(
                options, "PUT", chain.underlying.symbol, chain.underlying.mark
            )
            opportunities.extend(opps)
        
        return opportunities
    
    def _check_calendar_at_strike(
        self,
        options: List[OptionQuote],
        option_type: str,
        symbol: str,
        underlying_price: float
    ) -> List[ArbitrageOpportunity]:
        """Check for calendar anomalies at a specific strike."""
        opportunities = []
        
        if len(options) < 2:
            return opportunities
        
        # Sort by expiration
        options_sorted = sorted(options, key=lambda x: x.expiration)
        
        for i, near in enumerate(options_sorted[:-1]):
            for far in options_sorted[i+1:]:
                if not self._check_liquidity(near) or not self._check_liquidity(far):
                    continue
                
                # Near-term should be cheaper than far-term
                # If near ask > far bid, there's an anomaly
                if near.ask > far.bid:
                    edge = near.ask - far.bid
                    edge_pct = edge / near.ask * 100
                    
                    if edge_pct < self.min_edge_pct:
                        continue
                    
                    num_contracts = 2
                    commission = num_contracts * self.commission_per_contract
                    slippage = (near.spread + far.spread) * 0.5 * self.slippage_pct * 100
                    net_edge = edge * 100 - commission - slippage
                    
                    if net_edge > 0:
                        opp = ArbitrageOpportunity(
                            type=ArbitrageType.CALENDAR_SPREAD,
                            symbol=symbol,
                            expected_profit=net_edge,
                            expected_profit_pct=edge_pct,
                            max_loss=edge * 100,  # Not risk-free
                            risk_free=False,
                            confidence=0.7,
                            legs=[
                                {"action": "SELL", "type": option_type, "strike": near.strike, 
                                 "price": near.bid, "expiration": near.expiration.isoformat()},
                                {"action": "BUY", "type": option_type, "strike": far.strike,
                                 "price": far.ask, "expiration": far.expiration.isoformat()},
                            ],
                            gross_edge=edge * 100,
                            commission_cost=commission,
                            slippage_estimate=slippage,
                            net_edge=net_edge,
                            underlying_price=underlying_price,
                            notes=[f"Calendar anomaly: {option_type} K={near.strike}, "
                                   f"Near={near.days_to_expiration}d > Far={far.days_to_expiration}d"]
                        )
                        opportunities.append(opp)
        
        return opportunities
    
    def scan_volatility_skew(self, chain: OptionsChain) -> List[ArbitrageOpportunity]:
        """
        Scan for volatility skew anomalies.
        
        Look for significant deviations from expected skew patterns
        that might indicate mispricing.
        """
        opportunities = []
        S = chain.underlying.mark
        
        for exp_key, calls in chain.calls.items():
            puts = chain.puts.get(exp_key, [])
            if not puts:
                continue
            
            # Build IV surface for this expiration
            iv_data = []
            for call in calls:
                if call.implied_volatility and self._check_liquidity(call):
                    moneyness = call.strike / S
                    iv_data.append({
                        "strike": call.strike,
                        "moneyness": moneyness,
                        "iv": call.implied_volatility,
                        "type": "call",
                        "option": call
                    })
            
            for put in puts:
                if put.implied_volatility and self._check_liquidity(put):
                    moneyness = put.strike / S
                    iv_data.append({
                        "strike": put.strike,
                        "moneyness": moneyness,
                        "iv": put.implied_volatility,
                        "type": "put",
                        "option": put
                    })
            
            if len(iv_data) < 5:
                continue
            
            # Calculate IV statistics
            ivs = [d["iv"] for d in iv_data]
            mean_iv = np.mean(ivs)
            std_iv = np.std(ivs)
            
            if std_iv < 0.01:  # Too uniform
                continue
            
            # Find outliers (potential mispricing)
            for data in iv_data:
                z_score = (data["iv"] - mean_iv) / std_iv
                
                if abs(z_score) > 2.0:  # Significant deviation
                    opt = data["option"]
                    
                    # This is a potential mispricing
                    # If IV is too high, sell; if too low, buy
                    if z_score > 2.0:
                        action = "SELL"
                        edge_direction = "overpriced"
                    else:
                        action = "BUY"
                        edge_direction = "underpriced"
                    
                    # Estimate edge based on IV deviation
                    iv_edge = abs(data["iv"] - mean_iv)
                    vega = opt.vega or 0.1
                    price_edge = iv_edge * vega * 100
                    
                    if price_edge < 0.5:  # Minimum edge
                        continue
                    
                    opp = ArbitrageOpportunity(
                        type=ArbitrageType.VOLATILITY_SKEW,
                        symbol=chain.underlying.symbol,
                        expected_profit=price_edge,
                        expected_profit_pct=price_edge / opt.mid_price * 100 if opt.mid_price > 0 else 0,
                        max_loss=price_edge * 2,  # Not risk-free
                        risk_free=False,
                        confidence=min(0.6, abs(z_score) / 4),
                        legs=[{
                            "action": action,
                            "type": data["type"].upper(),
                            "strike": opt.strike,
                            "price": opt.bid if action == "SELL" else opt.ask,
                            "iv": data["iv"],
                            "z_score": z_score
                        }],
                        net_vega=-vega * 100 if action == "SELL" else vega * 100,
                        underlying_price=S,
                        expiration=opt.expiration,
                        days_to_expiration=opt.days_to_expiration,
                        notes=[f"IV {edge_direction}: {data['iv']*100:.1f}% vs mean {mean_iv*100:.1f}%, z={z_score:.2f}"]
                    )
                    opportunities.append(opp)
        
        return opportunities
    
    def _check_liquidity(self, option: OptionQuote) -> bool:
        """Check if option meets liquidity requirements."""
        if option.open_interest < self.min_open_interest:
            return False
        if option.volume < self.min_volume:
            return False
        if option.spread_pct > self.max_bid_ask_spread_pct:
            return False
        if option.bid <= 0 or option.ask <= 0:
            return False
        return True