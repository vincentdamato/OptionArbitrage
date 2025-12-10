"""
Schwab API Client for Options Market Data.
Handles authentication, real-time quotes, and options chain fetching.
"""

import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Any, Union
from dataclasses import dataclass, field
from enum import Enum

import httpx
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential
import pandas as pd
import numpy as np

try:
    from schwab import auth, client
    from schwab.client import Client
    SCHWAB_PY_AVAILABLE = True
except ImportError:
    SCHWAB_PY_AVAILABLE = False
    Client = None  # Type placeholder
    logger.warning("schwab-py not installed. Install with: pip install schwab-py")


class ContractType(Enum):
    CALL = "CALL"
    PUT = "PUT"
    ALL = "ALL"


class OptionStrategy(Enum):
    SINGLE = "SINGLE"
    COVERED = "COVERED"
    VERTICAL = "VERTICAL"
    CALENDAR = "CALENDAR"
    STRANGLE = "STRANGLE"
    STRADDLE = "STRADDLE"
    BUTTERFLY = "BUTTERFLY"
    CONDOR = "CONDOR"
    DIAGONAL = "DIAGONAL"
    COLLAR = "COLLAR"
    ROLL = "ROLL"


@dataclass
class OptionQuote:
    """Represents a single option contract quote."""
    symbol: str
    underlying: str
    strike: float
    expiration: datetime
    contract_type: ContractType
    bid: float
    ask: float
    last: float
    mark: float
    volume: int
    open_interest: int
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    rho: Optional[float] = None
    implied_volatility: Optional[float] = None
    intrinsic_value: float = 0.0
    extrinsic_value: float = 0.0
    theoretical_value: Optional[float] = None
    time_value: float = 0.0
    days_to_expiration: int = 0
    in_the_money: bool = False
    
    @property
    def mid_price(self) -> float:
        return (self.bid + self.ask) / 2
    
    @property
    def spread(self) -> float:
        return self.ask - self.bid
    
    @property
    def spread_pct(self) -> float:
        if self.mid_price > 0:
            return (self.spread / self.mid_price) * 100
        return float('inf')


@dataclass
class UnderlyingQuote:
    """Represents underlying stock/ETF quote."""
    symbol: str
    bid: float
    ask: float
    last: float
    mark: float
    volume: int
    high_52week: float = 0.0
    low_52week: float = 0.0
    dividend_yield: float = 0.0
    pe_ratio: Optional[float] = None
    
    @property
    def mid_price(self) -> float:
        return (self.bid + self.ask) / 2


@dataclass 
class OptionsChain:
    """Complete options chain for an underlying."""
    underlying: UnderlyingQuote
    calls: Dict[str, List[OptionQuote]] = field(default_factory=dict)
    puts: Dict[str, List[OptionQuote]] = field(default_factory=dict)
    expirations: List[datetime] = field(default_factory=list)
    strikes: List[float] = field(default_factory=list)
    fetch_time: datetime = field(default_factory=datetime.now)
    
    def get_options_at_strike(self, strike: float, expiration: datetime) -> Dict[str, Optional[OptionQuote]]:
        exp_key = expiration.strftime("%Y-%m-%d")
        call = next((o for o in self.calls.get(exp_key, []) if o.strike == strike), None)
        put = next((o for o in self.puts.get(exp_key, []) if o.strike == strike), None)
        return {"call": call, "put": put}
    
    def to_dataframe(self) -> pd.DataFrame:
        records = []
        for exp_key, options in self.calls.items():
            for opt in options:
                records.append({
                    "underlying": opt.underlying, "expiration": opt.expiration,
                    "strike": opt.strike, "type": "CALL", "bid": opt.bid,
                    "ask": opt.ask, "mid": opt.mid_price, "last": opt.last,
                    "volume": opt.volume, "open_interest": opt.open_interest,
                    "iv": opt.implied_volatility, "delta": opt.delta,
                    "gamma": opt.gamma, "theta": opt.theta, "vega": opt.vega,
                    "dte": opt.days_to_expiration, "itm": opt.in_the_money,
                    "spread_pct": opt.spread_pct
                })
        for exp_key, options in self.puts.items():
            for opt in options:
                records.append({
                    "underlying": opt.underlying, "expiration": opt.expiration,
                    "strike": opt.strike, "type": "PUT", "bid": opt.bid,
                    "ask": opt.ask, "mid": opt.mid_price, "last": opt.last,
                    "volume": opt.volume, "open_interest": opt.open_interest,
                    "iv": opt.implied_volatility, "delta": opt.delta,
                    "gamma": opt.gamma, "theta": opt.theta, "vega": opt.vega,
                    "dte": opt.days_to_expiration, "itm": opt.in_the_money,
                    "spread_pct": opt.spread_pct
                })
        return pd.DataFrame(records)


class SchwabClient:
    """Schwab API client wrapper for options market data."""
    
    def __init__(
        self,
        api_key: str,
        app_secret: str,
        callback_url: str = "https://127.0.0.1:8182",
        token_path: str = "~/.schwab/token.json"
    ):
        self.api_key = api_key
        self.app_secret = app_secret
        self.callback_url = callback_url
        self.token_path = Path(token_path).expanduser()
        self._client = None
        self._account_hash: Optional[str] = None
        self.token_path.parent.mkdir(parents=True, exist_ok=True)
    
    def _ensure_client(self):
        if self._client is None:
            self._client = self._authenticate()
        return self._client
    
    def _authenticate(self):
        if not SCHWAB_PY_AVAILABLE:
            raise ImportError("schwab-py is required. Install with: pip install schwab-py")
        
        logger.info("Authenticating with Schwab API...")
        try:
            c = auth.easy_client(
                self.api_key, self.app_secret,
                self.callback_url, str(self.token_path)
            )
            logger.info("Successfully authenticated with Schwab API")
            return c
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            raise
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def get_quote(self, symbol: str) -> UnderlyingQuote:
        client = self._ensure_client()
        resp = client.get_quote(symbol)
        resp.raise_for_status()
        data = resp.json()
        quote_data = data.get(symbol, {}).get("quote", {})
        
        return UnderlyingQuote(
            symbol=symbol,
            bid=quote_data.get("bidPrice", 0),
            ask=quote_data.get("askPrice", 0),
            last=quote_data.get("lastPrice", 0),
            mark=quote_data.get("mark", 0),
            volume=quote_data.get("totalVolume", 0),
            high_52week=quote_data.get("52WeekHigh", 0),
            low_52week=quote_data.get("52WeekLow", 0),
            dividend_yield=quote_data.get("divYield", 0),
            pe_ratio=quote_data.get("peRatio")
        )
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def get_quotes(self, symbols: List[str]) -> Dict[str, UnderlyingQuote]:
        client = self._ensure_client()
        quotes = {}
        for i in range(0, len(symbols), 500):
            batch = symbols[i:i+500]
            resp = client.get_quotes(batch)
            resp.raise_for_status()
            data = resp.json()
            
            for symbol, quote_info in data.items():
                quote_data = quote_info.get("quote", {})
                quotes[symbol] = UnderlyingQuote(
                    symbol=symbol,
                    bid=quote_data.get("bidPrice", 0),
                    ask=quote_data.get("askPrice", 0),
                    last=quote_data.get("lastPrice", 0),
                    mark=quote_data.get("mark", 0),
                    volume=quote_data.get("totalVolume", 0),
                    high_52week=quote_data.get("52WeekHigh", 0),
                    low_52week=quote_data.get("52WeekLow", 0)
                )
        return quotes
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def get_option_chain(
        self,
        symbol: str,
        contract_type: ContractType = ContractType.ALL,
        strike_count: Optional[int] = None,
        include_quotes: bool = True,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        min_dte: int = 0,
        max_dte: int = 60
    ) -> OptionsChain:
        client = self._ensure_client()
        
        if from_date is None:
            from_date = datetime.now() + timedelta(days=min_dte)
        if to_date is None:
            to_date = datetime.now() + timedelta(days=max_dte)
        
        params = {
            "symbol": symbol,
            "contractType": contract_type.value if contract_type != ContractType.ALL else "ALL",
            "includeQuotes": "TRUE" if include_quotes else "FALSE",
            "fromDate": from_date.strftime("%Y-%m-%d"),
            "toDate": to_date.strftime("%Y-%m-%d"),
        }
        if strike_count:
            params["strikeCount"] = strike_count
        
        resp = client.get_option_chain(**params)
        resp.raise_for_status()
        data = resp.json()
        
        underlying_data = data.get("underlying", {})
        underlying = UnderlyingQuote(
            symbol=symbol,
            bid=underlying_data.get("bid", 0),
            ask=underlying_data.get("ask", 0),
            last=underlying_data.get("last", 0),
            mark=underlying_data.get("mark", 0),
            volume=underlying_data.get("totalVolume", 0),
            high_52week=underlying_data.get("fiftyTwoWeekHigh", 0),
            low_52week=underlying_data.get("fiftyTwoWeekLow", 0)
        )
        
        chain = OptionsChain(underlying=underlying)
        
        call_map = data.get("callExpDateMap", {})
        for exp_date_str, strikes_data in call_map.items():
            exp_date = self._parse_expiration(exp_date_str)
            chain.expirations.append(exp_date)
            exp_key = exp_date.strftime("%Y-%m-%d")
            chain.calls[exp_key] = []
            
            for strike_str, options in strikes_data.items():
                strike = float(strike_str)
                if strike not in chain.strikes:
                    chain.strikes.append(strike)
                for opt_data in options:
                    option = self._parse_option(opt_data, symbol, strike, exp_date, ContractType.CALL)
                    chain.calls[exp_key].append(option)
        
        put_map = data.get("putExpDateMap", {})
        for exp_date_str, strikes_data in put_map.items():
            exp_date = self._parse_expiration(exp_date_str)
            if exp_date not in chain.expirations:
                chain.expirations.append(exp_date)
            exp_key = exp_date.strftime("%Y-%m-%d")
            chain.puts[exp_key] = []
            
            for strike_str, options in strikes_data.items():
                strike = float(strike_str)
                if strike not in chain.strikes:
                    chain.strikes.append(strike)
                for opt_data in options:
                    option = self._parse_option(opt_data, symbol, strike, exp_date, ContractType.PUT)
                    chain.puts[exp_key].append(option)
        
        chain.expirations.sort()
        chain.strikes.sort()
        logger.info(f"Fetched {symbol} chain: {len(chain.expirations)} expirations, {len(chain.strikes)} strikes")
        return chain
    
    def _parse_expiration(self, exp_str: str) -> datetime:
        date_part = exp_str.split(":")[0]
        return datetime.strptime(date_part, "%Y-%m-%d")
    
    def _parse_option(self, data: Dict, underlying: str, strike: float,
                      expiration: datetime, contract_type: ContractType) -> OptionQuote:
        return OptionQuote(
            symbol=data.get("symbol", ""),
            underlying=underlying,
            strike=strike,
            expiration=expiration,
            contract_type=contract_type,
            bid=data.get("bid", 0),
            ask=data.get("ask", 0),
            last=data.get("last", 0),
            mark=data.get("mark", 0),
            volume=data.get("totalVolume", 0),
            open_interest=data.get("openInterest", 0),
            delta=data.get("delta"),
            gamma=data.get("gamma"),
            theta=data.get("theta"),
            vega=data.get("vega"),
            rho=data.get("rho"),
            implied_volatility=data.get("volatility"),
            intrinsic_value=data.get("intrinsicValue", 0),
            extrinsic_value=data.get("extrinsicValue", 0),
            theoretical_value=data.get("theoreticalOptionValue"),
            time_value=data.get("timeValue", 0),
            days_to_expiration=data.get("daysToExpiration", 0),
            in_the_money=data.get("inTheMoney", False)
        )
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def get_price_history(
        self,
        symbol: str,
        period_type: str = "month",
        period: int = 1,
        frequency_type: str = "daily",
        frequency: int = 1,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        client = self._ensure_client()
        
        if frequency_type == "daily":
            resp = client.get_price_history_every_day(symbol, start_datetime=start_date, end_datetime=end_date)
        elif frequency_type == "minute":
            resp = client.get_price_history_every_minute(symbol, start_datetime=start_date, end_datetime=end_date)
        elif frequency_type == "weekly":
            resp = client.get_price_history_every_week(symbol, start_datetime=start_date, end_datetime=end_date)
        else:
            resp = client.get_price_history(symbol)
        
        resp.raise_for_status()
        data = resp.json()
        candles = data.get("candles", [])
        if not candles:
            return pd.DataFrame()
        
        df = pd.DataFrame(candles)
        df["datetime"] = pd.to_datetime(df["datetime"], unit="ms")
        df.set_index("datetime", inplace=True)
        df.columns = ["open", "high", "low", "close", "volume"]
        return df
    
    def get_account_hash(self) -> str:
        if self._account_hash is None:
            client = self._ensure_client()
            resp = client.get_account_numbers()
            resp.raise_for_status()
            accounts = resp.json()
            if accounts:
                self._account_hash = accounts[0]["hashValue"]
        return self._account_hash