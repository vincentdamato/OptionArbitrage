"""
Core Orchestrator for Options Arbitrage Platform.
Coordinates scanning, analysis, and execution workflows.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import threading
import queue
from loguru import logger

from api.schwab_client import SchwabClient, OptionsChain, ContractType
from strategies.arbitrage_scanner import ArbitrageScanner, ArbitrageOpportunity
from analysis.greeks_engine import GreeksCalculator
from analysis.statistical_analyzer import StatisticalAnalyzer
from backtesting.backtest_engine import BacktestEngine


@dataclass
class ScanResult:
    """Container for scan results."""
    symbol: str
    opportunities: List[ArbitrageOpportunity]
    chain_data: Optional[OptionsChain]
    volatility_metrics: Optional[Dict]
    scan_time: datetime
    scan_duration_ms: float
    error: Optional[str] = None


@dataclass
class PortfolioState:
    """Current portfolio state."""
    positions: List[Dict] = field(default_factory=list)
    total_delta: float = 0.0
    total_gamma: float = 0.0
    total_vega: float = 0.0
    total_theta: float = 0.0
    total_pnl: float = 0.0
    margin_used: float = 0.0
    buying_power: float = 0.0


class ArbitrageOrchestrator:
    """
    Main orchestrator for the arbitrage scanning platform.
    
    Responsibilities:
    - Coordinate API calls and data fetching
    - Run arbitrage scans across watchlist
    - Manage portfolio state
    - Handle callbacks for opportunities
    - Provide async scanning capabilities
    """
    
    def __init__(
        self,
        schwab_client: Optional[SchwabClient] = None,
        scanner: Optional[ArbitrageScanner] = None,
        analyzer: Optional[StatisticalAnalyzer] = None,
        backtester: Optional[BacktestEngine] = None,
        config: Optional[Dict] = None
    ):
        self.client = schwab_client
        self.scanner = scanner or ArbitrageScanner()
        self.analyzer = analyzer or StatisticalAnalyzer()
        self.backtester = backtester or BacktestEngine()
        self.greeks_calc = GreeksCalculator()
        
        # Configuration
        self.config = config or {}
        self.watchlist: List[str] = self.config.get("watchlist", [
            "SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA", "TSLA", "AMZN"
        ])
        
        # State
        self.portfolio = PortfolioState()
        self.scan_results: Dict[str, ScanResult] = {}
        self.all_opportunities: List[ArbitrageOpportunity] = []
        
        # Callbacks
        self._on_opportunity_callbacks: List[Callable] = []
        self._on_scan_complete_callbacks: List[Callable] = []
        
        # Threading
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._scan_queue: queue.Queue = queue.Queue()
        self._running = False
        self._scan_thread: Optional[threading.Thread] = None
        
        logger.info("ArbitrageOrchestrator initialized")
    
    def set_client(self, client: SchwabClient):
        """Set or update the Schwab client."""
        self.client = client
        logger.info("Schwab client configured")
    
    def update_watchlist(self, symbols: List[str]):
        """Update the watchlist."""
        self.watchlist = [s.upper().strip() for s in symbols if s.strip()]
        logger.info(f"Watchlist updated: {len(self.watchlist)} symbols")
    
    def register_opportunity_callback(self, callback: Callable[[ArbitrageOpportunity], None]):
        """Register callback for new opportunities."""
        self._on_opportunity_callbacks.append(callback)
    
    def register_scan_complete_callback(self, callback: Callable[[ScanResult], None]):
        """Register callback for scan completion."""
        self._on_scan_complete_callbacks.append(callback)
    
    def _notify_opportunity(self, opportunity: ArbitrageOpportunity):
        """Notify all registered callbacks of new opportunity."""
        for callback in self._on_opportunity_callbacks:
            try:
                callback(opportunity)
            except Exception as e:
                logger.error(f"Opportunity callback error: {e}")
    
    def _notify_scan_complete(self, result: ScanResult):
        """Notify all registered callbacks of scan completion."""
        for callback in self._on_scan_complete_callbacks:
            try:
                callback(result)
            except Exception as e:
                logger.error(f"Scan complete callback error: {e}")
    
    def scan_symbol(
        self,
        symbol: str,
        min_dte: int = 1,
        max_dte: int = 60,
        include_analysis: bool = True
    ) -> ScanResult:
        """
        Scan a single symbol for arbitrage opportunities.
        
        Args:
            symbol: Ticker symbol
            min_dte: Minimum days to expiration
            max_dte: Maximum days to expiration
            include_analysis: Include volatility analysis
        
        Returns:
            ScanResult with opportunities and analysis
        """
        start_time = datetime.now()
        
        try:
            # Fetch options chain
            if self.client:
                chain = self.client.get_option_chain(
                    symbol,
                    contract_type=ContractType.ALL,
                    min_dte=min_dte,
                    max_dte=max_dte
                )
            else:
                # Use mock data if no client
                from api.schwab_client import MockSchwabClient
                mock_client = MockSchwabClient()
                chain = mock_client.get_option_chain(symbol)
            
            # Scan for opportunities
            opportunities = self.scanner.scan_chain(chain)
            
            # Volatility analysis
            vol_metrics = None
            if include_analysis:
                # Extract IV data from chain
                iv_by_dte = {}
                for exp_key, calls in chain.calls.items():
                    for call in calls:
                        if call.implied_volatility and call.days_to_expiration > 0:
                            iv_by_dte[call.days_to_expiration] = call.implied_volatility
                            break
                
                if iv_by_dte:
                    vol_metrics = self.analyzer.analyze_iv_term_structure(iv_by_dte)
            
            duration = (datetime.now() - start_time).total_seconds() * 1000
            
            result = ScanResult(
                symbol=symbol,
                opportunities=opportunities,
                chain_data=chain,
                volatility_metrics=vol_metrics,
                scan_time=datetime.now(),
                scan_duration_ms=duration
            )
            
            # Cache result
            self.scan_results[symbol] = result
            
            # Add to all opportunities
            self.all_opportunities.extend(opportunities)
            
            # Notify callbacks
            self._notify_scan_complete(result)
            for opp in opportunities:
                self._notify_opportunity(opp)
            
            logger.info(f"Scanned {symbol}: {len(opportunities)} opportunities in {duration:.0f}ms")
            
            return result
            
        except Exception as e:
            logger.error(f"Error scanning {symbol}: {e}")
            return ScanResult(
                symbol=symbol,
                opportunities=[],
                chain_data=None,
                volatility_metrics=None,
                scan_time=datetime.now(),
                scan_duration_ms=0,
                error=str(e)
            )
    
    def scan_watchlist(
        self,
        min_dte: int = 1,
        max_dte: int = 60,
        parallel: bool = True
    ) -> List[ScanResult]:
        """
        Scan entire watchlist for opportunities.
        
        Args:
            min_dte: Minimum days to expiration
            max_dte: Maximum days to expiration
            parallel: Use parallel processing
        
        Returns:
            List of ScanResults for all symbols
        """
        logger.info(f"Starting watchlist scan: {len(self.watchlist)} symbols")
        start_time = datetime.now()
        
        # Clear previous opportunities
        self.all_opportunities = []
        
        results = []
        
        if parallel:
            # Parallel scanning
            futures = []
            for symbol in self.watchlist:
                future = self._executor.submit(
                    self.scan_symbol, symbol, min_dte, max_dte
                )
                futures.append(future)
            
            for future in futures:
                try:
                    result = future.result(timeout=30)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Parallel scan error: {e}")
        else:
            # Sequential scanning
            for symbol in self.watchlist:
                result = self.scan_symbol(symbol, min_dte, max_dte)
                results.append(result)
        
        # Sort all opportunities by expected profit
        self.all_opportunities.sort(
            key=lambda x: x.expected_profit, reverse=True
        )
        
        duration = (datetime.now() - start_time).total_seconds()
        total_opps = sum(len(r.opportunities) for r in results)
        
        logger.info(
            f"Watchlist scan complete: {total_opps} opportunities "
            f"across {len(self.watchlist)} symbols in {duration:.1f}s"
        )
        
        return results
    
    def get_top_opportunities(
        self,
        n: int = 10,
        risk_free_only: bool = False,
        min_confidence: float = 0.0,
        symbol_filter: Optional[List[str]] = None
    ) -> List[ArbitrageOpportunity]:
        """
        Get top N opportunities with filters.
        
        Args:
            n: Number of opportunities to return
            risk_free_only: Only return risk-free opportunities
            min_confidence: Minimum confidence threshold
            symbol_filter: Filter to specific symbols
        
        Returns:
            Filtered and sorted list of opportunities
        """
        filtered = self.all_opportunities
        
        if risk_free_only:
            filtered = [o for o in filtered if o.risk_free]
        
        if min_confidence > 0:
            filtered = [o for o in filtered if o.confidence >= min_confidence]
        
        if symbol_filter:
            filtered = [o for o in filtered if o.symbol in symbol_filter]
        
        return filtered[:n]
    
    def analyze_opportunity(
        self,
        opportunity: ArbitrageOpportunity,
        num_simulations: int = 10000
    ) -> Dict[str, Any]:
        """
        Deep analysis of a specific opportunity.
        
        Args:
            opportunity: The opportunity to analyze
            num_simulations: Number of Monte Carlo simulations
        
        Returns:
            Detailed analysis including Monte Carlo results
        """
        analysis = {
            "opportunity": opportunity.to_dict(),
            "monte_carlo": {},
            "greeks_analysis": {},
            "risk_metrics": {},
            "recommendation": ""
        }
        
        # Monte Carlo simulation for expected P&L
        expected_profit = opportunity.expected_profit
        profit_std = expected_profit * 0.3  # Assume 30% std dev
        
        simulated_profits = np.random.normal(
            expected_profit, profit_std, num_simulations
        )
        
        import numpy as np
        
        analysis["monte_carlo"] = {
            "mean_profit": float(np.mean(simulated_profits)),
            "median_profit": float(np.median(simulated_profits)),
            "std_profit": float(np.std(simulated_profits)),
            "prob_profit": float(np.mean(simulated_profits > 0)),
            "var_95": float(np.percentile(simulated_profits, 5)),
            "cvar_95": float(np.mean(simulated_profits[simulated_profits <= np.percentile(simulated_profits, 5)])),
            "best_case": float(np.percentile(simulated_profits, 95)),
            "worst_case": float(np.percentile(simulated_profits, 5))
        }
        
        # Greeks analysis
        analysis["greeks_analysis"] = {
            "net_delta": opportunity.net_delta,
            "net_gamma": opportunity.net_gamma,
            "net_vega": opportunity.net_vega,
            "net_theta": opportunity.net_theta,
            "delta_neutral": abs(opportunity.net_delta) < 0.05,
            "gamma_scalp_potential": opportunity.net_gamma > 0.01,
            "theta_positive": opportunity.net_theta > 0
        }
        
        # Risk metrics
        analysis["risk_metrics"] = {
            "max_loss": opportunity.max_loss,
            "risk_reward": (
                opportunity.expected_profit / abs(opportunity.max_loss)
                if opportunity.max_loss != 0 else float('inf')
            ),
            "breakeven_probability": opportunity.confidence,
            "time_decay_impact": opportunity.net_theta * opportunity.days_to_expiration
        }
        
        # Generate recommendation
        mc = analysis["monte_carlo"]
        if opportunity.risk_free and mc["prob_profit"] > 0.9:
            analysis["recommendation"] = "STRONG BUY - High confidence risk-free arbitrage"
        elif mc["prob_profit"] > 0.75 and mc["var_95"] > 0:
            analysis["recommendation"] = "BUY - Good statistical edge with limited downside"
        elif mc["prob_profit"] > 0.6:
            analysis["recommendation"] = "CONSIDER - Moderate edge, size appropriately"
        else:
            analysis["recommendation"] = "PASS - Insufficient edge or high risk"
        
        return analysis
    
    def start_continuous_scanning(
        self,
        interval_seconds: int = 60,
        callback: Optional[Callable] = None
    ):
        """
        Start continuous background scanning.
        
        Args:
            interval_seconds: Seconds between scans
            callback: Optional callback for each scan cycle
        """
        self._running = True
        
        def scan_loop():
            while self._running:
                try:
                    results = self.scan_watchlist()
                    if callback:
                        callback(results)
                except Exception as e:
                    logger.error(f"Continuous scan error: {e}")
                
                # Wait for interval
                for _ in range(interval_seconds):
                    if not self._running:
                        break
                    import time
                    time.sleep(1)
        
        self._scan_thread = threading.Thread(target=scan_loop, daemon=True)
        self._scan_thread.start()
        logger.info(f"Continuous scanning started (interval: {interval_seconds}s)")
    
    def stop_continuous_scanning(self):
        """Stop continuous background scanning."""
        self._running = False
        if self._scan_thread:
            self._scan_thread.join(timeout=5)
        logger.info("Continuous scanning stopped")
    
    def backtest_opportunities(
        self,
        opportunities: List[Dict],
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """
        Backtest historical opportunities.
        
        Args:
            opportunities: List of historical opportunities
            start_date: Backtest start date
            end_date: Backtest end date
        
        Returns:
            Backtest results
        """
        result = self.backtester.run_arbitrage_backtest(
            opportunities=opportunities,
            market_data=None,  # Would need historical data
            position_size=1
        )
        
        return {
            "summary": result.to_dict(),
            "report": self.backtester.generate_performance_report(result)
        }
    
    def get_portfolio_greeks(self) -> Dict[str, float]:
        """Get aggregate portfolio Greeks."""
        return {
            "delta": self.portfolio.total_delta,
            "gamma": self.portfolio.total_gamma,
            "vega": self.portfolio.total_vega,
            "theta": self.portfolio.total_theta
        }
    
    def calculate_hedge_requirements(self) -> Dict[str, Any]:
        """Calculate hedging requirements for current portfolio."""
        delta = self.portfolio.total_delta
        gamma = self.portfolio.total_gamma
        
        return {
            "shares_to_delta_hedge": -delta,
            "gamma_exposure": gamma,
            "vega_exposure": self.portfolio.total_vega,
            "recommendations": []
        }
    
    def export_opportunities(self, format: str = "json") -> str:
        """Export current opportunities to JSON or CSV."""
        import json
        
        data = [o.to_dict() for o in self.all_opportunities]
        
        if format == "json":
            return json.dumps(data, indent=2, default=str)
        elif format == "csv":
            import pandas as pd
            df = pd.DataFrame(data)
            return df.to_csv(index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def shutdown(self):
        """Clean shutdown of orchestrator."""
        self.stop_continuous_scanning()
        self._executor.shutdown(wait=False)
        logger.info("Orchestrator shutdown complete")


# Import numpy for analysis
import numpy as np


def create_orchestrator(
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
    config: Optional[Dict] = None
) -> ArbitrageOrchestrator:
    """
    Factory function to create configured orchestrator.
    
    Args:
        api_key: Schwab API key
        api_secret: Schwab API secret
        config: Additional configuration
    
    Returns:
        Configured ArbitrageOrchestrator
    """
    client = None
    
    if api_key and api_secret:
        client = SchwabClient(
            api_key=api_key,
            app_secret=api_secret
        )
    
    return ArbitrageOrchestrator(
        schwab_client=client,
        config=config or {}
    )