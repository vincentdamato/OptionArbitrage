"""
Backtesting Engine for Options Arbitrage Strategies.
Implements walk-forward optimization, transaction costs, and performance analysis.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger
from scipy import stats


@dataclass
class Trade:
    """Represents a single trade."""
    entry_time: datetime
    exit_time: Optional[datetime]
    symbol: str
    strategy: str
    direction: str  # long/short
    entry_price: float
    exit_price: Optional[float]
    quantity: int
    
    # P&L
    gross_pnl: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    net_pnl: float = 0.0
    
    # Greeks at entry
    entry_delta: float = 0.0
    entry_gamma: float = 0.0
    entry_vega: float = 0.0
    entry_theta: float = 0.0
    
    # Metadata
    is_winner: bool = False
    hold_time_days: float = 0.0
    max_favorable_excursion: float = 0.0
    max_adverse_excursion: float = 0.0
    notes: List[str] = field(default_factory=list)
    
    def close(self, exit_time: datetime, exit_price: float, commission: float = 0, slippage: float = 0):
        """Close the trade and calculate P&L."""
        self.exit_time = exit_time
        self.exit_price = exit_price
        self.commission += commission
        self.slippage = slippage
        
        if self.direction == "long":
            self.gross_pnl = (exit_price - self.entry_price) * self.quantity * 100
        else:
            self.gross_pnl = (self.entry_price - exit_price) * self.quantity * 100
        
        self.net_pnl = self.gross_pnl - self.commission - self.slippage
        self.is_winner = self.net_pnl > 0
        self.hold_time_days = (exit_time - self.entry_time).days


@dataclass
class BacktestResult:
    """Complete backtest results."""
    strategy_name: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    
    # P&L metrics
    total_pnl: float
    total_commission: float
    total_slippage: float
    net_pnl: float
    
    # Performance metrics
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    calmar_ratio: float
    
    # Risk metrics
    volatility: float
    var_95: float
    cvar_95: float
    
    # Trade quality
    avg_win: float
    avg_loss: float
    profit_factor: float
    expectancy: float
    avg_hold_time: float
    
    # Raw data
    trades: List[Trade] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=pd.Series)
    returns: pd.Series = field(default_factory=pd.Series)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary (excluding large arrays)."""
        return {
            "strategy_name": self.strategy_name,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "total_trades": self.total_trades,
            "win_rate": self.win_rate,
            "total_return": self.total_return,
            "annualized_return": self.annualized_return,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "max_drawdown": self.max_drawdown,
            "profit_factor": self.profit_factor,
            "expectancy": self.expectancy,
        }


class BacktestEngine:
    """
    Backtesting engine for options strategies.
    
    Features:
    - Walk-forward optimization
    - Realistic transaction costs
    - Slippage modeling
    - Performance analytics
    - Risk metrics (VaR, CVaR, drawdown)
    """
    
    def __init__(
        self,
        initial_capital: float = 100000,
        commission_per_contract: float = 0.65,
        slippage_model: str = "fixed",  # fixed, proportional, volatility
        slippage_value: float = 0.05,
        risk_free_rate: float = 0.05
    ):
        self.initial_capital = initial_capital
        self.commission_per_contract = commission_per_contract
        self.slippage_model = slippage_model
        self.slippage_value = slippage_value
        self.risk_free_rate = risk_free_rate
        
        self.trades: List[Trade] = []
        self.equity_curve: List[Tuple[datetime, float]] = []
        self.current_capital = initial_capital
    
    def calculate_slippage(
        self,
        price: float,
        quantity: int,
        bid_ask_spread: float = 0.05,
        volatility: float = 0.2
    ) -> float:
        """Calculate expected slippage based on model."""
        if self.slippage_model == "fixed":
            return self.slippage_value * quantity * 100
        elif self.slippage_model == "proportional":
            return price * self.slippage_value * quantity * 100
        elif self.slippage_model == "volatility":
            # Slippage proportional to volatility
            return price * volatility * self.slippage_value * quantity * 100
        else:
            return bid_ask_spread * 0.5 * quantity * 100
    
    def run_backtest(
        self,
        signals: pd.DataFrame,
        price_data: pd.DataFrame,
        strategy_name: str = "arbitrage",
        position_size: int = 1
    ) -> BacktestResult:
        """
        Run backtest on historical data.
        
        Args:
            signals: DataFrame with columns [date, signal, entry_price, exit_price, ...]
            price_data: OHLCV price data
            strategy_name: Name of strategy
            position_size: Number of contracts per trade
        
        Returns:
            BacktestResult with full analysis
        """
        self.trades = []
        self.equity_curve = []
        self.current_capital = self.initial_capital
        
        start_date = signals.index.min() if isinstance(signals.index, pd.DatetimeIndex) else signals["date"].min()
        end_date = signals.index.max() if isinstance(signals.index, pd.DatetimeIndex) else signals["date"].max()
        
        current_position: Optional[Trade] = None
        
        for idx, row in signals.iterrows():
            date = idx if isinstance(signals.index, pd.DatetimeIndex) else row["date"]
            signal = row.get("signal", 0)
            
            # Entry signal
            if signal != 0 and current_position is None:
                entry_price = row.get("entry_price", row.get("close", 0))
                commission = self.commission_per_contract * position_size * 2  # Entry
                
                current_position = Trade(
                    entry_time=date,
                    exit_time=None,
                    symbol=row.get("symbol", "UNKNOWN"),
                    strategy=strategy_name,
                    direction="long" if signal > 0 else "short",
                    entry_price=entry_price,
                    exit_price=None,
                    quantity=position_size,
                    commission=commission,
                    entry_delta=row.get("delta", 0),
                    entry_gamma=row.get("gamma", 0),
                    entry_vega=row.get("vega", 0),
                    entry_theta=row.get("theta", 0),
                )
            
            # Exit signal
            elif signal == 0 and current_position is not None:
                exit_price = row.get("exit_price", row.get("close", 0))
                commission = self.commission_per_contract * position_size * 2  # Exit
                slippage = self.calculate_slippage(exit_price, position_size)
                
                current_position.close(date, exit_price, commission, slippage)
                self.trades.append(current_position)
                self.current_capital += current_position.net_pnl
                current_position = None
            
            # Record equity
            self.equity_curve.append((date, self.current_capital))
        
        # Close any remaining position
        if current_position is not None:
            last_price = signals.iloc[-1].get("close", current_position.entry_price)
            current_position.close(
                end_date, last_price,
                self.commission_per_contract * position_size * 2,
                self.calculate_slippage(last_price, position_size)
            )
            self.trades.append(current_position)
            self.current_capital += current_position.net_pnl
        
        return self._calculate_results(strategy_name, start_date, end_date)
    
    def run_arbitrage_backtest(
        self,
        opportunities: List[Dict],
        market_data: pd.DataFrame,
        execution_delay: int = 0,
        position_size: int = 1
    ) -> BacktestResult:
        """
        Backtest arbitrage opportunities.
        
        Args:
            opportunities: List of arbitrage opportunities with timestamps
            market_data: Market data for execution
            execution_delay: Delay in seconds between signal and execution
            position_size: Contracts per trade
        
        Returns:
            BacktestResult
        """
        self.trades = []
        self.equity_curve = []
        self.current_capital = self.initial_capital
        
        if not opportunities:
            return self._empty_result("arbitrage")
        
        start_date = min(o["timestamp"] for o in opportunities)
        end_date = max(o["timestamp"] for o in opportunities)
        
        for opp in opportunities:
            timestamp = opp["timestamp"]
            expected_profit = opp.get("expected_profit", 0)
            confidence = opp.get("confidence", 0.5)
            
            # Simulate execution
            # Add some randomness based on confidence
            execution_success = np.random.random() < confidence
            
            if execution_success:
                # Realized profit is fraction of expected
                realized_factor = 0.5 + np.random.random() * 0.5  # 50-100% of expected
                gross_pnl = expected_profit * realized_factor
            else:
                # Failed execution, small loss from bid-ask
                gross_pnl = -opp.get("commission_cost", 5)
            
            commission = opp.get("commission_cost", self.commission_per_contract * 4)
            slippage = opp.get("slippage_estimate", 2)
            net_pnl = gross_pnl - commission - slippage
            
            trade = Trade(
                entry_time=timestamp,
                exit_time=timestamp + timedelta(minutes=5),  # Quick arbitrage
                symbol=opp.get("symbol", "UNKNOWN"),
                strategy=opp.get("type", "arbitrage"),
                direction="long",
                entry_price=expected_profit,
                exit_price=gross_pnl,
                quantity=position_size,
                gross_pnl=gross_pnl,
                commission=commission,
                slippage=slippage,
                net_pnl=net_pnl,
                is_winner=net_pnl > 0,
                hold_time_days=0.01,
                notes=[f"Confidence: {confidence:.2f}"]
            )
            
            self.trades.append(trade)
            self.current_capital += net_pnl
            self.equity_curve.append((timestamp, self.current_capital))
        
        return self._calculate_results("arbitrage", start_date, end_date)
    
    def walk_forward_optimization(
        self,
        data: pd.DataFrame,
        strategy_func: Callable,
        param_ranges: Dict[str, List],
        train_window: int = 126,
        test_window: int = 21,
        step_size: int = 21,
        optimization_metric: str = "sharpe"
    ) -> Dict[str, Any]:
        """
        Perform walk-forward optimization.
        
        Args:
            data: Full dataset
            strategy_func: Function that takes (data, params) -> signals
            param_ranges: Dict of parameter ranges to optimize
            train_window: Training window size (days)
            test_window: Testing window size (days)
            step_size: Step size for rolling (days)
            optimization_metric: Metric to optimize
        
        Returns:
            Walk-forward results with in-sample and out-of-sample performance
        """
        results = {
            "in_sample": [],
            "out_of_sample": [],
            "optimal_params": [],
            "timestamps": []
        }
        
        total_rows = len(data)
        start_idx = train_window
        
        while start_idx + test_window <= total_rows:
            # Training data
            train_start = start_idx - train_window
            train_end = start_idx
            train_data = data.iloc[train_start:train_end]
            
            # Testing data
            test_start = start_idx
            test_end = min(start_idx + test_window, total_rows)
            test_data = data.iloc[test_start:test_end]
            
            # Optimize on training data
            best_params = None
            best_metric = -np.inf
            
            # Grid search
            from itertools import product
            param_combinations = list(product(*param_ranges.values()))
            param_names = list(param_ranges.keys())
            
            for combo in param_combinations:
                params = dict(zip(param_names, combo))
                
                try:
                    signals = strategy_func(train_data, params)
                    result = self.run_backtest(signals, train_data, position_size=1)
                    
                    if optimization_metric == "sharpe":
                        metric = result.sharpe_ratio
                    elif optimization_metric == "sortino":
                        metric = result.sortino_ratio
                    elif optimization_metric == "profit_factor":
                        metric = result.profit_factor
                    else:
                        metric = result.total_return
                    
                    if metric > best_metric:
                        best_metric = metric
                        best_params = params
                except:
                    continue
            
            if best_params is None:
                best_params = dict(zip(param_names, param_combinations[0]))
            
            # Test on out-of-sample data
            try:
                train_signals = strategy_func(train_data, best_params)
                train_result = self.run_backtest(train_signals, train_data)
                
                test_signals = strategy_func(test_data, best_params)
                test_result = self.run_backtest(test_signals, test_data)
                
                results["in_sample"].append(train_result)
                results["out_of_sample"].append(test_result)
                results["optimal_params"].append(best_params)
                results["timestamps"].append(data.index[test_start])
            except:
                pass
            
            start_idx += step_size
        
        # Aggregate results
        if results["out_of_sample"]:
            oos_returns = [r.total_return for r in results["out_of_sample"]]
            results["aggregate"] = {
                "mean_oos_return": np.mean(oos_returns),
                "std_oos_return": np.std(oos_returns),
                "min_oos_return": np.min(oos_returns),
                "max_oos_return": np.max(oos_returns),
                "win_rate_periods": np.mean([1 if r > 0 else 0 for r in oos_returns]),
            }
        
        return results
    
    def _calculate_results(
        self,
        strategy_name: str,
        start_date: datetime,
        end_date: datetime
    ) -> BacktestResult:
        """Calculate comprehensive backtest results."""
        if not self.trades:
            return self._empty_result(strategy_name)
        
        # Trade statistics
        total_trades = len(self.trades)
        winning_trades = sum(1 for t in self.trades if t.is_winner)
        losing_trades = total_trades - winning_trades
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # P&L
        total_pnl = sum(t.gross_pnl for t in self.trades)
        total_commission = sum(t.commission for t in self.trades)
        total_slippage = sum(t.slippage for t in self.trades)
        net_pnl = sum(t.net_pnl for t in self.trades)
        
        # Equity curve
        equity_df = pd.DataFrame(self.equity_curve, columns=["date", "equity"])
        equity_df.set_index("date", inplace=True)
        equity_series = equity_df["equity"]
        
        # Returns
        returns = equity_series.pct_change().dropna()
        
        # Performance metrics
        total_return = (self.current_capital - self.initial_capital) / self.initial_capital
        
        days = (end_date - start_date).days
        years = days / 365 if days > 0 else 1
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else total_return
        
        # Volatility
        volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0
        
        # Sharpe ratio
        excess_return = annualized_return - self.risk_free_rate
        sharpe = excess_return / volatility if volatility > 0 else 0
        
        # Sortino ratio (downside deviation)
        negative_returns = returns[returns < 0]
        downside_vol = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 1 else volatility
        sortino = excess_return / downside_vol if downside_vol > 0 else 0
        
        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0
        
        # Drawdown duration
        in_drawdown = drawdown < 0
        drawdown_duration = 0
        current_duration = 0
        for is_dd in in_drawdown:
            if is_dd:
                current_duration += 1
                drawdown_duration = max(drawdown_duration, current_duration)
            else:
                current_duration = 0
        
        # Calmar ratio
        calmar = annualized_return / max_drawdown if max_drawdown > 0 else 0
        
        # VaR and CVaR
        var_95 = np.percentile(returns, 5) if len(returns) > 0 else 0
        cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else var_95
        
        # Trade quality metrics
        wins = [t.net_pnl for t in self.trades if t.is_winner]
        losses = [t.net_pnl for t in self.trades if not t.is_winner]
        
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        expectancy = (win_rate * avg_win + (1 - win_rate) * avg_loss) if total_trades > 0 else 0
        
        avg_hold_time = np.mean([t.hold_time_days for t in self.trades])
        
        return BacktestResult(
            strategy_name=strategy_name,
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
            final_capital=self.current_capital,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_pnl=total_pnl,
            total_commission=total_commission,
            total_slippage=total_slippage,
            net_pnl=net_pnl,
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_drawdown,
            max_drawdown_duration=drawdown_duration,
            calmar_ratio=calmar,
            volatility=volatility,
            var_95=var_95,
            cvar_95=cvar_95,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            expectancy=expectancy,
            avg_hold_time=avg_hold_time,
            trades=self.trades,
            equity_curve=equity_series,
            returns=returns
        )
    
    def _empty_result(self, strategy_name: str) -> BacktestResult:
        """Create empty result for no trades."""
        now = datetime.now()
        return BacktestResult(
            strategy_name=strategy_name,
            start_date=now,
            end_date=now,
            initial_capital=self.initial_capital,
            final_capital=self.initial_capital,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0,
            total_pnl=0,
            total_commission=0,
            total_slippage=0,
            net_pnl=0,
            total_return=0,
            annualized_return=0,
            sharpe_ratio=0,
            sortino_ratio=0,
            max_drawdown=0,
            max_drawdown_duration=0,
            calmar_ratio=0,
            volatility=0,
            var_95=0,
            cvar_95=0,
            avg_win=0,
            avg_loss=0,
            profit_factor=0,
            expectancy=0,
            avg_hold_time=0
        )
    
    def generate_performance_report(self, result: BacktestResult) -> str:
        """Generate human-readable performance report."""
        report = f"""
╔════════════════════════════════════════════════════════════════╗
║               BACKTEST PERFORMANCE REPORT                      ║
╠════════════════════════════════════════════════════════════════╣
║ Strategy: {result.strategy_name:<50} ║
║ Period: {result.start_date.strftime('%Y-%m-%d')} to {result.end_date.strftime('%Y-%m-%d'):<32} ║
╠════════════════════════════════════════════════════════════════╣
║                      TRADE STATISTICS                          ║
╠────────────────────────────────────────────────────────────────╣
║ Total Trades:      {result.total_trades:<10} Win Rate:       {result.win_rate*100:>6.1f}% ║
║ Winning Trades:    {result.winning_trades:<10} Losing Trades:  {result.losing_trades:>6}   ║
║ Avg Win:          ${result.avg_win:<10.2f} Avg Loss:      ${result.avg_loss:>8.2f}  ║
║ Profit Factor:     {result.profit_factor:<10.2f} Expectancy:    ${result.expectancy:>8.2f}  ║
╠════════════════════════════════════════════════════════════════╣
║                      P&L SUMMARY                               ║
╠────────────────────────────────────────────────────────────────╣
║ Initial Capital:  ${result.initial_capital:>12,.2f}                            ║
║ Final Capital:    ${result.final_capital:>12,.2f}                            ║
║ Net P&L:          ${result.net_pnl:>12,.2f}                            ║
║ Total Commission: ${result.total_commission:>12,.2f}                            ║
║ Total Slippage:   ${result.total_slippage:>12,.2f}                            ║
╠════════════════════════════════════════════════════════════════╣
║                   PERFORMANCE METRICS                          ║
╠────────────────────────────────────────────────────────────────╣
║ Total Return:       {result.total_return*100:>8.2f}%   Annualized:   {result.annualized_return*100:>8.2f}%  ║
║ Sharpe Ratio:       {result.sharpe_ratio:>8.2f}    Sortino:      {result.sortino_ratio:>8.2f}   ║
║ Max Drawdown:       {result.max_drawdown*100:>8.2f}%   Duration:     {result.max_drawdown_duration:>6} days  ║
║ Calmar Ratio:       {result.calmar_ratio:>8.2f}    Volatility:   {result.volatility*100:>8.2f}%  ║
╠════════════════════════════════════════════════════════════════╣
║                      RISK METRICS                              ║
╠────────────────────────────────────────────────────────────────╣
║ VaR (95%):          {result.var_95*100:>8.2f}%   CVaR (95%):   {result.cvar_95*100:>8.2f}%  ║
║ Avg Hold Time:      {result.avg_hold_time:>8.1f} days                           ║
╚════════════════════════════════════════════════════════════════╝
"""
        return report