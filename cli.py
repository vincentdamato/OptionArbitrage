#!/usr/bin/env python3
"""
Command Line Interface for Options Arbitrage Scanner.
"""

import argparse
import sys
import json
from datetime import datetime
from typing import Optional
import os

from loguru import logger


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = "DEBUG" if verbose else "INFO"
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        level=level,
        colorize=True
    )


def cmd_scan(args):
    """Run arbitrage scan."""
    from core.orchestrator import create_orchestrator
    
    logger.info(f"Starting scan for symbols: {args.symbols}")
    
    orchestrator = create_orchestrator(
        api_key=args.api_key,
        api_secret=args.api_secret
    )
    
    if args.symbols:
        orchestrator.update_watchlist(args.symbols)
    
    results = orchestrator.scan_watchlist(
        min_dte=args.min_dte,
        max_dte=args.max_dte,
        parallel=not args.sequential
    )
    
    # Display results
    total_opps = sum(len(r.opportunities) for r in results)
    print(f"\n{'='*60}")
    print(f"SCAN COMPLETE - {total_opps} opportunities found")
    print(f"{'='*60}\n")
    
    top_opps = orchestrator.get_top_opportunities(
        n=args.top_n,
        risk_free_only=args.risk_free_only,
        min_confidence=args.min_confidence
    )
    
    for i, opp in enumerate(top_opps, 1):
        risk_label = "🟢 RISK-FREE" if opp.risk_free else "🟡 STATISTICAL"
        print(f"{i}. {opp.symbol} - {opp.type.value}")
        print(f"   {risk_label} | Confidence: {opp.confidence*100:.0f}%")
        print(f"   Expected Profit: ${opp.expected_profit:.2f} ({opp.expected_profit_pct:.2f}%)")
        print(f"   Net Delta: {opp.net_delta:.4f} | DTE: {opp.days_to_expiration}")
        if opp.notes:
            print(f"   Notes: {'; '.join(opp.notes)}")
        print()
    
    if args.output:
        export_data = orchestrator.export_opportunities(
            format=args.output_format
        )
        
        output_file = f"opportunities_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{args.output_format}"
        with open(output_file, 'w') as f:
            f.write(export_data)
        logger.info(f"Results exported to {output_file}")


def cmd_analyze(args):
    """Analyze a specific opportunity or symbol."""
    from core.orchestrator import create_orchestrator
    from analysis.statistical_analyzer import StatisticalAnalyzer
    
    orchestrator = create_orchestrator(
        api_key=args.api_key,
        api_secret=args.api_secret
    )
    
    # Scan the symbol
    result = orchestrator.scan_symbol(args.symbol, include_analysis=True)
    
    print(f"\n{'='*60}")
    print(f"ANALYSIS: {args.symbol}")
    print(f"{'='*60}\n")
    
    # Print chain summary
    if result.chain_data:
        chain = result.chain_data
        print(f"Underlying: ${chain.underlying.mark:.2f}")
        print(f"Expirations: {len(chain.expirations)}")
        print(f"Strikes: {len(chain.strikes)}")
        print()
    
    # Print volatility analysis
    if result.volatility_metrics:
        print("Volatility Analysis:")
        for key, value in result.volatility_metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        print()
    
    # Print opportunities
    print(f"Opportunities Found: {len(result.opportunities)}")
    for i, opp in enumerate(result.opportunities[:5], 1):
        print(f"\n{i}. {opp.type.value}")
        print(f"   Expected: ${opp.expected_profit:.2f}")
        print(f"   Confidence: {opp.confidence*100:.0f}%")
        
        if args.detailed:
            analysis = orchestrator.analyze_opportunity(opp)
            mc = analysis["monte_carlo"]
            print(f"   Monte Carlo P(profit): {mc['prob_profit']*100:.1f}%")
            print(f"   VaR (95%): ${mc['var_95']:.2f}")
            print(f"   Recommendation: {analysis['recommendation']}")


def cmd_backtest(args):
    """Run backtest on historical data."""
    from backtesting.backtest_engine import BacktestEngine
    
    logger.info("Starting backtest...")
    
    engine = BacktestEngine(
        initial_capital=args.capital,
        commission_per_contract=args.commission
    )
    
    # For demo, create mock opportunities
    import numpy as np
    from datetime import timedelta
    
    opportunities = []
    start = datetime.strptime(args.start_date, "%Y-%m-%d")
    end = datetime.strptime(args.end_date, "%Y-%m-%d")
    
    current = start
    while current <= end:
        # Simulate finding 2-5 opportunities per day
        num_opps = np.random.randint(2, 6)
        for _ in range(num_opps):
            opportunities.append({
                "timestamp": current,
                "symbol": np.random.choice(["SPY", "QQQ", "AAPL", "NVDA"]),
                "type": np.random.choice(["put_call_parity", "box_spread", "butterfly"]),
                "expected_profit": np.random.uniform(20, 200),
                "confidence": np.random.uniform(0.6, 0.95),
                "commission_cost": 2.60,
                "slippage_estimate": 5.0
            })
        current += timedelta(days=1)
    
    result = engine.run_arbitrage_backtest(
        opportunities=opportunities,
        market_data=None
    )
    
    print(engine.generate_performance_report(result))
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(engine.generate_performance_report(result))
        logger.info(f"Report saved to {args.output}")


def cmd_watch(args):
    """Start continuous monitoring."""
    from core.orchestrator import create_orchestrator
    import time
    
    orchestrator = create_orchestrator(
        api_key=args.api_key,
        api_secret=args.api_secret
    )
    
    if args.symbols:
        orchestrator.update_watchlist(args.symbols)
    
    def on_opportunity(opp):
        """Callback for new opportunities."""
        if opp.expected_profit >= args.min_edge:
            print(f"\n🔔 NEW OPPORTUNITY: {opp.symbol} - {opp.type.value}")
            print(f"   Expected: ${opp.expected_profit:.2f} | Confidence: {opp.confidence*100:.0f}%")
    
    orchestrator.register_opportunity_callback(on_opportunity)
    
    print(f"Starting continuous monitoring (interval: {args.interval}s)")
    print("Press Ctrl+C to stop\n")
    
    try:
        orchestrator.start_continuous_scanning(interval_seconds=args.interval)
        
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping...")
        orchestrator.stop_continuous_scanning()
        orchestrator.shutdown()


def cmd_server(args):
    """Start Streamlit dashboard server."""
    import subprocess
    
    logger.info(f"Starting Streamlit server on port {args.port}")
    
    cmd = [
        "streamlit", "run", "app.py",
        "--server.port", str(args.port),
        "--server.headless", "true" if args.headless else "false"
    ]
    
    subprocess.run(cmd)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Options Arbitrage Scanner CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s scan -s SPY QQQ AAPL           Scan specific symbols
  %(prog)s scan --risk-free-only          Only show risk-free opportunities
  %(prog)s analyze SPY                     Analyze single symbol
  %(prog)s backtest -s 2024-01-01 -e 2024-12-01   Run backtest
  %(prog)s watch -i 30                     Continuous monitoring
  %(prog)s server                          Start dashboard
        """
    )
    
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--api-key", default=os.getenv("SCHWAB_API_KEY"), help="Schwab API key")
    parser.add_argument("--api-secret", default=os.getenv("SCHWAB_APP_SECRET"), help="Schwab API secret")
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Scan command
    scan_parser = subparsers.add_parser("scan", help="Scan for arbitrage opportunities")
    scan_parser.add_argument("-s", "--symbols", nargs="+", help="Symbols to scan")
    scan_parser.add_argument("--min-dte", type=int, default=1, help="Minimum days to expiration")
    scan_parser.add_argument("--max-dte", type=int, default=60, help="Maximum days to expiration")
    scan_parser.add_argument("--min-confidence", type=float, default=0.0, help="Minimum confidence")
    scan_parser.add_argument("--risk-free-only", action="store_true", help="Only risk-free opportunities")
    scan_parser.add_argument("--top-n", type=int, default=10, help="Number of top opportunities")
    scan_parser.add_argument("--sequential", action="store_true", help="Sequential (not parallel) scanning")
    scan_parser.add_argument("-o", "--output", action="store_true", help="Export results")
    scan_parser.add_argument("--output-format", choices=["json", "csv"], default="json")
    scan_parser.set_defaults(func=cmd_scan)
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze symbol or opportunity")
    analyze_parser.add_argument("symbol", help="Symbol to analyze")
    analyze_parser.add_argument("--detailed", action="store_true", help="Include Monte Carlo analysis")
    analyze_parser.set_defaults(func=cmd_analyze)
    
    # Backtest command
    backtest_parser = subparsers.add_parser("backtest", help="Run backtest")
    backtest_parser.add_argument("-s", "--start-date", required=True, help="Start date (YYYY-MM-DD)")
    backtest_parser.add_argument("-e", "--end-date", required=True, help="End date (YYYY-MM-DD)")
    backtest_parser.add_argument("--capital", type=float, default=100000, help="Initial capital")
    backtest_parser.add_argument("--commission", type=float, default=0.65, help="Commission per contract")
    backtest_parser.add_argument("-o", "--output", help="Output file for report")
    backtest_parser.set_defaults(func=cmd_backtest)
    
    # Watch command
    watch_parser = subparsers.add_parser("watch", help="Continuous monitoring")
    watch_parser.add_argument("-s", "--symbols", nargs="+", help="Symbols to watch")
    watch_parser.add_argument("-i", "--interval", type=int, default=60, help="Scan interval in seconds")
    watch_parser.add_argument("--min-edge", type=float, default=50, help="Minimum edge to alert ($)")
    watch_parser.set_defaults(func=cmd_watch)
    
    # Server command
    server_parser = subparsers.add_parser("server", help="Start dashboard server")
    server_parser.add_argument("-p", "--port", type=int, default=8501, help="Server port")
    server_parser.add_argument("--headless", action="store_true", help="Run headless")
    server_parser.set_defaults(func=cmd_server)
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    args.func(args)


if __name__ == "__main__":
    main()