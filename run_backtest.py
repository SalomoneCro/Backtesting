"""
Example script demonstrating how to run a backtest.

This script shows how to:
1. Load historical data
2. Create a predictor
3. Define a strategy
4. Run the backtest
5. Analyze results
"""

import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path to import backtesting modules
sys.path.append(str(Path(__file__).parent.parent))

from Backtesting.predictors import OraclePredictor
from Backtesting.strategies import AllInStrategy, FixedPercentageStrategy, StochasticErrorStrategy
from Backtesting.backtester import Backtester


def main():
    """Run a complete backtest example."""
    
    # 1. Load data
    print("Loading data...")
    data_path = Path(__file__).parent / "ExploratoryAnalysis" / "Datasets" / "USDCHFX_2018-01-02_2022-01-01.csv"
    # data_path = Path(__file__).parent / "ExploratoryAnalysis" / "Datasets" / "AAPL_2018-01-03_2022-01-01.csv"
    data = pd.read_csv(data_path)
    print(f"✓ Loaded {len(data)} rows of data")

    # 2. Create predictor (perfect oracle)
    print("Creating predictor...")
    predictor = OraclePredictor(data)
    signal_dist = predictor.get_signal_distribution()
    print(f"✓ Predictor ready")
    print(f"  Signal distribution: UP={signal_dist['up']}, DOWN={signal_dist['down']}, UNCERTAIN={signal_dist['uncertain']}")
    print()
    
    # 3. Define strategies to test
    strategies = [
        # AllInStrategy(),
        # FixedPercentageStrategy(0.15),
        StochasticErrorStrategy(error_rate=0.5, percentage=0.7)
    ]
    
    # 4. Run backtests for each strategy
    initial_capital = 1.0  # Start with $1
    
    for strategy in strategies:
        print("=" * 70)
        print(f"Testing: {strategy.get_name()}")
        print("=" * 70)
        
        # Create backtester
        backtester = Backtester(
            data=data,
            predictor=predictor,
            strategy=strategy,
            initial_capital=initial_capital
        )
        
        # Run backtest
        print("Running backtest...")
        results = backtester.run()
        print(f"✓ Backtest complete")
        print()
        
        # Print summary
        backtester.print_summary()
        print()
        
        # Plot equity curve
        print("Generating equity curve plot...")
        backtester.plot_equity_curve()
        print()
    
    print("All backtests completed!")


if __name__ == "__main__":
    main()