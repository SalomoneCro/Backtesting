"""
Run backtests using the backtesting.py library.

This script demonstrates how to use backtesting.py with our custom
ML predictors and strategies.
"""

import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from backtesting import Backtest
from predictors import OraclePredictor
from strategies import (
    MLStrategy,
    load_predictions_from_predictor
)


def prepare_data(csv_path: Path) -> pd.DataFrame:
    """
    Load and prepare data for backtesting.py.
    
    backtesting.py requires specific column names:
    - Date (as index)
    - Open
    - High  
    - Low
    - Close
    - Volume
    
    Args:
        csv_path: Path to CSV file with historical data
    
    Returns:
        pd.DataFrame: Prepared data with proper format
    """
    data = pd.read_csv(csv_path)
    
    # Ensure Date is datetime
    data['Date'] = pd.to_datetime(data['Date'])
    
    # Set Date as index
    data = data.set_index('Date')
    
    # backtesting.py needs these columns (case-sensitive)
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    # Check if all required columns exist
    missing = [col for col in required_columns if col not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns for backtesting.py: {missing}")
    
    return data[required_columns]


def run_single_backtest(
    data: pd.DataFrame,
    strategy_name: str,
    predictions: dict,
    initial_cash: float = 10000.0,
    commission: float = 0.0002,
    **strategy_params
):
    """
    Run a single backtest with given parameters.
    
    Args:
        data: Historical OHLCV data
        strategy_name: Name for display
        predictions: Dict of predictions from predictor
        initial_cash: Starting capital
        commission: Commission per trade (0.0002 = 0.02% = 2 basis points)
        **strategy_params: Additional parameters for strategy (e.g., percentage=0.5)
    
    Returns:
        Backtest stats object
    """
    print("=" * 70)
    print(f"Running: {strategy_name}")
    print("=" * 70)
    
    # Create a NEW subclass for this specific backtest
    # This avoids modifying the base class
    class CustomMLStrategy(MLStrategy):
        pass
    
    # Set predictions and parameters on the NEW class
    CustomMLStrategy.predictions = predictions
    for param, value in strategy_params.items():
        setattr(CustomMLStrategy, param, value)
    
    # Create and run backtest
    bt = Backtest(
        data,
        CustomMLStrategy,  # ← Use the custom class
        cash=initial_cash,
        commission=commission,
        exclusive_orders=True,
        trade_on_close=True,
        finalize_trades=True
    )
    
    stats = bt.run()
    
    # Print results
    print("\nPerformance Metrics:")
    print("-" * 70)
    print(stats)
    print()
    
    # Plot
    print("Generating interactive plot...")
    bt.plot(plot_equity=False, plot_return = True)
    print()
    
    return stats


def main():
    """Run complete backtest analysis."""
    
    # 1. Load data
    print("Loading data...")
    data_path = Path(__file__).parent.parent / "ExploratoryAnalysis" / "Datasets" / "USDCHFX_2018-01-02_2022-01-01.csv"
    
    # Read raw data for predictor (needs close_diff)
    raw_data = pd.read_csv(data_path)
    
    # Prepare data for backtesting.py (needs OHLCV)
    bt_data = prepare_data(data_path)
    
    print(f"✓ Loaded {len(bt_data)} rows of data")
    print(f"  Date range: {bt_data.index[0]} to {bt_data.index[-1]}")
    print()
    
    # 2. Create predictor and generate predictions
    print("Creating oracle predictor...")
    predictor = OraclePredictor(raw_data)
    predictions = load_predictions_from_predictor(predictor)
    
    signal_dist = predictor.get_signal_distribution()
    print(f"✓ Predictor ready")
    print(f"  Signals: UP={signal_dist['up']}, DOWN={signal_dist['down']}, UNCERTAIN={signal_dist['uncertain']}")
    print()
    
    # 3. Define test configurations
    initial_cash = 10000.0  # Start with $10,000
    commission = 0.0002  # 0.02% = 2 basis points (typical for forex)
    
    tests = [
        {
            'strategy_name': 'All-In Strategy (100%)',
            'params': {'percentage': 0.9}
        },
        # {
        #     'strategy_name': 'Fixed Percentage Strategy (50%)',
        #     'params': {'percentage': 0.5}
        # },
    ]
    
    # 4. Run all backtests
    results = []
    
    for test in tests:
        stats = run_single_backtest(
            data=bt_data,
            strategy_name=test['strategy_name'],
            predictions=predictions,
            initial_cash=initial_cash,
            commission=commission,
            **test['params']
        )
        
        results.append({
            'name': test['strategy_name'],
            'stats': stats
        })
    
    # 5. Summary comparison
    print("=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print()
    
    comparison_metrics = [
        'Return [%]',
        'Sharpe Ratio',
        'Max. Drawdown [%]',
        'Win Rate [%]',
        '# Trades'
    ]
    
    for metric in comparison_metrics:
        print(f"{metric}:")
        for result in results:
            value = result['stats'][metric]
            print(f"  {result['name']:<40} {value:>10}")
        print()
    
    print("All backtests completed!")

    print("DEBUG INFO:")
    print(f"  Initial Cash: ${initial_cash:.2f}")
    print(f"  Equity Final: ${stats['Equity Final [$]']:.2f}")
    print(f"  Return [%]: {stats['Return [%]']:.2f}")
    print(f"  Manual calculation: {((stats['Equity Final [$]'] - initial_cash) / initial_cash * 100):.2f}%")
    print()
    print(f"  Equity Final: ${stats['Equity Final [$]']:.2f}")
    print(f"  Return [%]: {stats['Return [%]']:.2f}")
    print(f"  Manual calc: {((stats['Equity Final [$]'] - initial_cash) / initial_cash * 100):.2f}%")
if __name__ == "__main__":
    main()