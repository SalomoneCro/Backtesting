"""
Backtesting engine for trading strategies.

This module provides the core backtesting infrastructure to simulate
trading strategies and evaluate their performance.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Trade:
    """
    Represents a single trade execution.
    
    Attributes:
        date: Date of the trade
        signal: Trading signal (1=long, -1=short)
        price: Execution price
        position: Number of units (positive=long, negative=short)
        capital_before: Capital before the trade
        capital_after: Capital after the trade (next day)
        return_pct: Percentage return on this trade
    """
    date: pd.Timestamp
    signal: int
    price: float
    position: float
    capital_before: float
    capital_after: float
    return_pct: float


@dataclass
class BacktestResults:
    """
    Contains all results from a backtest run.
    
    Attributes:
        equity_curve: DataFrame with Date and Capital columns
        trades: List of all executed trades
        metrics: Dictionary of performance metrics
        initial_capital: Starting capital
        final_capital: Ending capital
    """
    equity_curve: pd.DataFrame
    trades: List[Trade]
    metrics: Dict[str, float]
    initial_capital: float
    final_capital: float


class Backtester:
    """
    Backtesting engine for evaluating trading strategies.
    
    The backtester simulates trading by:
    1. Reading historical price data
    2. Getting predictions from a predictor
    3. Executing trades according to a strategy
    4. Tracking capital and returns over time
    5. Computing performance metrics
    
    Features:
    - Supports long and short positions
    - Tracks cumulative returns
    - Records individual trade performance
    - Computes risk-adjusted metrics
    - Visualizes equity curves
    
    Attributes:
        data: Historical price data
        predictor: Predictor instance for generating signals
        strategy: Strategy instance for position sizing
        initial_capital: Starting capital in dollars
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        predictor,
        strategy,
        initial_capital: float = 1.0
    ):
        """
        Initialize the backtester.
        
        Args:
            data: DataFrame with columns ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'close_diff']
            predictor: Predictor instance (e.g., OraclePredictor)
            strategy: Strategy instance (e.g., AllInStrategy)
            initial_capital: Starting capital in dollars (default: $1.0)
        
        Raises:
            ValueError: If data is invalid or initial_capital <= 0
        """
        # Validate inputs
        required_columns = ['Date', 'Close', 'close_diff']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        if initial_capital <= 0:
            raise ValueError(f"Initial capital must be positive, got {initial_capital}")
        
        # Ensure Date is datetime and set as index
        if not pd.api.types.is_datetime64_any_dtype(data['Date']):
            data = data.copy()
            data['Date'] = pd.to_datetime(data['Date'])
        
        self.data = data.set_index('Date').sort_index()
        self.predictor = predictor
        self.strategy = strategy
        self.initial_capital = initial_capital
        
        # Results (populated after running backtest)
        self.results: Optional[BacktestResults] = None
    
    def run(self) -> BacktestResults:
        """
        Execute the backtest simulation.
        
        Simulates trading day-by-day:
        1. Get prediction signal
        2. Calculate position size
        3. Execute trade at close price
        4. Calculate returns
        5. Update capital
        
        Returns:
            BacktestResults: Complete backtest results with metrics
        """
        # Initialize tracking variables
        current_capital = self.initial_capital
        equity_history = []
        trades = []
        
        # Get all trading dates
        dates = self.data.index
        
        for i, date in enumerate(dates):
            # Get current price and signal
            current_price = self.data.loc[date, 'Close']
            signal = self.predictor.predict(date)
            
            # Record capital at start of day
            capital_before = current_capital
            
            # Calculate position size
            position = self.strategy.calculate_position(
                signal=signal,
                current_capital=current_capital,
                current_price=current_price
            )
            
            # Calculate return if we have a next day
            if i < len(dates) - 1:
                next_date = dates[i + 1]
                next_price = self.data.loc[next_date, 'Close']
                
                trade_return = self.strategy._get_return(current_price, next_price, position)
                
                # Update capital
                current_capital = capital_before * (1 + trade_return)
                
                # Record trade
                if signal != 0:  # Only record actual trades
                    trade = Trade(
                        date=date,
                        signal=signal,
                        price=current_price,
                        position=position,
                        capital_before=capital_before,
                        capital_after=current_capital,
                        return_pct=trade_return * 100
                    )
                    trades.append(trade)
            
            # Record equity
            equity_history.append({
                'Date': date,
                'Capital': current_capital
            })
        
        # Create equity curve DataFrame
        equity_curve = pd.DataFrame(equity_history)
        
        # Calculate metrics
        metrics = self._calculate_metrics(equity_curve, trades)
        
        # Store results
        self.results = BacktestResults(
            equity_curve=equity_curve,
            trades=trades,
            metrics=metrics,
            initial_capital=self.initial_capital,
            final_capital=current_capital
        )
        
        return self.results
    
    def _calculate_metrics(
        self,
        equity_curve: pd.DataFrame,
        trades: List[Trade]
    ) -> Dict[str, float]:
        """
        Calculate performance metrics.
        
        Args:
            equity_curve: DataFrame with capital over time
            trades: List of executed trades
            
        Returns:
            Dict with performance metrics
        """
        final_capital = equity_curve['Capital'].iloc[-1]
        initial_capital = equity_curve['Capital'].iloc[0]
        
        # Total return
        total_return = ((final_capital - initial_capital) / initial_capital) * 100
        
        # Number of trading days
        n_days = len(equity_curve)
        years = n_days / 252  # Assuming 252 trading days per year
        
        # Annualized return
        if years > 0:
            annualized_return = (((final_capital / initial_capital) ** (1 / years)) - 1) * 100
        else:
            annualized_return = 0.0
        
        # Calculate daily returns
        equity_curve = equity_curve.copy()
        equity_curve['Daily_Return'] = equity_curve['Capital'].pct_change()
        
        # Sharpe Ratio (assuming risk-free rate = 0)
        returns = equity_curve['Daily_Return'].dropna()
        if len(returns) > 0 and returns.std() > 0:
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
        
        # Maximum Drawdown
        cumulative = equity_curve['Capital']
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() * 100
        
        # Trade statistics
        if trades:
            winning_trades = [t for t in trades if t.return_pct > 0]
            losing_trades = [t for t in trades if t.return_pct < 0]
            
            win_rate = (len(winning_trades) / len(trades)) * 100 if trades else 0.0
            
            avg_win = np.mean([t.return_pct for t in winning_trades]) if winning_trades else 0.0
            avg_loss = np.mean([t.return_pct for t in losing_trades]) if losing_trades else 0.0
        else:
            win_rate = 0.0
            avg_win = 0.0
            avg_loss = 0.0
        
        return {
            'Total Return (%)': total_return,
            'Annualized Return (%)': annualized_return,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown (%)': max_drawdown,
            'Number of Trades': len(trades),
            'Win Rate (%)': win_rate,
            'Average Win (%)': avg_win,
            'Average Loss (%)': avg_loss,
            'Final Capital': final_capital,
            'Total Days': n_days
        }
    
    def plot_equity_curve(self, figsize: Tuple[int, int] = (12, 6)) -> None:
        """
        Plot the equity curve showing capital growth over time.
        
        Args:
            figsize: Figure size as (width, height)
        
        Raises:
            RuntimeError: If backtest hasn't been run yet
        """
        if self.results is None:
            raise RuntimeError("Must run backtest before plotting. Call run() first.")
        
        # Set seaborn style
        sns.set_style("whitegrid")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot equity curve
        sns.lineplot(
            data=self.results.equity_curve,
            x='Date',
            y='Capital',
            ax=ax,
            linewidth=2,
            color='#2E86AB'
        )
        
        # Formatting
        ax.set_title(
            f'Equity Curve - {self.strategy.get_name()}',
            fontsize=16,
            fontweight='bold',
            pad=20
        )
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Capital ($)', fontsize=12, fontweight='bold')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add metrics text box
        metrics_text = (
            f"Initial Capital: ${self.results.initial_capital:.2f}\n"
            f"Final Capital: ${self.results.final_capital:.2f}\n"
            f"Total Return: {self.results.metrics['Total Return (%)']:.2f}%\n"
            f"Sharpe Ratio: {self.results.metrics['Sharpe Ratio']:.2f}\n"
            f"Max Drawdown: {self.results.metrics['Max Drawdown (%)']:.2f}%"
        )
        
        ax.text(
            0.02, 0.98,
            metrics_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
        
        plt.tight_layout()
        plt.show()
    
    def print_summary(self) -> None:
        """
        Print a comprehensive summary of backtest results.
        
        Raises:
            RuntimeError: If backtest hasn't been run yet
        """
        if self.results is None:
            raise RuntimeError("Must run backtest before printing summary. Call run() first.")
        
        print("=" * 70)
        print(f"BACKTEST SUMMARY - {self.strategy.get_name()}")
        print("=" * 70)
        print()
        
        print("PERFORMANCE METRICS:")
        print("-" * 70)
        for metric, value in self.results.metrics.items():
            if 'Rate' in metric or 'Return' in metric or 'Drawdown' in metric or 'Win' in metric or 'Loss' in metric:
                print(f"  {metric:<30} {value:>10.2f}")
            elif 'Ratio' in metric:
                print(f"  {metric:<30} {value:>10.3f}")
            elif 'Capital' in metric:
                print(f"  {metric:<30} ${value:>10.2f}")
            else:
                print(f"  {metric:<30} {value:>10.0f}")
        
        print()
        print("=" * 70)