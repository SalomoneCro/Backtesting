"""
Trading strategies compatible with backtesting.py library.

These strategies integrate with the backtesting.py framework while using
our custom predictors for signal generation.
"""

from backtesting import Strategy
import pandas as pd
from typing import Dict, Optional


class MLStrategy(Strategy):
    """
    Base machine learning strategy for backtesting.py.
    
    This strategy uses pre-computed predictions to make trading decisions.
    Predictions should be a dictionary mapping dates to signals:
    - 1: Go long (buy)
    - -1: Go short (sell)
    - 0: No position (close any open positions)
    
    Subclasses should implement position sizing logic.
    
    Class Attributes:
        predictions: Dict mapping dates (pd.Timestamp) to signals (int)
                     Must be set before running backtest
    """
    
    # Class variable to store predictions (set before running backtest)
    predictions: Optional[Dict[pd.Timestamp, int]] = None
    
    def init(self):
        """
        Initialize the strategy.
        
        Validates that predictions have been loaded.
        
        Raises:
            ValueError: If predictions are not set
        """
        if self.predictions is None:
            raise ValueError(
                "Predictions must be set before running backtest. "
                "Use: Strategy.predictions = predictor.predictions"
            )
    
    def next(self):
        """
        Execute strategy logic for current bar.
        
        This method is called for each bar in the backtest.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement next()")
    
    def get_current_signal(self) -> int:
        """
        Get the trading signal for the current date.
        
        Returns:
            int: Signal (1=long, -1=short, 0=no position)
                 Returns 0 if no prediction available for current date
        """
        current_date = self.data.index[-1]
        return self.predictions.get(current_date, 0)


class AllInMLStrategy(MLStrategy):
    """
    All-in strategy using ML predictions.
    
    Position sizing:
    - Signal = 1: Go 100% long
    - Signal = -1: Go 100% short  
    - Signal = 0: Close all positions
    
    This is an aggressive strategy that deploys all capital on each trade.
    Suitable for baseline testing with perfect predictions.
    
    Usage:
        AllInMLStrategy.predictions = predictor.predictions
        bt = Backtest(data, AllInMLStrategy, ...)
        stats = bt.run()
    """
    
    def next(self):
        """Execute all-in strategy logic."""
        signal = self.get_current_signal()
        
        if signal == 1:  # Long signal
            # Close any short position and go long
            if self.position.is_short:
                self.position.close()
            if not self.position.is_long:
                self.buy(size=1.0)  # 100% of equity
        
        elif signal == -1:  # Short signal
            # Close any long position and go short
            if self.position.is_long:
                self.position.close()
            if not self.position.is_short:
                self.sell(size=1.0)  # 100% of equity
        
        elif signal == 0:  # No position
            # Close any open position
            if self.position:
                self.position.close()


class FixedPercentageMLStrategy(MLStrategy):
    """
    Fixed percentage strategy using ML predictions.
    
    Deploys a fixed percentage of equity on each trade, keeping the rest in cash.
    This provides risk management by limiting exposure.
    
    Position sizing:
    - Signal = 1: Go long with X% of equity
    - Signal = -1: Go short with X% of equity
    - Signal = 0: Close all positions
    
    Class Attributes:
        percentage: Fraction of equity to deploy (0.0 to 1.0)
    
    Usage:
        FixedPercentageMLStrategy.predictions = predictor.predictions
        FixedPercentageMLStrategy.percentage = 0.5  # 50%
        bt = Backtest(data, FixedPercentageMLStrategy, ...)
        stats = bt.run()
    """
    
    # Class variable for percentage (set before running backtest)
    percentage: float = 0.5
    
    def init(self):
        """
        Initialize and validate percentage parameter.
        
        Raises:
            ValueError: If percentage is not between 0 and 1
        """
        super().init()
        
        if not 0.0 <= self.percentage <= 1.0:
            raise ValueError(
                f"Percentage must be between 0 and 1, got {self.percentage}"
            )
    
    def next(self):
        """Execute fixed percentage strategy logic."""
        signal = self.get_current_signal()
        
        if signal == 1:  # Long signal
            # Close any short position and go long
            if self.position.is_short:
                self.position.close()
            if not self.position.is_long:
                self.buy(size=self.percentage)
        
        elif signal == -1:  # Short signal
            # Close any long position and go short
            if self.position.is_long:
                self.position.close()
            if not self.position.is_short:
                self.sell(size=self.percentage)
        
        elif signal == 0:  # No position
            # Close any open position
            if self.position:
                self.position.close()


def load_predictions_from_predictor(predictor) -> Dict[pd.Timestamp, int]:
    """
    Helper function to extract predictions from a predictor.
    
    Args:
        predictor: Predictor instance (e.g., OraclePredictor)
    
    Returns:
        Dict mapping dates to signals
    
    Example:
        predictor = OraclePredictor(data)
        predictions = load_predictions_from_predictor(predictor)
        AllInMLStrategy.predictions = predictions
    """
    if hasattr(predictor, 'predictions'):
        return predictor.predictions
    else:
        raise AttributeError(
            f"Predictor {type(predictor).__name__} does not have 'predictions' attribute"
        )