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
    Machine learning strategy with configurable position sizing.
    
    This strategy uses pre-computed predictions to make trading decisions.
    Predictions should be a dictionary mapping dates to signals:
    - 1: Go long (buy)
    - -1: Go short (sell)
    - 0: No position (close any open positions)
    
    Position sizing is controlled by the 'percentage' parameter.
    
    Class Attributes:
        predictions: Dict mapping dates (pd.Timestamp) to signals (int)
                     Must be set before running backtest
        percentage: Fraction of equity to deploy (0.0 to 1.0)
                   Default is 0.99999 (effectively all-in, but avoids backtesting.py bug)
    
    Usage:
        # All-in strategy (default)
        MLStrategy.predictions = predictor.predictions
        bt = Backtest(data, MLStrategy, ...)
        
        # Fixed percentage (e.g., 50%)
        MLStrategy.predictions = predictor.predictions
        MLStrategy.percentage = 0.5
        bt = Backtest(data, MLStrategy, ...)
        
        # Or create subclasses for different percentages
        class MLStrategy50(MLStrategy):
            percentage = 0.5
        
        MLStrategy50.predictions = predictor.predictions
        bt = Backtest(data, MLStrategy50, ...)
    """
    
    # Class variables (set before running backtest)
    predictions: Optional[Dict[pd.Timestamp, int]] = None
    percentage: float = 0.99999  # Default: all-in (0.99999 to avoid backtesting.py bug with 1.0)
    
    def init(self):
        """
        Initialize the strategy and validate parameters.
        
        Raises:
            ValueError: If predictions are not set or percentage is invalid
        """
        if self.predictions is None:
            raise ValueError(
                "Predictions must be set before running backtest. "
                "Use: Strategy.predictions = predictor.predictions"
            )
        
        if not 0.0 < self.percentage <= 1.0:
            raise ValueError(
                f"Percentage must be between 0 and 1 (exclusive of 0), got {self.percentage}"
            )
    
    def next(self):
        """
        Execute strategy logic for current bar.
        
        Handles position management based on prediction signals and
        configured position sizing.
        """
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
    
    def get_current_signal(self) -> int:
        """
        Get the trading signal for the current date.
        
        Returns:
            int: Signal (1=long, -1=short, 0=no position)
                 Returns 0 if no prediction available for current date
        """
        current_date = self.data.index[-1]
        return self.predictions.get(current_date, 0)


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
        MLStrategy.predictions = predictions
    """
    if hasattr(predictor, 'predictions'):
        return predictor.predictions
    else:
        raise AttributeError(
            f"Predictor {type(predictor).__name__} does not have 'predictions' attribute"
        )