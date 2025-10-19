"""
Predictor classes for backtesting trading strategies.

This module contains predictor classes that generate trading signals based on
price data. Predictors can be perfect oracles (for baseline testing) or actual
ML models (for real strategy evaluation).
"""

from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict


class Predictor(ABC):
    """
    Abstract base class for all predictors.
    
    A predictor takes historical data and generates trading signals:
    - 1: Predict price will go UP (go long)
    - -1: Predict price will go DOWN (go short)
    - 0: UNCERTAIN (no position)
    """
    
    @abstractmethod
    def predict(self, date: pd.Timestamp) -> int:
        """
        Generate a trading signal for a given date.
        
        Args:
            date: The date to generate a prediction for
            
        Returns:
            int: Trading signal (1=up, -1=down, 0=uncertain)
        """
        pass
    
    @abstractmethod
    def get_available_dates(self) -> pd.DatetimeIndex:
        """
        Get all dates for which predictions can be made.
        
        Returns:
            pd.DatetimeIndex: Available trading dates
        """
        pass


class OraclePredictor(Predictor):
    """
    Perfect predictor that "knows" the future price movements.
    
    This predictor cheats by looking at the actual close_diff values
    to determine if prices went up or down. It's used as a baseline
    to measure the theoretical maximum performance of a strategy.
    
    Prediction logic:
    - If close_diff > 0 (price increased) → predict UP (1)
    - If close_diff < 0 (price decreased) → predict DOWN (-1)
    - If close_diff == 0 (no change) → predict UNCERTAIN (0)
    
    Attributes:
        data: DataFrame containing historical price data with close_diff
        predictions: Dictionary mapping dates to signals
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the oracle predictor.
        
        Args:
            data: DataFrame with columns ['Date', 'Close', 'close_diff']
                  Date should be datetime type
                  close_diff is the difference: Close[t] - Close[t-1]
        
        Raises:
            ValueError: If required columns are missing
        """
        required_columns = ['Date', 'Close', 'close_diff']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Ensure Date is datetime type
        if not pd.api.types.is_datetime64_any_dtype(data['Date']):
            data = data.copy()
            data['Date'] = pd.to_datetime(data['Date'])
        
        self.data = data.set_index('Date').sort_index()
        
        # Pre-compute all predictions
        self.predictions = self._generate_predictions()
    
    def _generate_predictions(self) -> Dict[pd.Timestamp, int]:
        """
        Generate predictions for all dates based on close_diff.
        
        Returns:
            Dict mapping dates to signals (1, -1, or 0)
        """

        predictions = {}
        dates = self.data.index.tolist()
        
        for i in range(len(dates) - 1):  # Stop one before end
            current_date = dates[i]
            next_date = dates[i + 1]
            
            # Look at NEXT day's close_diff to know what will happen
            next_close_diff = self.data.loc[next_date, 'close_diff']
            
            if next_close_diff > 0:
                signal = 1  # Tomorrow will go UP
            elif next_close_diff < 0:
                signal = -1  # Tomorrow will go DOWN
            else:
                signal = 0  # No change
            
            predictions[current_date] = signal
        
        ## This is to have an action on the last day that have no prediction.
        predictions[dates[-1]] = 0
        
        # Last day has no next day, so no prediction
        # (This is handled automatically by not including it)
        
        return predictions
    
    def predict(self, date: pd.Timestamp) -> int:
        """
        Get the prediction for a specific date.
        
        Args:
            date: The date to predict for
            
        Returns:
            int: Trading signal (1=up, -1=down, 0=uncertain)
            
        Raises:
            KeyError: If date is not in the dataset
        """
        if date not in self.predictions:
            raise KeyError(f"No prediction available for date {date}")
        
        return self.predictions[date]
    
    def get_available_dates(self) -> pd.DatetimeIndex:
        """
        Get all dates for which predictions are available.
        
        Returns:
            pd.DatetimeIndex: All available trading dates
        """
        return self.data.index
    
    def get_signal_distribution(self) -> Dict[str, int]:
        """
        Get the distribution of signals across all dates.
        
        Returns:
            Dict with counts for 'up', 'down', and 'uncertain' signals
        """
        up_count = sum(1 for signal in self.predictions.values() if signal == 1)
        down_count = sum(1 for signal in self.predictions.values() if signal == -1)
        uncertain_count = sum(1 for signal in self.predictions.values() if signal == 0)
        
        return {
            'up': up_count,
            'down': down_count,
            'uncertain': uncertain_count,
            'total': len(self.predictions)
        }