"""
Trading strategy classes for backtesting.

This module defines the position sizing and trade execution logic.
Strategies determine how much capital to deploy based on trading signals.
"""

from abc import ABC, abstractmethod
from typing import Optional
import random

class Strategy(ABC):
    """
    Abstract base class for all trading strategies.
    
    A strategy determines position sizing based on:
    - Trading signal (1=long, -1=short, 0=no position)
    - Current capital
    - Current price
    
    The strategy is stateless - it doesn't track positions internally.
    Position tracking is handled by the Backtester.
    """
    
    @abstractmethod
    def calculate_position(
        self,
        signal: int,
        current_capital: float,
        current_price: float
    ) -> float:
        """
        Calculate the position size to take.
        
        Args:
            signal: Trading signal (1=long, -1=short, 0=no position)
            current_capital: Available capital in dollars
            current_price: Current asset price
            
        Returns:
            float: Number of units to hold
                   Positive = long position
                   Negative = short position
                   Zero = no position (cash)
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """
        Get the name of the strategy.
        
        Returns:
            str: Human-readable strategy name
        """
        pass


class AllInStrategy(Strategy):
    """
    All-in strategy: Deploy 100% of capital on every trade.
    
    Position sizing:
    - Signal = 1 (UP): Go long with all capital
    - Signal = -1 (DOWN): Go short with all capital
    - Signal = 0 (UNCERTAIN): Stay in cash (no position)
    
    This is an aggressive strategy with maximum exposure and no risk management.
    Suitable for baseline testing with perfect predictions.
    
    Example:
        If capital = $1000 and price = $50:
        - Long signal: Buy 20 units ($1000 / $50)
        - Short signal: Short 20 units (-$1000 / $50)
        - No signal: Hold 0 units
    """
    
    def __init__(self):
        """Initialize the all-in strategy."""
        pass
    
    def calculate_position(
        self,
        signal: int,
        current_capital: float,
        current_price: float
    ) -> float:
        """
        Calculate position size using all available capital.
        
        Args:
            signal: Trading signal (1, -1, or 0)
            current_capital: Total capital available
            current_price: Current price of the asset
            
        Returns:
            float: Number of units to hold
                   Positive for long, negative for short, zero for cash
        """
        if signal == 0:
            # No position - stay in cash
            return 0.0

        if current_price <= 0:
            raise ValueError(f"Invalid price: {current_price}. Price must be positive.")
        
        if current_capital <= 0:
            # No capital available - can't take position
            return 0.0
        
        # Calculate number of units we can buy/short with all capital
        units = current_capital / current_price
        
        # Apply signal direction
        position = signal * units
        
        return position

    def _get_return(self, current_price, next_price, position) -> float:
        # Calculate return based on position type
        if position > 0:  # Long position
            trade_return = (next_price - current_price) / current_price
        elif position < 0:  # Short position
            trade_return = (current_price - next_price) / current_price
        else:  # No position
            trade_return = 0.0
        return trade_return

    def get_name(self) -> str:
        """Get strategy name."""
        return "All-In Strategy"


class FixedPercentageStrategy(Strategy):
    """
    Fixed percentage strategy: Deploy a fixed percentage of capital per trade.
    
    This strategy provides risk management by limiting exposure on each trade.
    Useful for comparing against the aggressive all-in approach.
    
    Position sizing:
    - Signal = 1 (UP): Go long with (percentage)% of capital
    - Signal = -1 (DOWN): Go short with (percentage)% of capital
    - Signal = 0 (UNCERTAIN): Stay in cash
    
    Example:
        With percentage=0.5 (50%), capital=$1000, price=$50:
        - Long signal: Buy 10 units ($500 / $50)
        - Short signal: Short 10 units (-$500 / $50)
    
    Attributes:
        percentage: Fraction of capital to use (0.0 to 1.0)
    """
    
    def __init__(self, percentage: float = 0.5):
        """
        Initialize the fixed percentage strategy.
        
        Args:
            percentage: Fraction of capital to deploy (0.0 to 1.0)
                        Default is 0.5 (50%)
        
        Raises:
            ValueError: If percentage is not between 0 and 1
        """
        if not 0.0 <= percentage <= 1.0:
            raise ValueError(f"Percentage must be between 0 and 1, got {percentage}")
        
        self.percentage = percentage
    
    def calculate_position(
        self,
        signal: int,
        current_capital: float,
        current_price: float
    ) -> float:
        """
        Calculate position size using a fixed percentage of capital.
        
        Args:
            signal: Trading signal (1, -1, or 0)
            current_capital: Total capital available
            current_price: Current price of the asset
            
        Returns:
            float: Number of units to hold
        """
        if signal == 0:
            return 0.0
        
        if current_price <= 0:
            raise ValueError(f"Invalid price: {current_price}")
        
        if current_capital <= 0:
            return 0.0
        
        # Use only the specified percentage of capital
        allocated_capital = current_capital * self.percentage
        units = allocated_capital / current_price
        position = signal * units
        
        return position
    
    def _get_return(self, current_price, next_price, position) -> float:
        # Calculate return based on position type
        if position > 0:  # Long position
            trade_return = (next_price - current_price) / current_price
        elif position < 0:  # Short position
            trade_return = (current_price - next_price) / current_price
        else:  # No position
            trade_return = 0.0
        return trade_return * self.percentage
    
    def get_name(self) -> str:
        """Get strategy name with percentage."""
        return f"Fixed Percentage Strategy ({self.percentage*100:.0f}%)"
    

class StochasticErrorStrategy(Strategy):
    """
    Stochastic error strategy: Introduce probabilistic errors in signal execution.

    This strategy simulates imperfect execution or decision-making by randomly 
    inverting trading signals with a given probability (`pct_error`). It is useful 
    for stress-testing trading systems, modeling human or algorithmic error, 
    and analyzing robustness against execution noise.

    Position sizing is still based on a fixed percentage of capital, similar 
    to the FixedPercentageStrategy, but the trade direction may be flipped 
    according to the stochastic error rate.

    Position sizing and behavior:
    - With probability = (1 - pct_error):
        - Signal = 1 (UP): Go long with (percentage)% of capital
        - Signal = -1 (DOWN): Go short with (percentage)% of capital
    - With probability = (pct_error):
        - The signal is inverted (long → short, short → long)
    - Signal = 0 (UNCERTAIN): Stay in cash

    Example:
        With pct_error=0.2 (20% error chance), percentage=0.5 (50%), capital=$1000, price=$50:
        - Normally: Signal=1 → Buy 10 units ($500 / $50)
        - Occasionally (20% chance): Signal=1 → Erroneously Short 10 units (-$500 / $50)

    Attributes:
        pct_error: Probability (0.0 to 1.0) that the signal direction is flipped.
        percentage: Fraction of capital to use per trade (0.0 to 1.0)
    """

    def __init__(self, error_rate, percentage: float = 1):
        """
        Initialize the stochastic error strategy.

        Args:
            pct_error: Probability that the trading signal will be inverted.
            percentage: Fraction of capital to deploy (0.0 to 1.0).
                        Default is 1 (100%).

        Raises:
            ValueError: If percentage is not between 0 and 1.
        """
        if not 0.0 <= percentage <= 1.0:
            raise ValueError(f"Percentage must be between 0 and 1, got {percentage}")
        if not 0.0 <= error_rate <= 1.0:
            raise ValueError(f"Percentage must be between 0 and 1, got {percentage}")
        
        self.error_rate = error_rate
        self.percentage = percentage
    
    def calculate_position(
        self,
        signal: int,
        current_capital: float,
        current_price: float
    ) -> float:
        """
        Calculate position size using a fixed percentage of capital,
        introducing stochastic errors in signal direction.

        Args:
            signal: Trading signal (1 for long, -1 for short, 0 for neutral).
            current_capital: Total capital available.
            current_price: Current price of the asset.
            
        Returns:
            float: Number of units to hold (positive for long, negative for short).
        """

        # Randomly flip the signal based on the error probability
        if random.random() < self.error_rate and signal != 0:
            signal_copy = -signal
        else:
            signal_copy = signal

        if signal_copy == 0:
            return 0.0
        
        if current_price <= 0:
            raise ValueError(f"Invalid price: {current_price}")
        
        if current_capital <= 0:
            return 0.0
        
        # Allocate capital based on fixed percentage
        allocated_capital = current_capital * self.percentage
        units = allocated_capital / current_price
        position = signal_copy * units
        
        return position
    
    def _get_return(self, current_price, next_price, position) -> float:
        """
        Compute trade return based on price change and position type.

        Args:
            current_price: Price at the start of the trade.
            next_price: Price at the end of the trade.
            position: Number of units held (positive for long, negative for short).

        Returns:
            float: Fractional return for this trade, adjusted by `percentage`.
        """
        if position > 0:  # Long position
            trade_return = (next_price - current_price) / current_price
        elif position < 0:  # Short position
            trade_return = (current_price - next_price) / current_price
        else:  # No position
            trade_return = 0.0
        return trade_return * self.percentage
    
    def get_name(self) -> str:
        """Get strategy name with percentage."""
        return f"Stochastic Error Strategy ({self.percentage*100:.0f}%)"
