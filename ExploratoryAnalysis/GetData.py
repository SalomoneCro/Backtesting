import yfinance as yf
import pandas as pd
import os
from datetime import datetime
from pathlib import Path


class DataFetcher:
    """
    Class to download historical financial asset data using yfinance.
    Works with stocks, indices, forex, ETFs and crypto.
    Supports both daily and intraday data.
    """
    
    def __init__(self):
        self.datasets_dir = Path(__file__).parent / "Datasets"
        self.datasets_dir.mkdir(exist_ok=True)
    
    def fetch_and_save(
        self, 
        ticker: str, 
        start_date: str, 
        end_date: str,
        interval: str = '1d'
    ) -> str:
        """
        Downloads historical data and saves it to CSV with engineered features.
        
        NOTE: The start_date is NOT inclusive in the final dataset because the first row
        is dropped when calculating close_diff (which requires a previous period's close price).
        
        Args:
            ticker: Asset symbol (e.g., 'AAPL', '^GSPC', 'EURUSD=X', 'BTC-USD')
            start_date: Start date in 'YYYY-MM-DD' format (NOT inclusive in final dataset)
            end_date: End date in 'YYYY-MM-DD' format (inclusive)
            interval: Data interval. Valid values:
                      '1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', 
                      '1d', '5d', '1wk', '1mo', '3mo'
                      Default: '1d' (daily)
        
        Returns:
            str: Path to saved CSV file
        
        Raises:
            ValueError: If data cannot be downloaded for the ticker
        """
        print(f"Downloading {interval} data for {ticker}...")
        
        # Download data
        try:
            data = yf.download(
                ticker, 
                start=start_date, 
                end=end_date, 
                interval=interval,
                progress=False
            )
        except Exception as e:
            raise ValueError(f"Error downloading data for {ticker}: {str(e)}")
        
        # Validate that data was downloaded
        if data.empty:
            raise ValueError(
                f"No data found for {ticker} in range {start_date} to {end_date} "
                f"with interval {interval}"
            )
        
        # Reset index to have datetime as a column
        data.reset_index(inplace=True)
        
        # Rename column based on interval (yfinance uses 'Date' for daily, 'Datetime' for intraday)
        if 'Datetime' in data.columns:
            data.rename(columns={'Datetime': 'Date'}, inplace=True)
        
        # Filter only universal columns that exist
        universal_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        available_columns = [col for col in universal_columns if col in data.columns]
        data = data[available_columns]
        
        # Add engineered features
        data = self._add_features(data)
        
        # Drop first row (no previous period for close_diff calculation)
        data = data.iloc[1:].reset_index(drop=True)
        
        # Clean ticker for filename (replace special characters)
        clean_ticker = ticker.replace('^', '').replace('=', '').replace('/', '')
        
        # Get actual start date from data (after dropping first row)
        actual_start_date = data['Date'].min().strftime('%Y-%m-%d')
        
        # Create filename with interval
        interval_suffix = interval.replace('m', 'min').replace('h', 'H')
        filename = f"{clean_ticker}_{actual_start_date}_{end_date}_{interval_suffix}.csv"
        filepath = self.datasets_dir / filename
        
        # Save CSV (overwrites if exists)
        data.to_csv(filepath, index=False)
        
        print(f"✓ Data saved to: {filepath}")
        print(f"  - Rows: {len(data)}")
        print(f"  - Columns: {', '.join(available_columns)}")
        print(f"  - Range: {data['Date'].min()} to {data['Date'].max()}")
        
        return str(filepath)
    
    def _add_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Adds engineered features to the dataset.
        This method is designed to be easily extensible for future features.
        
        Args:
            data: DataFrame with raw OHLCV data
        
        Returns:
            pd.DataFrame: DataFrame with added features
        """
        # Feature 1: close_diff - Percentage change from previous close
        # Positive if price increased, negative if price decreased
        data['close_diff'] = data['Close'].pct_change()
        
        # Future features can be added here:
        # data['returns'] = data['Close'].pct_change()
        # data['sma_20'] = data['Close'].rolling(window=20).mean()
        # data['rsi'] = self._calculate_rsi(data['Close'])
        # etc.
        
        return data
    
    def load_dataset(self, filepath: str) -> pd.DataFrame:
        """
        Loads a previously saved dataset.
        
        Args:
            filepath: Path to CSV file
        
        Returns:
            pd.DataFrame: Loaded dataset
        """
        return pd.read_csv(filepath, parse_dates=['Date'])


# Usage example
if __name__ == "__main__":
    fetcher = DataFetcher()
    
    # Example: Download hourly data
    # fetcher.fetch_and_save("SPY", "2023-01-01", "2023-12-31", interval="1h")
    
    # Example: Download daily data (default)
    # fetcher.fetch_and_save("AAPL", "2020-01-01", "2024-12-31")
    
    print("DataFetcher ready. Uncomment examples to test.")