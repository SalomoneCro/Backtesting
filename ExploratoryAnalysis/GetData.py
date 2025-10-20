import yfinance as yf
import pandas as pd
import os
from datetime import datetime
from pathlib import Path


class DataFetcher:
    """
    Class to download historical financial asset data using yfinance.
    Works with stocks, indices, forex, ETFs and crypto.
    """
    
    def __init__(self):
        self.datasets_dir = Path(__file__).parent / "Datasets"
        self.datasets_dir.mkdir(exist_ok=True)
    
    def fetch_and_save(self, ticker: str, start_date: str, end_date: str) -> str:
        """
        Downloads historical data and saves it to CSV with engineered features.
        
        NOTE: The start_date is NOT inclusive in the final dataset because the first row
        is dropped when calculating close_diff (which requires a previous day's close price).
        
        Args:
            ticker: Asset symbol (e.g., 'AAPL', '^GSPC', 'EURUSD=X', 'BTC-USD')
            start_date: Start date in 'YYYY-MM-DD' format (NOT inclusive in final dataset)
            end_date: End date in 'YYYY-MM-DD' format (inclusive)
        
        Returns:
            str: Path to saved CSV file
        
        Raises:
            ValueError: If data cannot be downloaded for the ticker
        """
        print(f"Downloading data for {ticker}...")
        
        # Download data
        try:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        except Exception as e:
            raise ValueError(f"Error downloading data for {ticker}: {str(e)}")
        
        # Validate that data was downloaded
        if data.empty:
            raise ValueError(f"No data found for {ticker} in range {start_date} to {end_date}")

        # Reset index to have date as a column
        data.reset_index(inplace=True)
        
        # Filter only universal columns that exist
        universal_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        available_columns = [col for col in universal_columns if col in data.columns]
        data = data[available_columns]
        
        # Add engineered features
        data = self._add_features(data)
        
        # Drop first row (no previous day for close_diff calculation)
        data = data.iloc[1:].reset_index(drop=True)
        
        # Clean ticker for filename (replace special characters)
        clean_ticker = ticker.replace('^', '').replace('=', '').replace('/', '')
        
        # Create filename
        filename = f"{clean_ticker}_{data['Date'].min().strftime('%Y-%m-%d')}_{end_date}.csv"
        filepath = self.datasets_dir / filename
        
        #By default, the columns gotten by yfinance are MultiIndex, this flatten them
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] if col[0] else col[1] for col in data.columns]

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
        # Feature 1: close_diff - Difference between current and previous close
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
    
    # Examples with different asset types
    examples = [
        ("SPY", "2020-01-01", "2024-12-31"),  # S&P 500 ETF
        ("AAPL", "2020-01-01", "2024-12-31"),  # Stock
        ("^GSPC", "2020-01-01", "2024-12-31"),  # S&P 500 Index
        ("EURUSD=X", "2020-01-01", "2024-12-31"),  # Forex
        ("BTC-USD", "2020-01-01", "2024-12-31"),  # Crypto
    ]
    
    # Uncomment to test
    # for ticker, start, end in examples:
    #     try:
    #         fetcher.fetch_and_save(ticker, start, end)
    #         print()
    #     except ValueError as e:
    #         print(f"Error: {e}\n")