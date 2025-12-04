import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta
from pathlib import Path


class DataFetcher:
    """
    Class to download historical financial asset data using yfinance.
    Now includes advanced feature engineering: rolling mean features
    for 3, 7, 30, and 90 periods, and a categorical target variable.
    """
    
    def __init__(self):
        # Setting up directory relative to the script location
        try:
            self.datasets_dir = Path(__file__).parent / "Datasets"
        except NameError:
            # Fallback for environments like notebooks where __file__ is not defined
            self.datasets_dir = Path(".").resolve() / "Datasets"
            
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
        
        Args:
            ticker: Asset symbol (e.g., 'AAPL', '^GSPC', 'EURUSD=X', 'BTC-USD')
            start_date: Start date in 'YYYY-MM-DD' format (This is the start date 
                        of the final dataset, NOT the download start date).
            end_date: End date in 'YYYY-MM-DD' format (inclusive)
            interval: Data interval.
            
        Returns:
            str: Path to saved CSV file
            
        Raises:
                ValueError: If data cannot be downloaded for the ticker
        """
        
        # --- 1. Ajustar la fecha de inicio para el lookback ---
        # El lookback más grande es de 90 días. Usamos 100 días de buffer.
        MAX_LOOKBACK_DAYS = 200 
        
        original_start_date_dt = datetime.strptime(start_date, '%Y-%m-%d')
        download_start_date_dt = original_start_date_dt - timedelta(days=MAX_LOOKBACK_DAYS)
        download_start_date_str = download_start_date_dt.strftime('%Y-%m-%d')
        
        print(f"Downloading {interval} data for {ticker} (Lookback from {download_start_date_str} to {end_date})...")
        
        # Download data
        try:
            data = yf.download(
                ticker, 
                start=download_start_date_str, # Usamos la fecha extendida
                end=end_date, 
                interval=interval,
                progress=False
            )
        except Exception as e:
            raise ValueError(f"Error downloading data for {ticker}: {str(e)}")
        
        # Validate that data was downloaded
        if data.empty:
            raise ValueError(
                f"No data found for {ticker} in range {download_start_date_str} to {end_date} "
                f"with interval {interval}"
            )
        
        # FIX: Flatten MultiIndex Columns (Needed for some tickers/intervals)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(-1)

        # Reset index to have datetime as a column
        data.reset_index(inplace=True)
        
        # Rename column based on interval (yfinance uses 'Date' for daily, 'Datetime' for intraday)
        if 'Datetime' in data.columns:
            data.rename(columns={'Datetime': 'Date'}, inplace=True)
        
        # Filter only universal columns that exist
        universal_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        available_columns = [col for col in universal_columns if col in data.columns]
        data = data[available_columns]
        
        # --- 2. Add engineered features (including the new target and rolling means) ---
        data = self._add_features(data)
        
        # --- 3. Filtrar el dataset a las fechas originales y limpiar NaN ---
        
        # 3a. Filtrar para que solo queden las filas solicitadas por el usuario
        data = data[data['Date'] >= original_start_date_dt].reset_index(drop=True)
        
        # 3b. Eliminar la última fila que tiene NaN en la columna 'target' (ya que no tiene día siguiente)
        # Solo eliminamos NaNs en la columna 'target', las features ya deben estar completas.
        data = data.dropna(subset=['target']).reset_index(drop=True)
        
        # Clean ticker for filename (replace special characters)
        clean_ticker = ticker.replace('^', '').replace('=', '').replace('/', '')
        
        # Get actual start date from data (after dropping rows)
        actual_start_date = data['Date'].min().strftime('%Y-%m-%d')
        
        filename = f"{clean_ticker}_{actual_start_date}_{end_date}.csv"
        filepath = self.datasets_dir / filename
        
        # Save CSV (overwrites if exists)
        data.to_csv(filepath, index=False)
        
        print(f"✓ Data saved to: {filepath}")
        print(f"  - Rows: {len(data)}")
        print(f"  - Columns: {', '.join(data.columns)}")
        print(f"  - Range: {data['Date'].min()} to {data['Date'].max()}")
        
        return str(filepath)
    
    def _add_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Adds engineered features to the dataset, including:
        1. Categorical target variable (0, 1, 2) based on the next period's price change (0.5% threshold).
        2. Rolling Mean (MA) features for [Open, High, Low, Close, Volume] across 3, 7, 30, and 90 periods.
        """
        
        # --- A. Creación de Rolling Features (Ventana Móvil) ---
        features_to_roll = ['Open', 'High', 'Low', 'Close', 'Volume']
        windows = [3, 7, 30, 90]

        for feature in features_to_roll:
            for w in windows:
                # Calculamos la media móvil (MA) y usamos .shift(1) para EVITAR DATA LEAKAGE.
                # El valor de T debe basarse solo en datos de T-1, T-2, ...
                data[f'{feature}_MA{w}'] = data[feature].rolling(window=w).mean().shift(1)
        
        # --- B. Creación de la Variable Objetivo (Target) ---

        # 1. Calcular el cambio porcentual del precio de cierre del DÍA SIGUIENTE
        pct_change = data['Close'].pct_change().shift(-1)
        
        # Definir los umbrales (0.5% según tu última solicitud)
        THRESHOLD = 0.005
        
        # 2. Asignar las etiquetas categóricas
        
        # Inicializar la columna Target con el valor '2' (Neutral/Moderado)
        data['target'] = 2 
        
        # Asignar '0' (Bajada Fuerte) si la caída es >= 0.5%
        data.loc[pct_change <= -THRESHOLD, 'target'] = 0
        
        # Asignar '1' (Subida Fuerte) si la subida es >= 0.5%
        data.loc[pct_change >= THRESHOLD, 'target'] = 1
        
        # 3. Eliminar la columna 'Adj Close' si existe 
        if 'Adj Close' in data.columns:
            data = data.drop(columns=['Adj Close'])
            
        return data
    
    def load_dataset(self, filepath: str) -> pd.DataFrame:
        """
        Loads a previously saved dataset.
        """
        return pd.read_csv(filepath, parse_dates=['Date'])
