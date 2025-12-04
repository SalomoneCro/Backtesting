from ExploratoryAnalysis.GetData import DataFetcher

fetcher = DataFetcher()
fetcher.fetch_and_save("AAPL", "2023-12-31", "2025-01-01", interval='1d')