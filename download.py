from ExploratoryAnalysis.GetData import DataFetcher

fetcher = DataFetcher()
fetcher.fetch_and_save("USDCHF=X", "2017-12-31", "2022-01-01")