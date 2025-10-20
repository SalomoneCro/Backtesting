import pandas as pd

# Load your data
data = pd.read_csv("ExploratoryAnalysis/Datasets/USDCHFX_2018-01-02_2022-01-01.csv")

# Calculate gaps
data['Gap'] = data['Open'] - data['Close'].shift(1)
data['Intraday'] = data['Close'] - data['Open']

# Stats
print("Close-to-Close moves (what oracle predicts):")
print(f"  Mean: {data['close_diff'].mean():.6f}")
print(f"  Std: {data['close_diff'].std():.6f}")

print("\nGaps (Close to Open):")
print(f"  Mean: {data['Gap'].mean():.6f}")
print(f"  Std: {data['Gap'].std():.6f}")

print("\nIntraday moves (what you actually trade):")
print(f"  Mean: {data['Intraday'].mean():.6f}")
print(f"  Std: {data['Intraday'].std():.6f}")

# Correlation
print(f"\nCorrelation between close_diff and intraday move: {data['close_diff'].corr(data['Intraday']):.3f}")