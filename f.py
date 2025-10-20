import pandas as pd
import numpy as np

data = pd.read_csv( "ExploratoryAnalysis/Datasets/USDCHFX_2018-01-02_2022-01-01.csv")

print("AAPL Analysis:")
print(f"Total days: {len(data)}")
print(f"Date range: {data['Date'].min()} to {data['Date'].max()}")
print()

print("close_diff statistics:")
print(f"  Mean: {data['close_diff'].mean():.6f}")
print(f"  Median: {data['close_diff'].median():.6f}")
print(f"  Std: {data['close_diff'].std():.6f}")
print()

print("Distribution:")
print(f"  Up days: {(data['close_diff'] > 0).sum()}")
print(f"  Down days: {(data['close_diff'] < 0).sum()}")
print(f"  Flat days: {(data['close_diff'] == 0).sum()}")
print()

# Theoretical maximum return
print("If we capture every move perfectly:")
up_moves = data[data['close_diff'] > 0]['close_diff']
down_moves = data[data['close_diff'] < 0]['close_diff'].abs()

# All-in strategy: capture all up moves, capture all down moves (shorting)
theoretical_return = 1.0
for move in up_moves:
    theoretical_return *= (1 + move)
for move in down_moves:
    theoretical_return *= (1 + move)

print(f"  Theoretical return: {(theoretical_return - 1) * 100:.2f}%")
print()

# What's the actual sum if we add them?
total_captured = up_moves.sum() + down_moves.sum()
print(f"  Sum of all absolute moves: {total_captured * 100:.2f}%")