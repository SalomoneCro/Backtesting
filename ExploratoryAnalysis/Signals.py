import pandas as pd
import numpy as np
# Load your data
data = pd.read_csv("ExploratoryAnalysis/Datasets/USDCHFX_2018-01-02_2022-01-01.csv")

data['signal'] = np.sign(data['close_diff'])
data['Date'] = pd.to_datetime(data['Date'])

signal_map = {
    1:'buy',
    -1:'sell'
}

signals = data[data['signal'] != 0].copy()

signals['action'] = signals['signal'].map(signal_map)

signals['format_Date'] = pd.to_datetime(signals['Date'].dt.date) + pd.Timedelta(hours=8)

signals['format_Date'] = signals['format_Date'].dt.strftime('%Y.%m.%d %H:%M:%S')

signals[['format_Date', 'action']].to_csv('trading_schedule.csv', header=False, index=False)
