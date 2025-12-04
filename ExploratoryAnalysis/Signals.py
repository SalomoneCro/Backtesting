import pandas as pd
import numpy as np

# Load your data
data = pd.read_csv("ExploratoryAnalysis/Datasets/USDCHFX_2018-01-02_2022-01-01.csv")

data['signal'] = np.sign(data['close_diff'])
data['Date'] = pd.to_datetime(data['Date'])

signal_map = {
    1: 'buy',
    -1: 'sell'
}

# Filtrar solo señales válidas (no cero)
signals = data[data['signal'] != 0].copy()
signals['action'] = signals['signal'].map(signal_map)

# Crear DataFrame para las señales horarias
hourly_signals = []

for _, row in signals.iterrows():
    base_date = row['Date'].date()
    action = row['action']
    
    # Generar 16 señales: de 8:00 a 23:00 (8, 9, 10, ..., 23)
    for hour in range(8, 24):
        signal_time = pd.Timestamp(year=base_date.year, 
                                   month=base_date.month, 
                                   day=base_date.day, 
                                   hour=hour, 
                                   minute=0, 
                                   second=0)
        
        hourly_signals.append({
            'format_Date': signal_time.strftime('%Y.%m.%d %H:%M:%S'),
            'action': action
        })

# Convertir a DataFrame
hourly_df = pd.DataFrame(hourly_signals)

# Guardar CSV
hourly_df.to_csv('trading_schedule.csv', header=False, index=False)