# preprocess_data.py

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the raw data
data = pd.read_csv('data/outputs/combined_data.csv')

# Convert Timestamps to datetime
data['Timestamps'] = pd.to_datetime(data['Timestamps'])

# Set Timestamps as the index
data.set_index('Timestamps', inplace=True)

# Handle missing values by forward filling
data.fillna(method='ffill', inplace=True)

# Extract datetime features (hour, day, month, etc.)
data['hour'] = data.index.hour
data['day_of_week'] = data.index.dayofweek
data['month'] = data.index.month
data['is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)

# Normalize weather-related features
scaler = MinMaxScaler()
weather_columns = ['air temperature', 'Atm pressure mm of mercury', 'Relative humidity (%)']
data[weather_columns] = scaler.fit_transform(data[weather_columns])

# Create lag and rolling features
data['lag_1'] = data['ICT'].shift(1)
data['rolling_3h'] = data['ICT'].rolling(window=3).mean()
data['rolling_6h'] = data['ICT'].rolling(window=6).mean()

# Drop missing values due to lag/rolling window
data.dropna(inplace=True)

# Save the preprocessed data
data.to_csv('data/outputs/combined_data_preprocessed.csv')

# Optionally, print the first few rows to check
print(data.head())
