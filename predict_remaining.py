# predict_remaining_random_forest.py

import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler

# Load the trained model
model = joblib.load('model/random_forest_model.pkl')

# Simulate new data for the next year (e.g., first 3 months of 2024)
# Replace this with actual new 3-month data for testing
new_year_data = pd.read_csv('data/new_year_data.csv')  # This file will be given at the event
new_year_data['Timestamps'] = pd.to_datetime(new_year_data['Timestamps'])
new_year_data.set_index('Timestamps', inplace=True)

# Preprocess the new year's data (same as above)
new_year_data['hour'] = new_year_data.index.hour
new_year_data['day_of_week'] = new_year_data.index.dayofweek
new_year_data['month'] = new_year_data.index.month
new_year_data['is_weekend'] = new_year_data['day_of_week'].isin([5, 6]).astype(int)

# Normalize the weather-related features
scaler = MinMaxScaler()
weather_columns = ['air temperature', 'Atm pressure mm of mercury', 'Relative humidity (%)']
new_year_data[weather_columns] = scaler.fit_transform(new_year_data[weather_columns])

# Create lag and rolling features
new_year_data['lag_1'] = new_year_data['ICT'].shift(1)
new_year_data['rolling_3h'] = new_year_data['ICT'].rolling(window=3).mean()
new_year_data['rolling_6h'] = new_year_data['ICT'].rolling(window=6).mean()

# Drop missing values due to lag/rolling window
new_year_data.dropna(inplace=True)

# Make predictions for the next 9 months (remaining months of the year)
X_new = new_year_data[['lag_1', 'rolling_3h', 'rolling_6h', 'hour', 'day_of_week', 'month', 'is_weekend',
                        'air temperature', 'Atm pressure mm of mercury', 'Relative humidity (%)']]

# Predict the energy consumption for the next months
predictions = model.predict(X_new)

# Add predictions back to the data for visualization
new_year_data['Predicted ICT'] = predictions

# Visualize predictions (actual vs. predicted for new year)
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(new_year_data.index, new_year_data['ICT'], label='Actual', color='blue')
plt.plot(new_year_data.index, new_year_data['Predicted ICT'], label='Predicted', color='red', linestyle='--')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Energy Consumption (ICT)')
plt.title('Energy Consumption Prediction for the New Year')
plt.xticks(rotation=45)
plt.show()
plt.savefig(f"prediction/random_forest/plot.png")