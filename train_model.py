# train_model.py

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# Load the preprocessed data
data = pd.read_csv('data/outputs/combined_data_preprocessed.csv')
data['Timestamps'] = pd.to_datetime(data['Timestamps'])
data.set_index('Timestamps', inplace=True)

# Split data into features (X) and target (y)
X = data[['lag_1', 'rolling_3h', 'rolling_6h', 'hour', 'day_of_week', 'month', 'is_weekend',
          'air temperature', 'Atm pressure mm of mercury', 'Relative humidity (%)']]
y = data['ICT']  # Target variable (Energy Consumption)

# Split data into training and test sets (use the first 9 months for training and last 3 for testing)
train_data = data['2023-01-01':'2023-09-30']
test_data = data['2023-10-01':'2023-12-31']

# Features and target for training
X_train = train_data[['lag_1', 'rolling_3h', 'rolling_6h', 'hour', 'day_of_week', 'month', 'is_weekend',
                      'air temperature', 'Atm pressure mm of mercury', 'Relative humidity (%)']]
y_train = train_data['ICT']

# Features and target for testing
X_test = test_data[['lag_1', 'rolling_3h', 'rolling_6h', 'hour', 'day_of_week', 'month', 'is_weekend',
                     'air temperature', 'Atm pressure mm of mercury', 'Relative humidity (%)']]
y_test = test_data['ICT']

# Initialize the model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test data (3 months you will receive during the hackathon)
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error on 3-month test data: {mae}')

# Save the model if needed (optional)
import joblib
joblib.dump(model, 'model/random_forest_model.pkl')

# Visualize results
import matplotlib.pyplot as plt

# Plot predicted vs actual values for the test period
plt.figure(figsize=(12, 6))
plt.plot(y_test.index, y_test, label='Actual', color='blue')
plt.plot(y_test.index, y_pred, label='Predicted', color='red', linestyle='--')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Energy Consumption (ICT)')
plt.title('Energy Consumption Prediction vs Actual (3-month test data)')
plt.xticks(rotation=45)
plt.show()
