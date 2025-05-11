# train_models_all_random_forest.py

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib
import matplotlib.pyplot as plt

import os

# Create directories if they don't exist
os.makedirs("models/random_forest", exist_ok=True)
os.makedirs("plots/random_forest", exist_ok=True)

# Load the preprocessed data
data = pd.read_csv('data/outputs/combined_data_preprocessed.csv')
data['Timestamps'] = pd.to_datetime(data['Timestamps'])
data.set_index('Timestamps', inplace=True)

# List of building columns (corrected based on your actual data)
building_columns = ['ICT', 'U06, U06A, U05B', 'OBS', 'U05, U04, U04B, GEO',
                    'TEG', 'LIB', 'MEK', 'SOC', 'S01', 'D04']

# Features for training (assumed from preprocessing)
features = ['lag_1', 'rolling_3h', 'rolling_6h', 'hour', 'day_of_week', 'month',
            'is_weekend', 'air temperature', 'Atm pressure mm of mercury', 'Relative humidity (%)']

for building in building_columns:
    print(f"\nTraining model for: {building}")

    # Train/Test split
    train_data = data.loc['2023-01-01':'2023-08-31']
    test_data = data.loc['2023-09-01':'2023-12-31']

    # Drop rows with NaNs in target
    train_data = train_data.dropna(subset=[building])
    test_data = test_data.dropna(subset=[building])

    X_train = train_data[features]
    y_train = train_data[building]

    X_test = test_data[features]
    y_test = test_data[building]

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"MAE for {building}: {mae:.3f}")

    # Save model
    safe_building_name = building.replace(", ", "_").replace(" ", "_")
    joblib.dump(model, f'models/random_forest/{safe_building_name}_model.pkl')

    # Plot actual vs predicted
    plt.figure(figsize=(12, 4))
    plt.plot(y_test.index, y_test, label='Actual', color='blue')
    plt.plot(y_test.index, y_pred, label='Predicted', color='red', linestyle='--')
    plt.title(f"{building} - Actual vs Predicted (Sep–Dec)")
    plt.xlabel("Date")
    plt.ylabel("Energy Consumption")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/random_forest/{safe_building_name}_plot.png")
    plt.close()

print("\nAll models trained and saved.")
