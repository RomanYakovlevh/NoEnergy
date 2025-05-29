import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib
import matplotlib.pyplot as plt

# Create directories if they don't exist
os.makedirs("models/random_forest", exist_ok=True)
os.makedirs("plots/random_forest", exist_ok=True)

# Load the preprocessed data
data = pd.read_csv('data/outputs/combined_data_preprocessed.csv')
data['Timestamps'] = pd.to_datetime(data['Timestamps'])
data.set_index('Timestamps', inplace=True)

# Load building area data
area_df = pd.read_csv("data/areas.csv")
area_map = dict(zip(area_df["Buid_ID"], area_df["Area [m2]"]))

# List of building columns
building_columns = ['ICT', 'U06, U06A, U05B', 'OBS', 'U05, U04, U04B, GEO',
                    'TEG', 'LIB', 'MEK', 'SOC', 'S01', 'D04']

# Features for training
base_features = ['lag_1', 'rolling_3h', 'rolling_6h', 'hour', 'day_of_week', 'month',
                 'is_weekend', 'air temperature', 'Atm pressure mm of mercury', 'Relative humidity (%)']

for building in building_columns:
    print(f"\nTraining model for: {building}")

    # Train/Test split
    train_data = data.loc['2023-01-01':'2023-08-31']
    test_data = data.loc['2023-09-01':'2023-12-31']

    # Drop rows with NaNs in target
    train_data = train_data.dropna(subset=[building])
    test_data = test_data.dropna(subset=[building])

    # Add building area as a feature
    area = area_map[building]
    train_data["area"] = area
    test_data["area"] = area

    # Features to use
    features = base_features + ["area"]

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
    joblib.dump(model, f'models/random_forest/{safe_building_name}_rndfrst_model.pkl')

    # Plot actual vs predicted
    plt.figure(figsize=(12, 4))
    plt.plot(y_test.index, y_test, label='Actual', color='blue')
    plt.plot(y_test.index, y_pred, label='Predicted', color='red', linestyle='--')
    plt.title(f"{building} - Actual vs Predicted (Sep–Dec)")
    plt.xlabel("Date")
    plt.ylabel("Energy Consumption")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/random_forest/{safe_building_name}_rndfrst_plot.png")
    plt.close()

print("\nAll random forest models trained and saved.")
