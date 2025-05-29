import os
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# Load preprocessed data
df = pd.read_csv("data/outputs/combined_data_preprocessed.csv", parse_dates=["Timestamps"])
df.set_index("Timestamps", inplace=True)

# Load building area data
area_df = pd.read_csv("data/areas.csv")
area_map = dict(zip(area_df["Buid_ID"], area_df["Area [m2]"]))

# Base features
base_features = ['lag_1', 'rolling_3h', 'rolling_6h', 'hour', 'day_of_week', 'month', 'is_weekend',
                 'air temperature', 'Atm pressure mm of mercury', 'Relative humidity (%)']

# Building groups
building_columns = ['ICT', 'U06, U06A, U05B', 'OBS', 'U05, U04, U04B, GEO',
                    'TEG', 'LIB', 'MEK', 'SOC', 'S01', 'D04']

# Output directory
model_dir = "models/svr"
os.makedirs(model_dir, exist_ok=True)

for building in building_columns:
    print(f"\nTraining SVR model for {building}...")

    df_building = df.dropna(subset=[building] + base_features)

    # Add building area as a constant feature
    area = area_map[building]
    df_building["area"] = area

    features = base_features + ["area"]
    X = df_building[features]
    y = df_building[building]

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save scaler
    scaler_filename = f"{building.replace(', ', '_').replace(' ', '')}_scaler.pkl"
    joblib.dump(scaler, os.path.join(model_dir, scaler_filename))

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

    # Train model
    model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"{building} - MAE: {mae:.2f}, R²: {r2:.2f}")

    # Save model
    model_filename = building.replace(", ", "_").replace(" ", "") + "_svr_model.pkl"
    joblib.dump(model, os.path.join(model_dir, model_filename))

print("\nAll SVR models trained and saved.")
