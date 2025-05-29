import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from glob import glob
from sklearn.preprocessing import MinMaxScaler

# Configuration
MODELS_BASE_DIR = "models"
PREDICTION_BASE_DIR = "prediction"
NEW_DATA_PATH = "data/new_year_data.csv"

# Features used during training
features = ['lag_1', 'rolling_3h', 'rolling_6h', 'hour', 'day_of_week', 'month', 'is_weekend',
            'air temperature', 'Atm pressure mm of mercury', 'Relative humidity (%)']

# Load and preprocess new year data
df = pd.read_csv(NEW_DATA_PATH)
df['Timestamps'] = pd.to_datetime(df['Timestamps'])
df.set_index('Timestamps', inplace=True)

# Time-based features
df['hour'] = df.index.hour
df['day_of_week'] = df.index.dayofweek
df['month'] = df.index.month
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

# Normalize weather columns
weather_columns = ['air temperature', 'Atm pressure mm of mercury', 'Relative humidity (%)']
scaler = MinMaxScaler()
df[weather_columns] = scaler.fit_transform(df[weather_columns])

# Go through model types
for model_type in os.listdir(MODELS_BASE_DIR):
    model_dir = os.path.join(MODELS_BASE_DIR, model_type)
    if not os.path.isdir(model_dir):
        continue

    print(f"\n🔍 Processing model type: {model_type}")

    # Output directory
    output_dir = os.path.join(PREDICTION_BASE_DIR, model_type)
    os.makedirs(output_dir, exist_ok=True)

    for model_path in glob(f"{model_dir}/*.pkl"):
        filename = os.path.basename(model_path)

        if model_type == "random_forest":
            building_key = filename.replace("_rndfrst_model.pkl", "")
        elif model_type == "xgboost":
            building_key = filename.replace("_xgb_model.pkl", "")
        else:
            continue  # skip unknown model types

        building_column = building_key.replace("_", ", ")

        if building_column not in df.columns:
            print(f"⚠️ Skipping {building_column} — not found in data columns")
            continue

        print(f"⏳ Predicting for: {building_column}")

        # Create lag and rolling features
        df['lag_1'] = df[building_column].shift(1)
        df['rolling_3h'] = df[building_column].rolling(3).mean()
        df['rolling_6h'] = df[building_column].rolling(6).mean()

        df_model = df.dropna(subset=['lag_1', 'rolling_3h', 'rolling_6h'])

        X = df_model[features]
        y_true = df_model[building_column]

        # Load model and predict
        model = joblib.load(model_path)
        y_pred = model.predict(X)

        # Output file paths
        safe_name = building_key  # e.g., U06_U06A_U05B
        pred_csv_path = os.path.join(output_dir, f"prediction_{safe_name}.csv")
        plot_path = os.path.join(output_dir, f"plot_{safe_name}.png")

        # Save CSV
        pd.DataFrame({
            'Timestamps': df_model.index,
            'Actual': y_true.values,
            'Predicted': y_pred
        }).to_csv(pred_csv_path, index=False)

        # Save plot
        plt.figure(figsize=(12, 5))
        plt.plot(df_model.index, y_true, label="Actual", color='blue')
        plt.plot(df_model.index, y_pred, label="Predicted", color='red', linestyle='--')
        plt.title(f"{building_column} - Prediction ({model_type})")
        plt.xlabel("Time")
        plt.ylabel("Energy Consumption")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

        print(f"✅ Saved: {pred_csv_path}, {plot_path}")

print("\n🎯 All model predictions complete.")
