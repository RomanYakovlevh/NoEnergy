import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load the trained model
model = load_model("models/ml/lstm/ICT_lstm_model.keras")

# Load data
df = pd.read_csv("data/outputs/combined_data_preprocessed.csv", parse_dates=["Timestamps"], index_col="Timestamps")
target_col = "ICT"
feature_cols = ['lag_1', 'rolling_3h', 'rolling_6h', 'hour', 'day_of_week', 'month', 'is_weekend',
                'air temperature', 'Atm pressure mm of mercury', 'Relative humidity (%)']
df = df.dropna(subset=[target_col] + feature_cols)

# Scale the data
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(df[feature_cols])
y_scaled = scaler_y.fit_transform(df[[target_col]])

# Create sequences for LSTM prediction
def create_sequences(X, y, lookback=24):
    Xs, ys, timestamps = [], [], []
    for i in range(lookback, len(X)):
        Xs.append(X[i - lookback:i])
        ys.append(y[i])
        timestamps.append(df.index[i])
    return np.array(Xs), np.array(ys), timestamps

X_seq, y_seq, timestamps = create_sequences(X_scaled, y_scaled)
timestamps = pd.to_datetime(timestamps)

# Make predictions
y_pred_scaled = model.predict(X_seq)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_true = scaler_y.inverse_transform(y_seq)

# Ensure directory exists
output_path = "prediction/ml/lstm/plot_ICT.png"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Plot the actual vs predicted energy consumption
plt.figure(figsize=(14, 5))
plt.plot(timestamps, y_true, label="Actual", color="blue")
plt.plot(timestamps, y_pred, label="Predicted", color="red", linestyle="--")
plt.title("ICT - LSTM Prediction")
plt.xlabel("Time")
plt.ylabel("Energy Consumption")
plt.legend()
plt.tight_layout()
plt.savefig(output_path)
plt.show()

print(f"Prediction plot saved as {output_path}")
