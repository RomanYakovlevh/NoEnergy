import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import os

# Load preprocessed data
df = pd.read_csv("data/outputs/combined_data_preprocessed.csv", parse_dates=["Timestamps"], index_col="Timestamps")

# Select only the target and features relevant to ICT
target_col = "ICT"
feature_cols = ['lag_1', 'rolling_3h', 'rolling_6h', 'hour', 'day_of_week', 'month', 'is_weekend',
                'air temperature', 'Atm pressure mm of mercury', 'Relative humidity (%)']

# Drop rows with missing values
df = df.dropna(subset=[target_col] + feature_cols)

# Scale features
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(df[feature_cols])
y_scaled = scaler_y.fit_transform(df[[target_col]])

# Create sequences for LSTM (lookback=24 hours)
def create_sequences(X, y, lookback=24):
    Xs, ys = [], []
    for i in range(lookback, len(X)):
        Xs.append(X[i - lookback:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)

X_seq, y_seq = create_sequences(X_scaled, y_scaled)

# Train/test split (last 20% as test set)
split = int(0.8 * len(X_seq))
X_train, X_test = X_seq[:split], X_seq[split:]
y_train, y_test = y_seq[:split], y_seq[split:]

# Build LSTM model
model = Sequential([
    LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# Train model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1)

# Save model
os.makedirs("models/ml/lstm", exist_ok=True)
model.save("models/ml/lstm/ICT_lstm_model.keras")
print("✅ Model saved to models/ml/lstm/ICT_lstm_model.keras")
