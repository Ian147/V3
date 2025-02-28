import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Simulasi data harga kripto (gunakan data asli jika tersedia)
data = {'close': np.random.rand(1000) * 50000 + 30000}  # Harga antara 30K-80K
df = pd.DataFrame(data)

# Normalisasi data
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df)

# Buat dataset time series untuk LSTM
X, y = [], []
seq_length = 10  # Panjang sequence untuk prediksi
for i in range(len(df_scaled) - seq_length):
    X.append(df_scaled[i:i+seq_length])
    y.append(df_scaled[i+seq_length])

X, y = np.array(X), np.array(y)

# Bangun model LSTM
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
    LSTM(50, return_sequences=False),
    Dense(25, activation="relu"),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=20, batch_size=16, verbose=1)

# Simpan model
model.save("lstm_model.h5")
