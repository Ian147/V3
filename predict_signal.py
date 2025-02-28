import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load model
model = tf.keras.models.load_model("lstm_model.h5")

# Simulasi data terbaru untuk prediksi
seq_length = 10  # Sesuaikan dengan panjang sequence yang digunakan saat training
last_prices = np.random.rand(seq_length) * 50000 + 30000  # Data harga acak

# Pastikan scaler cocok dengan training
scaler = MinMaxScaler()
last_prices_scaled = scaler.fit_transform(last_prices.reshape(-1, 1))

# Sesuaikan format input untuk model LSTM
input_data = np.expand_dims(last_prices_scaled, axis=0)

# Prediksi harga berikutnya
predicted_price = model.predict(input_data)
predicted_price = scaler.inverse_transform(predicted_price.reshape(-1, 1))

# Logika untuk sinyal BUY/SELL
signal = "BUY" if predicted_price > last_prices[-1] else "SELL"

print(f"Prediksi Harga Berikutnya: {predicted_price[0][0]}")
print(f"Sinyal: {signal}")
