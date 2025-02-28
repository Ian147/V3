# Load model
model = tf.keras.models.load_model("lstm_model.h5")

# Simulasi data terbaru untuk prediksi
last_prices = np.random.rand(seq_length) * 50000 + 30000  # Data harga acak
input_data = scaler.transform(last_prices.reshape(-1, 1))
input_data = np.expand_dims(input_data, axis=0)  # Sesuaikan dengan input LSTM

# Prediksi harga berikutnya
predicted_price = model.predict(input_data)
predicted_price = scaler.inverse_transform(predicted_price.reshape(-1, 1))

# Logika untuk sinyal BUY/SELL
if predicted_price > last_prices[-1]:
    signal = "BUY"
else:
    signal = "SELL"

print(f"Prediksi Harga Berikutnya: {predicted_price[0][0]}")
print(f"Sinyal: {signal}")
