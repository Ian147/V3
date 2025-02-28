import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler

# **1. Load Model LSTM**
model = tf.keras.models.load_model("lstm_model.keras")

# **2. Simulasi Data Harga Terakhir (Gantilah dengan Data Aktual)**
seq_length = 10  # Sesuai dengan model yang sudah dilatih
last_prices = np.random.rand(seq_length) * 50000 + 30000  # Data harga acak
volume = np.random.rand(seq_length) * 100000  # Data volume acak

df = pd.DataFrame({"Close": last_prices, "Volume": volume})

# **3. Standarisasi Input LSTM**
scaler = StandardScaler()
input_data = scaler.fit_transform(last_prices.reshape(-1, 1))
input_data = np.expand_dims(input_data, axis=0)  # Sesuaikan format LSTM

# **4. Prediksi Harga Menggunakan LSTM**
predicted_price_lstm = model.predict(input_data)
predicted_price_lstm = scaler.inverse_transform(predicted_price_lstm.reshape(-1, 1))

# **5. Machine Learning Model (Random Forest & XGBoost)**
X_train = np.random.rand(100, 10)
y_train = np.random.rand(100) * 50000 + 30000  # Harga acak
X_test = np.random.rand(1, 10)

scaler_ml = StandardScaler()
X_train_scaled = scaler_ml.fit_transform(X_train)
X_test_scaled = scaler_ml.transform(X_test)

# **Random Forest**
rf_model = RandomForestRegressor(n_estimators=100)
rf_model.fit(X_train_scaled, y_train)
rf_predicted_price = rf_model.predict(X_test_scaled)

# **XGBoost**
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
xgb_model.fit(X_train_scaled, y_train)
xgb_predicted_price = xgb_model.predict(X_test_scaled)

# **6. Ensemble Learning (Menggabungkan Semua Model)**
final_predicted_price = (predicted_price_lstm[0][0] + rf_predicted_price[0] + xgb_predicted_price[0]) / 3

# **7. Output Prediksi Harga**
print(f"Prediksi Harga LSTM: {predicted_price_lstm[0][0]}")
print(f"Prediksi Harga RF: {rf_predicted_price[0]}")
print(f"Prediksi Harga XGBoost: {xgb_predicted_price[0]}")
print(f"Final Prediksi Harga: {final_predicted_price}")
