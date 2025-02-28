import os
import numpy as np
import pandas as pd
import tensorflow as tf
from binance.client import Client
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv

# **1. Load API Key dari .env**
load_dotenv()
API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET")

# **2. Koneksi ke Binance**
client = Client(API_KEY, API_SECRET)

# **3. Ambil Data Harga BTC/USDT Real-Time dari Binance**
symbol = "BTCUSDT"
klines = client.get_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_1MINUTE, limit=50)

# **4. Konversi ke DataFrame**
df = pd.DataFrame(klines, columns=["time", "open", "high", "low", "close", "volume", "close_time",
                                   "quote_asset_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"])
df["close"] = df["close"].astype(float)
df["volume"] = df["volume"].astype(float)

# **5. Muat Model LSTM**
model = tf.keras.models.load_model("lstm_model.keras")

# **6. Standarisasi Input LSTM**
scaler = StandardScaler()
input_data = scaler.fit_transform(df["close"].values[-10:].reshape(-1, 1))
input_data = np.expand_dims(input_data, axis=0)  # Sesuaikan format LSTM

# **7. Prediksi Harga dengan LSTM**
predicted_price_lstm = model.predict(input_data)
predicted_price_lstm = scaler.inverse_transform(predicted_price_lstm.reshape(-1, 1))

# **8. Machine Learning Model (Random Forest & XGBoost)**
X_train = np.random.rand(100, 10)
y_train = np.random.rand(100) * 50000 + 30000  # Harga acak untuk latihan
X_test = np.array([df["close"].values[-10:]])  # Gunakan data real sebagai input

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

# **9. Ensemble Learning (Gabungkan Semua Model)**
final_predicted_price = (predicted_price_lstm[0][0] + rf_predicted_price[0] + xgb_predicted_price[0]) / 3

# **10. Output Prediksi Harga**
print(f"Prediksi Harga LSTM: {predicted_price_lstm[0][0]}")
print(f"Prediksi Harga RF: {rf_predicted_price[0]}")
print(f"Prediksi Harga XGBoost: {xgb_predicted_price[0]}")
print(f"Final Prediksi Harga: {final_predicted_price}")
