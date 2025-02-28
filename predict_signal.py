import numpy as np
import pandas as pd
import tensorflow as tf
from binance.client import Client
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
import time
import os
from dotenv import load_dotenv

# Load API Key Binance dari v3.env
load_dotenv("v3.env")
API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")

# Koneksi ke Binance
client = Client(API_KEY, API_SECRET)

# Muat model LSTM
model = tf.keras.models.load_model("lstm_model.keras")

# Fungsi untuk mengambil data harga terbaru
def get_market_data(symbol="BTCUSDT", limit=50):
    klines = client.get_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_1MINUTE, limit=limit)
    df = pd.DataFrame(klines, columns=["time", "open", "high", "low", "close", "volume", "close_time",
                                       "quote_asset_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"])
    df["close"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)
    return df

# Fungsi untuk menganalisis sinyal
def analyze_market():
    df = get_market_data()
    last_prices = df["close"].values[-50:]
    volume = df["volume"].values[-50:]

    # **1. Prediksi LSTM**
    scaler = StandardScaler()
    input_data = scaler.fit_transform(last_prices.reshape(-1, 1))
    input_data = np.expand_dims(input_data, axis=0)
    predicted_price_lstm = model.predict(input_data)
    predicted_price_lstm = scaler.inverse_transform(predicted_price_lstm.reshape(-1, 1))[0][0]

    # **2. Prediksi Random Forest & XGBoost**
    X_train = np.random.rand(100, 10)
    y_train = np.random.rand(100) * 50000 + 30000
    X_test = np.random.rand(1, 10)

    scaler_ml = StandardScaler()
    X_train_scaled = scaler_ml.fit_transform(X_train)
    X_test_scaled = scaler_ml.transform(X_test)

    # Random Forest
    rf_model = RandomForestRegressor(n_estimators=100)
    rf_model.fit(X_train_scaled, y_train)
    rf_predicted_price = rf_model.predict(X_test_scaled)[0]

    # XGBoost
    xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
    xgb_model.fit(X_train_scaled, y_train)
    xgb_predicted_price = xgb_model.predict(X_test_scaled)[0]

    # **3. Final Prediksi Harga**
    final_predicted_price = (predicted_price_lstm + rf_predicted_price + xgb_predicted_price) / 3

    # **4. Analisis Sinyal (Buy, Sell, Hold)**
    current_price = df["close"].values[-1]
    change = (final_predicted_price - current_price) / current_price * 100

    if change > 1.5:
        signal = "BUY ✅"
        market_status = "Market Bullish 🚀 (Harga diprediksi naik)"
    elif change < -1.5:
        signal = "SELL ❌"
        market_status = "Market Bearish 📉 (Harga diprediksi turun)"
    else:
        signal = "HOLD ⏳"
        market_status = "Market Sideways 🔄 (Belum ada pergerakan signifikan)"

    # **5. Output Hasil Analisis**
    print(f"-----------------------------------")
    print(f"📌 Harga Saat Ini: {current_price:.2f} USDT")
    print(f"🔮 Prediksi Harga: {final_predicted_price:.2f} USDT")
    print(f"📊 Perubahan: {change:.2f}%")
    print(f"📢 Sinyal Trading: {signal}")
    print(f"📜 Keterangan: {market_status}")
    print(f"-----------------------------------\n")

# **Loop Otomatis 24/7**
while True:
    try:
        analyze_market()
        time.sleep(60)  # Tunggu 1 menit sebelum update berikutnya
    except Exception as e:
        print(f"⚠️ ERROR: {e}")
        time.sleep(60)  # Coba lagi setelah 1 menit jika terjadi error
