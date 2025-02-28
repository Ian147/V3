import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from binance.client import Client
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
import requests

# API Key tai (GANTI dengan API Key Anda)
API_KEY = "JryFhqvCFpXLzHtinwqY4zzRqNZhX9uUvRme0fpJ5In6tj4tv2HAGEnZcpFdWDuH"
API_SECRET = "ZFrepyWy3QtgfCN0V20KgkMt0zQ8uUsIoLUYeGowGUi6MAPvJHQsIghnhkutDAaj"

# API Telegram (GANTI dengan Token Bot dan Chat ID Anda)
TELEGRAM_BOT_TOKEN = "8011128170:AAEvCJrvMRinnIsInJmqLjzpWguz88tPWVw"
TELEGRAM_CHAT_ID = "681125756"

# Koneksi ke Binance
client = Client(API_KEY, API_SECRET)

def get_binance_data(symbol="BTCUSDT", interval=Client.KLINE_INTERVAL_1MINUTE, limit=50):
    """Mengambil data harga dari Binance."""
    klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(klines, columns=["time", "open", "high", "low", "close", "volume", "close_time",
                                       "quote_asset_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"])
    df["close"] = df["close"].astype(float)
    return df

# Load Model LSTM
try:
    model = tf.keras.models.load_model("lstm_model.keras")
except Exception as e:
    print(f"⚠️ ERROR: Model LSTM gagal dimuat - {e}")
    model = None

def predict_price(df):
    """Memprediksi harga menggunakan model LSTM."""
    try:
        last_prices = df["close"].values[-10:]
        scaler = StandardScaler()
        input_data = scaler.fit_transform(last_prices.reshape(-1, 1))
        input_data = np.expand_dims(input_data, axis=0)

        predicted_price_lstm = model.predict(input_data)
        predicted_price_lstm = scaler.inverse_transform(predicted_price_lstm.reshape(-1, 1))

        return predicted_price_lstm[0][0]
    except Exception as e:
        print(f"⚠️ ERROR: Prediksi LSTM gagal - {e}")
        return None

def train_ml_models():
    """Melatih model Random Forest dan XGBoost secara dummy (random data)."""
    X_train = np.random.rand(100, 10)
    y_train = np.random.rand(100) * 50000 + 30000
    X_test = np.random.rand(1, 10)

    scaler_ml = StandardScaler()
    X_train_scaled = scaler_ml.fit_transform(X_train)
    X_test_scaled = scaler_ml.transform(X_test)

    rf_model = RandomForestRegressor(n_estimators=100)
    rf_model.fit(X_train_scaled, y_train)
    rf_predicted_price = rf_model.predict(X_test_scaled)

    xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
    xgb_model.fit(X_train_scaled, y_train)
    xgb_predicted_price = xgb_model.predict(X_test_scaled)

    return rf_predicted_price[0], xgb_predicted_price[0]

def generate_signal(current_price, predicted_price):
    """Menentukan sinyal BUY, SELL, atau HOLD berdasarkan prediksi harga."""
    if predicted_price > current_price * 1.01:
        return "📈 BUY"
    elif predicted_price < current_price * 0.99:
        return "📉 SELL"
    else:
        return "⚖️ HOLD"

def send_telegram_message(message):
    """Mengirim pesan ke Telegram."""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        requests.post(url, data=data)
    except Exception as e:
        print(f"⚠️ ERROR: Gagal mengirim pesan ke Telegram - {e}")

def main():
    """Fungsi utama untuk menjalankan analisis market setiap 1 menit."""
    while True:
        try:
            df = get_binance_data()
            current_price = df["close"].values[-1]
            
            predicted_price_lstm = predict_price(df) if model else None
            predicted_price_rf, predicted_price_xgb = train_ml_models()
            
            if predicted_price_lstm is None:
                final_predicted_price = (predicted_price_rf + predicted_price_xgb) / 2
            else:
                final_predicted_price = (predicted_price_lstm + predicted_price_rf + predicted_price_xgb) / 3
            
            signal = generate_signal(current_price, final_predicted_price)

            message = (
                f"📊 **Market Update - BTC/USDT**\n"
                f"🔹 Harga Saat Ini: ${current_price:.2f}\n"
                f"🔹 Prediksi Harga: ${final_predicted_price:.2f}\n"
                f"🔹 Sinyal: {signal}\n"
                f"⏳ (Update otomatis setiap 1 menit)"
            )

            print(message)
            send_telegram_message(message)

        except Exception as e:
            print(f"⚠️ ERROR: {e}")

        time.sleep(60)  # Tunggu 1 menit sebelum menjalankan ulang

if __name__ == "__main__":
    main()
