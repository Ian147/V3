import ccxt
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Setup Exchange dengan API
exchange = ccxt.binance({
    'apiKey': 'YOUR_API_KEY',
    'secret': 'YOUR_SECRET_KEY',
})

# Fungsi untuk mendapatkan data OHLCV (Open, High, Low, Close, Volume)
def fetch_data(symbol, timeframe='1h', limit=200):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    data = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    return data

# Fungsi untuk menghitung indikator teknikal
def calculate_indicators(data):
    # Menghitung RSI
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))

    # Menghitung Moving Average (MA)
    data['SMA'] = data['close'].rolling(window=20).mean()
    
    # Menghitung MACD
    short_ema = data['close'].ewm(span=12, adjust=False).mean()
    long_ema = data['close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = short_ema - long_ema
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
    
    # Menghitung target (apakah harga naik atau turun)
    data['Target'] = np.where(data['close'].shift(-1) > data['close'], 1, 0)  # 1 untuk naik, 0 untuk turun
    return data.dropna()

# Fungsi untuk melatih model AI (XGBoost)
def train_model(data):
    features = ['RSI', 'SMA', 'MACD', 'Signal_Line']
    X = data[features]
    y = data['Target']
    
    # Normalisasi fitur
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Membagi data untuk pelatihan dan pengujian
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Latih model XGBoost
    model = XGBClassifier()
    model.fit(X_train, y_train)
    
    # Evaluasi model
    accuracy = model.score(X_test, y_test)
    print(f'Model Accuracy: {accuracy:.2f}')
    return model, scaler

# Fungsi untuk membuat prediksi menggunakan model yang dilatih
def predict_signal(model, scaler, data):
    features = ['RSI', 'SMA', 'MACD', 'Signal_Line']
    X = data[features]
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)
    return prediction[-1]  # Mengembalikan sinyal terakhir

# Fungsi utama untuk menjalankan bot analisis sinyal dengan AI
def run_bot(symbol, model, scaler):
    data = fetch_data(symbol)
    data = calculate_indicators(data)
    
    signal = predict_signal(model, scaler, data)
    
    if signal == 1:
        print("Sinyal Buy: Harga diprediksi akan naik.")
    elif signal == 0:
        print("Sinyal Sell: Harga diprediksi akan turun.")
    else:
        print("Tidak ada sinyal yang jelas.")
        
# Pelatihan model dan scaler
symbol = 'BTC/USDT'
data = fetch_data(symbol)
data = calculate_indicators(data)
model, scaler = train_model(data)

# Menjalankan bot secara berkala
while True:
    run_bot(symbol, model, scaler)
    time.sleep(60)  # Cek setiap menit
