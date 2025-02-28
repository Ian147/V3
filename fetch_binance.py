import requests
import pandas as pd
import pandas_ta as ta

BINANCE_URL = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"

def get_price():
    """Mengambil harga BTC/USDT dari Binance dan menampilkan respon lengkap jika ada error."""
    response = requests.get(BINANCE_URL)
    data = response.json()
    print("Respon dari Binance:", data)  # Debugging

    price = float(data.get("price", 0))
    return price

def analyze_market():
    """Menganalisis pasar menggunakan indikator teknikal."""
    prices = []

    # Simpan 50 harga terbaru untuk analisis
    for _ in range(50):
        price = get_price()
        prices.append(price)

    # Simpan data ke dalam DataFrame
    df = pd.DataFrame(prices, columns=["close"])

    # Tambahkan indikator teknikal
    df["SMA_10"] = ta.sma(df["close"], length=10)  # Simple Moving Average 10
    df["RSI_14"] = ta.rsi(df["close"], length=14)  # Relative Strength Index 14

    print(df.tail())  # Menampilkan 5 baris terakhir
    return df

if __name__ == "__main__":
    analyze_market()
