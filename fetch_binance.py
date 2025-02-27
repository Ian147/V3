import requests

# Endpoint Binance untuk harga terbaru
BINANCE_URL = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"

def get_price():
    """Mengambil harga BTC/USDT dari Binance."""
    response = requests.get(BINANCE_URL)
    data = response.json()
    return data["price"]

if __name__ == "__main__":
    price = get_price()
    print(f"Harga BTC/USDT saat ini: ${price}")
