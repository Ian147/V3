import requests

BINANCE_URL = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"

def get_price():
    """Mengambil harga BTC/USDT dari Binance dan menampilkan respon lengkap jika ada error."""
    response = requests.get(BINANCE_URL)
    data = response.json()
    print("Respon dari Binance:", data)  # Tambahkan ini untuk debugging
    return data.get("price", "Tidak ada harga ditemukan!")

if __name__ == "__main__":
    price = get_price()
    print(f"Harga BTC/USDT saat ini: ${price}")
