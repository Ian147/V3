import requests
from telegram import Bot

# Token dan Chat ID Telegram
BOT_TOKEN = "8011128170:AAEvCJrvMRinnIsInJmqLjzpWguz88tPWVw"
CHAT_ID = "681125756"

# Fungsi untuk mendapatkan harga BTC/USDT dari Binance
def get_price():
    url = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"
    response = requests.get(url)
    data = response.json()
    return data.get("price", "Tidak ada harga ditemukan!")

# Fungsi untuk mengirim pesan ke Telegram
def send_signal():
    bot = Bot(token=BOT_TOKEN)
    price = get_price()
    message = f"💰 Harga BTC/USDT saat ini: ${price}"
    bot.send_message(chat_id=CHAT_ID, text=message)

# Jalankan fungsi kirim sinyal
send_signal()
