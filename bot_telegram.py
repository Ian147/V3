from telegram import Bot
import pandas as pd
import numpy as np

# Masukkan TOKEN BOT dari BotFather
BOT_TOKEN = "8011128170:AAEvCJrvMRinnIsInJmqLjzpWguz88tPWVw"

# Masukkan Chat ID kamu
CHAT_ID = "681125756"

def send_signal(signal):
    """Mengirim pesan sinyal ke Telegram."""
    bot = Bot(token=BOT_TOKEN)
    bot.send_message(chat_id=CHAT_ID, text=signal)

def generate_signal():
    """Contoh fungsi analisis sinyal (nanti bisa diganti dengan logika AI)."""
    # Simulasi data harga (bisa diganti dengan data real-time)
    prices = np.random.randn(10) * 10 + 100  # 10 harga acak sekitar 100
    df = pd.DataFrame(prices, columns=["Price"])

    # Logika sederhana: Jika harga terakhir naik dibanding sebelumnya, kirim sinyal beli
    if df["Price"].iloc[-1] > df["Price"].iloc[-2]:
        return "📈 Sinyal Beli: Harga naik!"
    else:
        return "📉 Sinyal Jual: Harga turun!"

# Jalankan analisis dan kirim sinyal
signal = generate_signal()
send_signal(signal)
