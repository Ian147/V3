from telegram import Bot

# Masukkan TOKEN BOT dari BotFather
BOT_TOKEN = "8011128170:AAH6YRLvwzM1y9c1JgPkzNUUYT1mW_t6RlY"

# Masukkan Chat ID kamu
CHAT_ID = "681125756"

def send_signal(signal):
    """Mengirim pesan sinyal ke Telegram."""
    bot = Bot(token=BOT_TOKEN)  # Variabel bot harus ada di dalam fungsi
    bot.send_message(chat_id=CHAT_ID, text=signal)  # Kirim pesan ke Telegram

# Panggil fungsi dengan teks sebagai parameter
send_signal("🔔 Bot Telegram berhasil terkoneksi!")
