import asyncio
from telegram import Bot

# Masukkan TOKEN BOT dari BotFather
BOT_TOKEN = "8011128170:AAEvCJrvMRinnIsInJmqLjzpWguz88tPWVw"

# Masukkan Chat ID kamu
CHAT_ID = "681125756"

async def send_signal(signal):
    """Mengirim pesan sinyal ke Telegram secara async."""
    bot = Bot(token=BOT_TOKEN)
    await bot.send_message(chat_id=CHAT_ID, text=signal)

async def main():
    """Fungsi utama untuk menjalankan bot."""
    signal = "📢 Sinyal terbaru dari bot!"
    await send_signal(signal)

# Jalankan event loop
asyncio.run(main())
