from telegram import Bot
from telegram.ext import Updater, CommandHandler
import talib

# Konfigurasi bot Telegram
TOKEN = '8011128170:AAEvCJrvMRinnIsInJmqLjzpWguz88tPWVw'
CHAT_ID = '681125756'
bot = Bot(TOKEN)

# Fungsi untuk mendapatkan sinyal dari model
def get_signal():
    ohlcv = exchange.fetch_ohlcv('BTC/USDT', '1h')
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['SMA_50'] = talib.SMA(df['close'], timeperiod=50)
    df['RSI'] = talib.RSI(df['close'], timeperiod=14)
    df = df.dropna()

    X = df[['SMA_50', 'RSI']]
    prediction = model.predict(X.iloc[-1:])  # Prediksi menggunakan data terakhir
    return "Buy Signal!" if prediction[0] else "Sell Signal!"

# Fungsi untuk mengirim sinyal ke Telegram
def send_signal(update, context):
    signal = get_signal()
    bot.send_message(chat_id=CHAT_ID, text=signal)

# Setup Updater dan Handler
updater = Updater(TOKEN, use_context=True)
dp = updater.dispatcher

dp.add_handler(CommandHandler('signal', send_signal))

# Menjalankan bot
updater.start_polling()
updater.idle()
