# === Automatische Paketinstallation falls notwendig ===
import importlib
import subprocess
import sys

def install_if_missing(package_name, pip_name=None):
    try:
        importlib.import_module(package_name)
        print(f"{package_name} bereits installiert.")
    except ImportError:
        print(f"{package_name} wird installiert...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name or package_name])

install_if_missing('yfinance')
install_if_missing('pandas')
install_if_missing('numpy')
install_if_missing('sklearn', 'scikit-learn')
install_if_missing('imbalanced_learn', 'imbalanced-learn')
install_if_missing('matplotlib')

# === Imports ===
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import timedelta, datetime
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
import time

# === Parameter ===
SYMBOL = 'BTC-USD'
PERIOD = '30d'
INTERVAL = '5m'
START_CAPITAL = 100000
TRADE_FEE_RATE = 0.001
STOP_LOSS_PCT = 0.05
TAKE_PROFIT_PCT = 0.008
TRAIL_STOP_PCT = 0.02
FUTURE_WINDOW = 24
BUY_PROB_THRESHOLD = 0.9
SHIFT_DAYS = 0
LOG_KEIN_EINSTIEG = False
PROGNOSE_ACTIVE = False
LIVE_MODE = True  # Neuer Parameter fÃ¼r Live-Modus

# === Funktionen ===
def calc_wallet(end_capital, total_fees):
    return end_capital - total_fees

def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def load_and_prepare_data():
    df = yf.download(SYMBOL, period=PERIOD, interval=INTERVAL, progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.dropna(inplace=True)
    df['returns'] = df['Close'].pct_change()
    df['volatility'] = df['returns'].rolling(window=5).std()
    df['rsi'] = compute_rsi(df['Close'])
    df['future_max'] = df['High'].rolling(window=FUTURE_WINDOW).max().shift(-FUTURE_WINDOW)
    df.dropna(subset=['future_max'], inplace=True)
    df['target'] = (df['future_max'] > df['Close'] * (1 + TAKE_PROFIT_PCT)).astype(int)
    df.dropna(subset=['returns', 'volatility', 'rsi'], inplace=True)
    return df

def predict_next_candle(model, scaler, last_data):
    features = ['returns', 'volatility', 'rsi']
    X_last = last_data[features].values.reshape(1, -1)
    X_last_scaled = scaler.transform(X_last)
    prob = model.predict_proba(X_last_scaled)[:, 1][0]
    pred = model.predict(X_last_scaled)[0]
    return prob, pred

# === Training ===
df = load_and_prepare_data()
start_date = df.index[0] + timedelta(days=SHIFT_DAYS)
df = df[df.index >= start_date]
features = ['returns', 'volatility', 'rsi']
X = df[features]
y = df['target']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, shuffle=False, test_size=0.3)
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
model = HistGradientBoostingClassifier(max_iter=100, random_state=42)
model.fit(X_resampled, y_resampled)
print("\n=== Klassifikationsbericht ===")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# === Live-Modus ===
if LIVE_MODE:
    print("\n=== Live Trading aktiv ===")
    last_seen_time = None
    capital = START_CAPITAL
    btc_amount = 0
    position = 0
    entry_price = 0
    total_fees_paid = 0.0
    highest_price_since_entry = 0

    while True:
        now = datetime.utcnow()
        if now.minute % 5 == 0 and (last_seen_time is None or now != last_seen_time):
            last_seen_time = now
            print("Neue Kerze erkannt. Warte 2 Minuten zur Stabilisierung...")
            time.sleep(120)

            try:
                live_df = yf.download(SYMBOL, period='1d', interval='5m', progress=False)
                live_df['returns'] = live_df['Close'].pct_change()
                live_df['volatility'] = live_df['returns'].rolling(window=5).std()
                live_df['rsi'] = compute_rsi(live_df['Close'])
                live_df.dropna(inplace=True)
                last_row = live_df.iloc[-1]

                prob, pred = predict_next_candle(model, scaler, last_row)
                close_price = last_row['Close']
                print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] Preis: {close_price:.2f} | Prob: {prob:.2f} | Signal: {pred}")

                if position == 0 and pred == 1 and prob > BUY_PROB_THRESHOLD:
                    fee_buy = capital * TRADE_FEE_RATE
                    total_fees_paid += fee_buy
                    btc_amount = (capital - fee_buy) / close_price
                    capital -= fee_buy
                    entry_price = close_price
                    highest_price_since_entry = close_price
                    position = 1
                    print(f"ðŸŸ Einstieg bei {close_price:.2f} USDT")

                elif position == 1:
                    if close_price > highest_price_since_entry:
                        highest_price_since_entry = close_price
                    fee_sell = btc_amount * close_price * TRADE_FEE_RATE

                    if close_price <= entry_price * (1 - STOP_LOSS_PCT):
                        pnl = (close_price - entry_price) * btc_amount - fee_buy - fee_sell
                        capital = btc_amount * close_price - fee_sell
                        total_fees_paid += fee_sell
                        print(f"ðŸ” Stop Loss bei {close_price:.2f} USDT | PnL: {pnl:.2f}")
                        position = 0
                        btc_amount = 0

                    elif close_price >= entry_price * (1 + TAKE_PROFIT_PCT):
                        pnl = (close_price - entry_price) * btc_amount - fee_buy - fee_sell
                        capital = btc_amount * close_price - fee_sell
                        total_fees_paid += fee_sell
                        print(f"ðŸŸ Take Profit bei {close_price:.2f} USDT | PnL: {pnl:.2f}")
                        position = 0
                        btc_amount = 0

                    elif close_price <= highest_price_since_entry * (1 - TRAIL_STOP_PCT):
                        pnl = (close_price - entry_price) * btc_amount - fee_buy - fee_sell
                        capital = btc_amount * close_price - fee_sell
                        total_fees_paid += fee_sell
                        print(f"â³ Trail Stop bei {close_price:.2f} USDT | PnL: {pnl:.2f}")
                        position = 0
                        btc_amount = 0

            except Exception as e:
                print(f"Fehler beim Abrufen der Live-Daten: {e}")

        else:
            time.sleep(10)

else:
    print("Live-Modus ist deaktiviert.":q!
