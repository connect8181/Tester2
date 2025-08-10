
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
LIVE_MODE = True
LIVE_INTERVAL = '5m'

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

def live_predict(model, scaler, features):
    print("\n=== LIVE-MODUS MIT TRADING-AKTIONEN AKTIV ===")
    trade_counter = 1
    capital = START_CAPITAL
    btc_amount = 0
    position = 0
    entry_price = 0
    highest_price_since_entry = 0
    total_fees_paid = 0.0

    while True:
        now = datetime.now()
        live_data = yf.download(SYMBOL, period='2d', interval=LIVE_INTERVAL, progress=False, auto_adjust=False).copy()

        if live_data.empty:
            print("\u26a0\ufe0f Keine Daten geladen.")
            time.sleep(300)
            continue

        if isinstance(live_data.columns, pd.MultiIndex):
            live_data.columns = live_data.columns.get_level_values(0)
        live_data.dropna(inplace=True)
        live_data['returns'] = live_data['Close'].pct_change()
        live_data['volatility'] = live_data['returns'].rolling(window=5).std()
        live_data['rsi'] = compute_rsi(live_data['Close'])
        live_data.dropna(subset=['returns', 'volatility', 'rsi'], inplace=True)

        if live_data.empty:
            print("\u26a0\ufe0f Nach Feature-Berechnung keine Daten übrig.")
            time.sleep(300)
            continue

        last = live_data.iloc[-1:]
        X_live = scaler.transform(last[features])
        prob_live = model.predict_proba(X_live)[0, 1]
        close_price = last['Close'].values[0]
        rsi_val = float(last['rsi'].values[0])
        vol_val = float(last['volatility'].values[0])

        print(f"\n{trade_counter:03d} \U0001f535 LIVE-PROGNOSE @ {now.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Preis: {close_price:.2f} | RSI: {rsi_val:.1f} | Vol: {vol_val:.5f} | Buy-Wahrscheinlichkeit: {prob_live:.2f}")

        if position == 0 and prob_live > BUY_PROB_THRESHOLD:
            fee_buy = capital * TRADE_FEE_RATE
            total_fees_paid += fee_buy
            btc_amount = (capital - fee_buy) / close_price
            capital -= fee_buy
            entry_price = close_price
            highest_price_since_entry = close_price
            position = 1
            print(f"{trade_counter:03d} \U0001f7e2 LIVE-KAUF @ {close_price:.2f} | Gebühr: {fee_buy:.2f}")
            trade_counter += 1

        elif position == 1:
            if close_price > highest_price_since_entry:
                highest_price_since_entry = close_price

            fee_sell = btc_amount * close_price * TRADE_FEE_RATE
            # Exit: STOP LOSS
            if close_price <= entry_price * (1 - STOP_LOSS_PCT):
                pnl = (close_price - entry_price) * btc_amount - fee_buy - fee_sell
                total_fees_paid += fee_sell
                capital = btc_amount * close_price - fee_sell
                btc_amount = 0
                position = 0
                print(f"{trade_counter:03d} \U0001f53b STOP LOSS @ {close_price:.2f} | Gewinn/Verlust: {pnl:.2f}")
                trade_counter += 1

            # Exit: TAKE PROFIT
            elif close_price >= entry_price * (1 + TAKE_PROFIT_PCT):
                pnl = (close_price - entry_price) * btc_amount - fee_buy - fee_sell
                total_fees_paid += fee_sell
                capital = btc_amount * close_price - fee_sell
                btc_amount = 0
                position = 0
                print(f"{trade_counter:03d} \U0001f7e2 TAKE PROFIT @ {close_price:.2f} | Gewinn/Verlust: {pnl:.2f}")
                trade_counter += 1

            # Exit: TRAILING STOP
            elif (close_price <= highest_price_since_entry * (1 - TRAIL_STOP_PCT) and
                  close_price >= entry_price + (fee_buy + fee_sell) / btc_amount):
                pnl = (close_price - entry_price) * btc_amount - fee_buy - fee_sell
                total_fees_paid += fee_sell
                capital = btc_amount * close_price - fee_sell
                btc_amount = 0
                position = 0
                print(f"{trade_counter:03d} \u23f3 TRAIL STOP @ {close_price:.2f} | Gewinn/Verlust: {pnl:.2f}")
                trade_counter += 1

        print(f"Wallet: {capital - total_fees_paid:.2f} USDT (Netto)")
        time.sleep(300)  # 5 Minuten Pause

# === Hauptprogramm ===
if __name__ == "__main__":
    df = load_and_prepare_data()
    start_date = df.index[0] + timedelta(days=SHIFT_DAYS)
    df = df[df.index >= start_date]
    print("Zielverteilung (target):")
    print(df['target'].value_counts(normalize=True))

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

    if LIVE_MODE:
        live_predict(model, scaler, features)


