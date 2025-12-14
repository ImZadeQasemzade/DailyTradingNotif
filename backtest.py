import yfinance as yf
import pandas as pd
import numpy as np
import requests
import os

# --- Telegram Configuration ---
# 1. Create a bot with @BotFather on Telegram to get the token.
# 2. Send a message to your bot and then visit https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates to find your "id" (chat_id).
# Default to environment variables (for GitHub Actions), fallback to hardcoded strings for local testing.
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
SYMBOLS = ["SPY", "TQQQ", "TECL", "SPXL", "UVXY", "SQQQ", "BSV"]
START_DATE = "2020-01-01" # Adjust as needed
END_DATE = None # Today

def calculate_sma(series, window):
    return series.rolling(window=window).mean()

def calculate_rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0))
    loss = (-delta.where(delta < 0, 0))
    
    # Wilder's Smoothing (alpha = 1/n)
    # The first value is usually SMA. 
    # Pandas ewm with com=(window-1) matches Wilder's smoothing roughly.
    avg_gain = gain.ewm(com=window-1, min_periods=window).mean()
    avg_loss = loss.ewm(com=window-1, min_periods=window).mean()
    
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def fetch_data(symbols, start_date, end_date):
    print(f"Fetching data for {symbols}...")
    data = yf.download(symbols, start=start_date, end=end_date, progress=False)
    # yfinance returns a multi-index columns (Price, Ticker). We need 'Adj Close' or 'Close'.
    # Using 'Adj Close' is better for long term returns.
    try:
        df = data['Adj Close']
    except KeyError:
        df = data['Close']
    
    # Forward fill missing data (e.g. holidays or slight mismatches)
    df = df.ffill()
    return df

def calculate_indicators(df):
    print("Calculating indicators...")
    indicators = {}
    
    # SPY SMA 200
    indicators['SPY_SMA_200'] = calculate_sma(df['SPY'], 200)
    
    # TQQQ SMA 20
    indicators['TQQQ_SMA_20'] = calculate_sma(df['TQQQ'], 20)
    
    # RSI 10 for TQQQ, SPXL, SPY, UVXY
    indicators['TQQQ_RSI_10'] = calculate_rsi(df['TQQQ'], 10)
    indicators['SPXL_RSI_10'] = calculate_rsi(df['SPXL'], 10)
    indicators['SPY_RSI_10'] = calculate_rsi(df['SPY'], 10)
    indicators['UVXY_RSI_10'] = calculate_rsi(df['UVXY'], 10)
    
    return pd.DataFrame(indicators, index=df.index)

def select_asset(row, indicators_row):
    # Extract scalar values for readability
    spy_price = row['SPY']
    spy_sma_200 = indicators_row['SPY_SMA_200']
    
    tqqq_price = row['TQQQ']
    tqqq_sma_20 = indicators_row['TQQQ_SMA_20']
    
    tqqq_rsi_10 = indicators_row['TQQQ_RSI_10']
    spxl_rsi_10 = indicators_row['SPXL_RSI_10']
    spy_rsi_10 = indicators_row['SPY_RSI_10']
    uvxy_rsi_10 = indicators_row['UVXY_RSI_10']
    
    # --- Logic Tree ---
    
    # Root: SPY > SPY SMA 200?
    if spy_price > spy_sma_200:
        # Bull Market Branch
        
        # Check TQQQ RSI
        if tqqq_rsi_10 > 79:
            return "UVXY"
        else:
            # Check SPXL RSI
            if spxl_rsi_10 > 80:
                return "UVXY"
            else:
                return "TQQQ"
    else:
        # Bear/Correction Branch
        
        # Check TQQQ RSI
        if tqqq_rsi_10 < 31:
            return "TECL"
        else:
            # Check SPY RSI
            if spy_rsi_10 < 30:
                return "SPXL"
            else:
                # Check UVXY RSI
                if uvxy_rsi_10 > 74:
                    # High Volatility
                    if uvxy_rsi_10 > 84:
                        # Extreme Volatility
                        if tqqq_price > tqqq_sma_20:
                            return "TQQQ"
                        else:
                            # Filter: Select top 1 by RSI 10 between SQQQ and BSV
                            return select_top_rsi(["SQQQ", "BSV"], indicators_row)
                    else:
                        return "UVXY"
                else:
                    # Low/Moderate Volatility in Downtrend
                    if tqqq_price > tqqq_sma_20:
                        return "TQQQ"
                    else:
                        return select_top_rsi(["SQQQ", "BSV"], indicators_row)

def select_top_rsi(assets, indicators_row):
    # Helper to pick asset with highest RSI
    # We need to ensure we have these RSIs in indicators_row
    best_asset = None
    best_rsi = -1
    
    for asset in assets:
        rsi_key = f"{asset}_RSI_10"
        if rsi_key in indicators_row:
            val = indicators_row[rsi_key]
            if val > best_rsi:
                best_rsi = val
                best_asset = asset
    return best_asset

def send_telegram_message(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram configuration missing. Skipping message.")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        print("Telegram notification sent!")
    except Exception as e:
        print(f"Failed to send Telegram message: {e}")


def run_daily_signal():
    # 1. Fetch Data (Optimized for signal generation, just need enough data for indicators)
    # We need at least 200 days for SMA200 + some buffer for lookback
    print("Fetching recent data for signal generation...")
    
    # Fetch last 365 days to be safe for 200 SMA
    start_date_signal = (pd.Timestamp.now() - pd.DateOffset(days=400)).strftime('%Y-%m-%d')
    df_prices = fetch_data(SYMBOLS, start_date_signal, None)
    df_prices = df_prices.dropna()
    
    # 2. Indicators
    indicators_df = calculate_indicators(df_prices)
    indicators_df['SQQQ_RSI_10'] = calculate_rsi(df_prices['SQQQ'], 10)
    indicators_df['BSV_RSI_10'] = calculate_rsi(df_prices['BSV'], 10)
    
    # Align Dataframes
    combined = pd.concat([df_prices, indicators_df], axis=1).dropna()
    
    if combined.empty:
        print("Error: Not enough data to calculate indicators.")
        return

    # 3. Latest Signal
    print("\n" + "=" * 30)
    print(" LATEST TRADING SIGNAL")
    print("=" * 30)
    
    last_date = combined.index[-1]
    latest_row_prices = combined.loc[last_date]
    latest_row_indicators = combined.loc[last_date]
    
    recommended_asset = select_asset(latest_row_prices, latest_row_indicators)
    
    print(f"Date: {last_date.date()}")
    print(f"Current Allocation Recommendation: {recommended_asset}")
    print(f"Action: Buy/Hold {recommended_asset} at Close")
    print("=" * 30)

    # Send Notification
    msg = (
        f"🚀 *Composer Strategy Signal*\n"
        f"📅 Date: {last_date.date()}\n"
        f"📈 Recommendation: *{recommended_asset}*\n"
        f"👉 Action: Buy/Hold at Close"
    )
    send_telegram_message(msg)

if __name__ == "__main__":
    run_daily_signal()

