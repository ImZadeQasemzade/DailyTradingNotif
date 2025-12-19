import yfinance as yf
import pandas as pd
import numpy as np
import requests
import os
import matplotlib.pyplot as plt
import io
import pytz
from datetime import datetime, time

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
            return "UVXY", f"Bull Market (SPY > SMA200) but TQQQ Overbought (RSI {tqqq_rsi_10:.1f} > 79)"
        else:
            # Check SPXL RSI
            if spxl_rsi_10 > 80:
                return "UVXY", f"Bull Market (SPY > SMA200) but SPXL Overbought (RSI {spxl_rsi_10:.1f} > 80)"
            else:
                return "TQQQ", f"Bull Market (SPY > SMA200) and Momentum Healthy"
    else:
        # Bear/Correction Branch
        
        # Check TQQQ RSI
        if tqqq_rsi_10 < 31:
            return "TECL", f"Bear Market (SPY < SMA200) but TQQQ Oversold (RSI {tqqq_rsi_10:.1f} < 31)"
        else:
            # Check SPY RSI
            if spy_rsi_10 < 30:
                return "SPXL", f"Bear Market (SPY < SMA200) but SPY Oversold (RSI {spy_rsi_10:.1f} < 30)"
            else:
                # Check UVXY RSI
                if uvxy_rsi_10 > 74:
                    # High Volatility
                    if uvxy_rsi_10 > 84:
                        # Extreme Volatility
                        if tqqq_price > tqqq_sma_20:
                            return "TQQQ", f"Bear Market, Extreme Volatility (UVXY RSI {uvxy_rsi_10:.1f}), but TQQQ > SMA20"
                        else:
                            # Filter: Select top 1 by RSI 10 between SQQQ and BSV
                            asset = select_top_rsi(["SQQQ", "BSV"], indicators_row)
                            return asset, f"Bear Market, Extreme Volatility (UVXY RSI {uvxy_rsi_10:.1f}), TQQQ < SMA20. Selected {asset} by RSI."
                    else:
                        return "UVXY", f"Bear Market, High Volatility (UVXY RSI {uvxy_rsi_10:.1f} > 74)"
                else:
                    # Low/Moderate Volatility in Downtrend
                    if tqqq_price > tqqq_sma_20:
                        return "TQQQ", f"Bear Market, Moderate Volatility, but TQQQ > SMA20"
                    else:
                        asset = select_top_rsi(["SQQQ", "BSV"], indicators_row)
                        return asset, f"Bear Market (SPY < SMA200), Moderate Volatility. Selected {asset} by RSI."

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

def generate_chart(df_prices, indicators_df, recommended_asset, window_days=180):
    # Filter for the last N days to make the chart readable
    last_date = df_prices.index[-1]
    start_plot = last_date - pd.Timedelta(days=window_days)
    
    df_p = df_prices.loc[start_plot:]
    df_i = indicators_df.loc[start_plot:]
    
    # Setup Figure with 3 subplots sharing x-axis
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # Subplot 1: SPY vs SMA 200 (Market Trend)
    ax1.plot(df_p.index, df_p['SPY'], label='SPY Price', color='black', linewidth=1.5)
    ax1.plot(df_i.index, df_i['SPY_SMA_200'], label='SMA 200', color='orange', linestyle='--', linewidth=1.5)
    ax1.set_title(f"Market Trend: SPY vs SMA 200 (Signal Date: {last_date.date()})")
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: TQQQ vs SMA 20 (Momentum)
    ax2.plot(df_p.index, df_p['TQQQ'], label='TQQQ Price', color='blue', linewidth=1.5)
    ax2.plot(df_i.index, df_i['TQQQ_SMA_20'], label='SMA 20', color='cyan', linestyle='--', linewidth=1.5)
    ax2.set_title("Short-Term Momentum: TQQQ vs SMA 20")
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # Subplot 3: Oscillators (RSI)
    # We want to see TQQQ, SPXL, UVXY, SPY RSIs. 
    # Use standard colors: TQQQ=Blue, SPXL=Green, UVXY=Red, SPY=Black
    ax3.plot(df_i.index, df_i['TQQQ_RSI_10'], label='TQQQ RSI', color='blue', linewidth=1)
    ax3.plot(df_i.index, df_i['SPXL_RSI_10'], label='SPXL RSI', color='green', linewidth=1, alpha=0.7)
    ax3.plot(df_i.index, df_i['UVXY_RSI_10'], label='UVXY RSI', color='red', linewidth=1)
    
    # Determine critical levels based on logic (79/80/30/31 etc)
    ax3.axhline(79, color='red', linestyle=':', alpha=0.5, label='Overbought (79/80)')
    ax3.axhline(30, color='green', linestyle=':', alpha=0.5, label='Oversold (30/31)')
    
    ax3.set_title("Indicators: RSI 10 (Critical levels at 30 & 79)")
    ax3.set_ylabel("RSI")
    ax3.set_ylim(0, 100)
    ax3.legend(loc='upper left', ncol=3)
    ax3.grid(True, alpha=0.3)
    
    # Highlight recommendation in Title
    plt.suptitle(f"Strategy Signal: {recommended_asset}", fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    plt.close(fig)
    return buf

def send_telegram_message(message, image_file=None):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram configuration missing. Skipping message.")
        return

    if image_file:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
        files = {'photo': ('chart.png', image_file, 'image/png')}
        data = {
            "chat_id": TELEGRAM_CHAT_ID,
            "caption": message,
            "parse_mode": "Markdown"
        }
        try:
            response = requests.post(url, data=data, files=files)
            response.raise_for_status()
            print("Telegram notification (with photo) sent!")
        except Exception as e:
            print(f"Failed to send Telegram photo: {e}")
            try:
                print("Falling back to text only message...")
                # Reset stream position just in case, though not needed for text fallback
                send_telegram_message(message, image_file=None) 
            except:
                pass
    else:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "Markdown"
        }
        
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            print("Telegram notification sent!")
        except Exception as e:
            print(f"Failed to send Telegram message: {e}")


def run_daily_signal():
    # --- Time Check for DST Support ---
    # Goal: Run ONLY if current ET time is within 15:30 - 15:55 window.
    # This supports the dual cron schedule (19:45, 20:45 UTC).
    
    # Check for FORCE_RUN env var (for manual dispatch or testing)
    force_run = os.environ.get("FORCE_RUN", "false").lower() == "true"
    
    if not force_run:
        try:
            tz_et = pytz.timezone('US/Eastern')
            now_et = datetime.now(tz_et)
            current_time = now_et.time()
            
            # Target window: 15:35 to 15:55 (giving plenty of buffer for 15:45 target)
            start_window = time(15, 35)
            end_window = time(15, 55)
            
            if not (start_window <= current_time <= end_window):
                print(f"Skipping run: Current ET time {now_et.strftime('%H:%M')} is not within target window (15:35-15:55).")
                print("This allows the dual UTC schedule to handle DST correctly under Github Actions.")
                return
            else:
                print(f"Time check passed: Current ET time {now_et.strftime('%H:%M')} is within target window.")
        except Exception as e:
            print(f"Warning: Time check failed ({e}). Proceeding with caution.")

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
    
    recommended_asset, reason = select_asset(latest_row_prices, latest_row_indicators)
    
    print(f"Date: {last_date.date()}")
    print(f"Current Allocation Recommendation: {recommended_asset}")
    print(f"Reason: {reason}")
    print(f"Action: Buy/Hold {recommended_asset} at Close")
    print("=" * 30)

    # Generate Chart
    print("Generating Chart...")
    try:
        chart_buf = generate_chart(df_prices, indicators_df, recommended_asset)
    except Exception as e:
        print(f"Error generating chart: {e}")
        chart_buf = None

    # Send Notification
    msg = (
        f"🚀 *Composer Strategy Signal*\n"
        f"📅 Date: {last_date.date()}\n"
        f"📈 Recommendation: *{recommended_asset}*\n"
        f"🧠 Logic: {reason}\n"
        f"👉 Action: Buy/Hold at Close"
    )
    send_telegram_message(msg, image_file=chart_buf)

if __name__ == "__main__":
    run_daily_signal()
