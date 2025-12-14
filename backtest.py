import yfinance as yf
import pandas as pd
# import pandas_ta as ta # Removed due to install errors
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import numpy as np
import requests
import os

# --- Telegram Configuration ---
# 1. Create a bot with @BotFather on Telegram to get the token.
# 2. Send a message to your bot and then visit https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates to find your "id" (chat_id).
# Default to environment variables (for GitHub Actions), fallback to hardcoded strings for local testing.
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "8294907161:AAE_2zZ8DzdpB2-wN5Z3nQH8kBgi24B3_0w")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "259928123")
SYMBOLS = ["SPY", "TQQQ", "TECL", "SPXL", "UVXY", "SQQQ", "BSV"]
START_DATE = "2010-01-01" # Adjust as needed
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

def run_backtest():
    # 1. Fetch Data
    df_prices = fetch_data(SYMBOLS, START_DATE, END_DATE)
    df_prices = df_prices.dropna() # Drop initial NaNs if any, though ffill helps.
    
    # 2. Indicators
    # We need to make sure we have RSIs for SQQQ and BSV as used in the logic
    indicators_df = calculate_indicators(df_prices)
    indicators_df['SQQQ_RSI_10'] = calculate_rsi(df_prices['SQQQ'], 10)
    indicators_df['BSV_RSI_10'] = calculate_rsi(df_prices['BSV'], 10)
    
    # Align Dataframes
    # Indicators will have NaNs at the start (first 200 days for SMA 200). We must drop these rows.
    combined = pd.concat([df_prices, indicators_df], axis=1).dropna()
    
    # 3. Simulate
    print("Running simulation...")
    portfolio_values = [10000.0] # Start with $10k
    allocations = []
    
    # Iterate through days. 
    # Logic: On Day T (close), we calculate signal. We buy Asset X.
    # We hold Asset X from Day T Close to Day T+1 Close.
    # Return = (Price_Asset_T+1 / Price_Asset_T) - 1
    
    dates = combined.index
    
    for i in range(len(dates) - 1):
        current_date = dates[i]
        next_date = dates[i+1]
        
        row_prices = combined.loc[current_date]
        row_indicators = combined.loc[current_date] # Use same row for indicators
        
        # Determine target asset
        target_asset = select_asset(row_prices, row_indicators)
        allocations.append(target_asset)
        
        # Calculate Return for NEXT day
        start_price = row_prices[target_asset] # Price at Close of Signal Day
        end_price = combined.loc[next_date, target_asset] # Price at Close of Next Day
        
        daily_return = (end_price - start_price) / start_price
        
        new_value = portfolio_values[-1] * (1 + daily_return)
        portfolio_values.append(new_value)
        
    # Append last allocation placeholder
    allocations.append(allocations[-1]) 
        
    # 4. Results
    results_df = pd.DataFrame({
        'Portfolio': portfolio_values,
        'Allocation': allocations
    }, index=dates)
    
    # Benchmark (Buy and Hold SPY)
    spy_start = combined.loc[dates[0], 'SPY']
    results_df['SPY_Benchmark'] = (combined['SPY'] / spy_start) * 10000.0
    
    # Stats
    total_return = (results_df['Portfolio'].iloc[-1] / results_df['Portfolio'].iloc[0]) - 1
    cagr = (results_df['Portfolio'].iloc[-1] / results_df['Portfolio'].iloc[0]) ** (252 / len(results_df)) - 1
    
    # Max Drawdown
    rolling_max = results_df['Portfolio'].cummax()
    drawdown = (results_df['Portfolio'] - rolling_max) / rolling_max
    max_dd = drawdown.min()
    
    print("-" * 30)
    print(f"Backtest Results ({dates[0].date()} to {dates[-1].date()})")
    print("-" * 30)
    print(f"Total Return: {total_return:.2%}")
    print(f"CAGR:         {cagr:.2%}")
    print(f"Max Drawdown: {max_dd:.2%}")
    print("-" * 30)
    
    # Identify trades
    # A trade occurs when the allocation changes from the previous day
    # shift(1) gives the previous day's allocation. 
    # We treat the first day as a trade (entry).
    trades_mask = results_df['Allocation'] != results_df['Allocation'].shift(1)
    trade_dates = results_df.index[trades_mask]
    trade_values = results_df.loc[trade_dates, 'Portfolio']

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(results_df.index, results_df['Portfolio'], label='Strategy')
    plt.plot(results_df.index, results_df['SPY_Benchmark'], label='SPY Benchmark', alpha=0.7)
    
    # Plot Trade Markers
    plt.scatter(trade_dates, trade_values, marker='^', color='black', s=30, label='Trade', zorder=5)

    plt.yscale('log')
    plt.title("TQQQ Long Term V2 vs SPY")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig('backtest_results.png')
    print("Chart saved to backtest_results.png")
    
    # 5. Latest Signal
    print("\n" + "=" * 30)
    print(" LATEST TRADING SIGNAL")
    print("=" * 30)
    
    last_date = combined.index[-1]
    latest_row_prices = combined.loc[last_date]
    latest_row_indicators = combined.loc[last_date]
    
    recommended_asset = select_asset(latest_row_prices, latest_row_indicators)
    
    print(f"Date: {last_date.date()}")
    print(f"Current Allocation Recommendation: {recommended_asset}")
    
    # Check if we need to trade
    # We compare with the previous day's allocation in the backtest results
    # The 'Allocation' column in results_df represents what we held FROM that day TO the next.
    # So results_df.iloc[-1]['Allocation'] is what we decided to hold on the last day of the simulation.
    # Wait, let's be careful. 
    # Logic in loop: 
    # target_asset = select_asset(row_prices, row_indicators) -> decided on 'current_date' close.
    # So for the last date in 'combined', we just calculated 'recommended_asset'.
    # This is what we should hold starting from 'last_date' close until the next day.
    
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
    run_backtest()
