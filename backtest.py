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
SYMBOLS = ["SPY", "TQQQ", "TECL", "SPXL", "UVXY", "SQQQ", "BSV", "GLD"]
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

def fetch_fred_data(series_id, start_date):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd={start_date}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text), index_col='observation_date', parse_dates=True, na_values='.')
        return df[series_id].ffill()
    except Exception as e:
        print(f"Error fetching FRED data for {series_id}: {e}")
        return pd.Series(dtype=float)

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

from matplotlib.gridspec import GridSpec

def generate_chart(df_prices, indicators_df, recommended_asset, window_days=180, yield_curve_history=None):
    # Filter for the last N days to make the chart readable
    last_date = df_prices.index[-1]
    start_plot = last_date - pd.Timedelta(days=window_days)
    
    df_p = df_prices.loc[start_plot:]
    df_i = indicators_df.loc[start_plot:]
    
    # Setup Figure with GridSpec so we can have 6 subplots, 
    # where 1-5 share time X-axis, and 6 has categorical X-axis.
    fig = plt.figure(figsize=(10, 22))
    gs = GridSpec(6, 1, figure=fig)
    
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    ax5 = fig.add_subplot(gs[4], sharex=ax1)
    ax6 = fig.add_subplot(gs[5])
    
    # Hide x labels and tick labels for top plots and y ticks for right plots.
    plt.setp([a.get_xticklabels() for a in [ax1, ax2, ax3, ax4]], visible=False)
    
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
    
    # Subplot 4: Macro (Yield Curve & Fed Funds Rate)
    if 'Yield_Curve' in df_p.columns and 'Fed_Funds_Rate' in df_p.columns:
        ax4.plot(df_p.index, df_p['Yield_Curve'], label='10Y-2Y Yield Curve', color='purple', linewidth=1.5)
        ax4.axhline(0, color='red', linestyle='--', alpha=0.5, label='Inversion Line (0)')
        ax4.set_ylabel("Yield Spread (%)")
        
        ax4_twin = ax4.twinx()
        ax4_twin.plot(df_p.index, df_p['Fed_Funds_Rate'], label='Fed Funds Rate', color='orange', linewidth=1.5)
        ax4_twin.set_ylabel("Rate (%)")
        ax4.set_title("Macro Variables: Yield Curve & Effective Fed Funds Rate")
        
        # Combine legends for ax4
        lines_1, labels_1 = ax4.get_legend_handles_labels()
        lines_2, labels_2 = ax4_twin.get_legend_handles_labels()
        ax4.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')
        ax4.grid(True, alpha=0.3)
    
    # Subplot 5: SPY / GLD Ratio
    if 'SPY_GLD_Ratio' in df_i.columns:
        ax5.plot(df_i.index, df_i['SPY_GLD_Ratio'], label='SPY / GLD Ratio', color='goldenrod', linewidth=1.5)
        # Add a 20-day SMA to ratio for context
        ratio_sma = df_i['SPY_GLD_Ratio'].rolling(window=20).mean()
        ax5.plot(df_i.index, ratio_sma, label='Ratio SMA 20', color='black', linestyle='--', linewidth=1)
        ax5.set_title("Risk-On vs Safe-Haven: SPY to Gold Ratio")
        ax5.set_ylabel("Ratio")
        ax5.legend(loc='upper left')
        ax5.grid(True, alpha=0.3)

    # Subplot 6: Full Yield Curve Snapshot
    if yield_curve_history is not None and not yield_curve_history.empty:
        mats = list(yield_curve_history.columns)
        
        # Plot today's curve
        latest_series = yield_curve_history.iloc[-1]
        yields = list(latest_series.values)
        ax6.plot(mats, yields, marker='o', color='crimson', linewidth=2.5, label='Current')
        
        # Highlight inverts or normal curve shape visually for current
        ax6.fill_between(mats, 0, yields, color='red', alpha=0.1)
        for i, txt in enumerate(yields):
            ax6.annotate(f"{txt:.2f}%", (mats[i], yields[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9, color='crimson')
            
        # Plot older curves
        # 1 month is roughly 21 trading days
        offsets = [(21, '1M Ago', 0.6), (42, '2M Ago', 0.4), (63, '3M Ago', 0.2)]
        for days_back, label, alpha in offsets:
            if len(yield_curve_history) > days_back:
                old_series = yield_curve_history.iloc[-(days_back + 1)]
                old_yields = list(old_series.values)
                ax6.plot(mats, old_yields, linestyle='--', color='purple', linewidth=1.5, alpha=alpha, label=label)

        ax6.set_title(f"US Treasury Yield Curve Evolution")
        ax6.set_ylabel("Yield (%)")
        ax6.set_xlabel("Maturity")
        ax6.set_ylim(yield_curve_history.min().min() - 0.5, yield_curve_history.max().max() + 0.5)
        ax6.legend(loc='lower center', ncol=4)
        ax6.grid(True, alpha=0.3)

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
    # Time check removed to allow running anytime.


    # 1. Fetch Data (Optimized for signal generation, just need enough data for indicators)
    # We need at least 200 days for SMA200 + some buffer for lookback
    print("Fetching recent data for signal generation...")
    
    # Fetch last 365 days to be safe for 200 SMA
    start_date_signal = (pd.Timestamp.now() - pd.DateOffset(days=400)).strftime('%Y-%m-%d')
    df_prices = fetch_data(SYMBOLS, start_date_signal, None)
    df_prices = df_prices.dropna()
    
    # Fetch FRED data
    print("Fetching FRED macro data...")
    t10y2y = fetch_fred_data("T10Y2Y", start_date_signal)
    dff = fetch_fred_data("DFF", start_date_signal)
    
    # Align FRED data with trading days
    df_prices['Yield_Curve'] = t10y2y.reindex(df_prices.index).ffill()
    df_prices['Fed_Funds_Rate'] = dff.reindex(df_prices.index).ffill()
    
    # 2. Indicators
    indicators_df = calculate_indicators(df_prices)
    indicators_df['SQQQ_RSI_10'] = calculate_rsi(df_prices['SQQQ'], 10)
    indicators_df['BSV_RSI_10'] = calculate_rsi(df_prices['BSV'], 10)
    
    # SPY vs GLD Ratio
    indicators_df['SPY_GLD_Ratio'] = df_prices['SPY'] / df_prices['GLD']
    
    # Align Dataframes
    combined = pd.concat([df_prices, indicators_df], axis=1).dropna()
    
    if combined.empty:
        print("Error: Not enough data to calculate indicators.")
        return

    # 3. Latest Signal
    print("\n" + "=" * 30)
    print(" LATEST TRADING SIGNAL")
    print("=" * 30)
    
    # 4. Fetch Yield Curve Snapshot (need last 120 days to get 3 months ago)
    print("Fetching Yield Curve History...")
    start_date_recent = (pd.Timestamp.now() - pd.DateOffset(days=120)).strftime('%Y-%m-%d')
    yc_maturities = ['1Mo', '3Mo', '6Mo', '1Yr', '2Yr', '5Yr', '10Yr', '30Yr']
    yc_tickers = ['DGS1MO', 'DGS3MO', 'DGS6MO', 'DGS1', 'DGS2', 'DGS5', 'DGS10', 'DGS30']
    
    yc_df = pd.DataFrame()
    for m, t in zip(yc_maturities, yc_tickers):
        try:
            s_data = fetch_fred_data(t, start_date_recent)
            if not s_data.empty:
                yc_df[m] = s_data.astype(float)
        except Exception as e:
            print(f"Failed to fetch {m}: {e}")
            
    yc_df = yc_df.ffill().dropna()

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
        chart_buf = generate_chart(df_prices, indicators_df, recommended_asset, yield_curve_history=yc_df)
    except Exception as e:
        print(f"Error generating chart: {e}")
        chart_buf = None

    # Send Notification
    msg = (
        f"🚀 *Composer Strategy Signal*\n"
        f"📅 Date: {last_date.date()}\n"
        f"📈 Recommendation: *{recommended_asset}*\n"
        f"🧠 Logic: {reason}\n"
        f"👉 Action: Buy/Hold at Close\n"
        f"--- Macro Indicators ---\n"
        f"Yield Curve (10Y-2Y): {latest_row_prices.get('Yield_Curve', 'N/A'):.2f}%\n"
        f"Fed Funds Rate: {latest_row_prices.get('Fed_Funds_Rate', 'N/A'):.2f}%\n"
        f"SPY/GLD Ratio: {latest_row_indicators.get('SPY_GLD_Ratio', 'N/A'):.2f}"
    )
    send_telegram_message(msg, image_file=chart_buf)

if __name__ == "__main__":
    run_daily_signal()
