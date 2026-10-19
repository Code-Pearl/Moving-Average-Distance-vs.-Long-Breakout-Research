import numpy as np
import pandas as pd
import yfinance as yf

# --- 1. CONFIGURATION & PARAMETERS ---
TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'JPM', 'AMD', 'NFLX']
BENCHMARK = 'SPY'
START_DATE = '2023-01-01'
END_DATE = '2026-01-01'

MA_WINDOW = 50  # Key moving average window (e.g., 50-day SMA)
BREAKOUT_WINDOW = 20  # Resistance lookback window (20-day high)
MA_PROXIMITY_PCT = 0.02  # Within 2% of the moving average
FORWARD_DAYS = 10  # Horizon to measure forward performance

print('Downloading historical data...')
data = yf.download(TICKERS + [BENCHMARK], start=START_DATE, end=END_DATE, progress=False)

# Handle multi-index columns returned by newer yfinance versions
if isinstance(data.columns, pd.MultiIndex):
  close_prices = data['Close']
else:
  close_prices = data


def run_breakout_backtest(prices, benchmark_ticker):
  benchmark = prices[benchmark_ticker]
  performance_records = []

  for ticker in prices.columns:
    if ticker == benchmark_ticker:
      continue

    # Build individual stock dataframe
    df = pd.DataFrame(
        {
            'Close': prices[ticker],
            'Benchmark': benchmark,
        }
    ).dropna()

    # Step 1: Relative Strength Filter (3-month / 63-day ROC vs Benchmark)
    df['Stock_ROC'] = df['Close'].pct_change(63)
    df['Bench_ROC'] = df['Benchmark'].pct_change(63)
    df['Outperforming'] = df['Stock_ROC'] > df['Bench_ROC']

    # Step 2: Moving Average & Proximity Calculation (Wyckoff LPS Proxy)
    df['MA'] = df['Close'].rolling(window=MA_WINDOW).mean()
    df['Dist_From_MA'] = (df['Close'] - df['MA']).abs() / df['MA']
    df['Near_MA'] = df['Dist_From_MA'] <= MA_PROXIMITY_PCT

    # Step 3: Breakout Trigger (20-day High Breakout)
    df['Highest_High'] = (
        df['Close'].shift(1).rolling(window=BREAKOUT_WINDOW).max()
    )
    df['Breakout'] = df['Close'] > df['Highest_High']

    # Step 4: Forward Return Measurement
    df['Fwd_Return'] = df['Close'].shift(-FORWARD_DAYS) / df['Close'] - 1

    # Filter for valid breakout signals in outperforming stocks
    valid_signals = df[df['Outperforming'] & df['Breakout']].dropna(
        subset=['Fwd_Return']
    )

    # Separate into Test (Near MA) vs Control (Far from MA)
    near_ma_signals = valid_signals[valid_signals['Near_MA']]
    far_ma_signals = valid_signals[~valid_signals['Near_MA']]

    performance_records.append({
        'Ticker': ticker,
        'Near_MA_Count': len(near_ma_signals),
        'Near_MA_WinRate': (
            (near_ma_signals['Fwd_Return'] > 0).mean()
            if len(near_ma_signals) > 0
            else np.nan
        ),
        'Near_MA_AvgReturn': (
            near_ma_signals['Fwd_Return'].mean()
            if len(near_ma_signals) > 0
            else np.nan
        ),
        'Far_MA_Count': len(far_ma_signals),
        'Far_MA_WinRate': (
            (far_ma_signals['Fwd_Return'] > 0).mean()
            if len(far_ma_signals) > 0
            else np.nan
        ),
        'Far_MA_AvgReturn': (
            far_ma_signals['Fwd_Return'].mean()
            if len(far_ma_signals) > 0
            else np.nan
        ),
    })

  return pd.DataFrame(performance_records)


# --- 3. EXECUTE & DISPLAY ---
results_df = run_breakout_backtest(close_prices, BENCHMARK)

print('\n' + '=' * 60)
print(f' BACKTEST RESULTS ({FORWARD_DAYS}-Day Forward Return)')
print('=' * 60)
print(results_df.to_string(index=False))

# Aggregate Portfolio-Wide Statistics
print('\n' + '=' * 60)
print(' AGGREGATE HYPOTHESIS TEST SUMMARY')
print('=' * 60)
print(
    f"Average Return (Near MA / LPS Style):"
    f" {results_df['Near_MA_AvgReturn'].mean():.2%}"
)
print(
    f"Average Return (Far from MA / Extended):"
    f" {results_df['Far_MA_AvgReturn'].mean():.2%}"
)
print(
    f"Win Rate (Near MA): {results_df['Near_MA_WinRate'].mean():.2%}"
)  # noqa: E501
print(f"Win Rate (Far from MA): {results_df['Far_MA_WinRate'].mean():.2%}")