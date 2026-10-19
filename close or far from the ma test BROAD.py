import warnings
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings('ignore')

# --- 1. CONFIGURATION & GLOBAL BASKETS ---
COUNTRY_BASKETS = {
    'United States': {
        'benchmark': 'SPY',
        'tickers': [
            'AAPL',
            'MSFT',
            'NVDA',
            'AMZN',
            'JPM',
            'GOOGL',
            'META',
            'AMD',
            'NFLX',
            'CAT',
        ],
    },
    'Germany': {
        'benchmark': '^GDAXI',
        'tickers': [
            'SAP.DE',
            'SIE.DE',
            'ALV.DE',
            'DTG.DE',
            'MBG.DE',
            'BMW.DE',
            'BAS.DE',
            'BAYN.DE',
            'ADS.DE',
            'DTE.DE',
        ],
    },
    'France': {
        'benchmark': '^FCHI',
        'tickers': [
            'OR.PA',
            'MC.PA',
            'TTE.PA',
            'SAN.PA',
            'AIR.PA',
            'SU.PA',
            'BNP.PA',
            'EL.PA',
            'RMS.PA',
            'DG.PA',
        ],
    },
    'Japan': {
        'benchmark': '^N225',
        'tickers': [
            '7203.T',
            '6758.T',
            '9983.T',
            '6861.T',
            '8306.T',
            '8035.T',
            '6501.T',
            '7974.T',
            '9984.T',
            '4063.T',
        ],
    },
    'Canada': {
        'benchmark': '^GSPTSE',
        'tickers': [
            'RY.TO',
            'TD.TO',
            'SHOP.TO',
            'ENB.TO',
            'CNR.TO',
            'CP.TO',
            'BNS.TO',
            'BMO.TO',
            'TRI.TO',
            'SU.TO',
        ],
    },
    'South Korea': {
        'benchmark': '^KS11',
        'tickers': [
            '005930.KS',
            '000660.KS',
            '005380.KS',
            '000270.KS',
            '005490.KS',
            '035420.KS',
            '035720.KS',
            '051910.KS',
            '006400.KS',
            '105560.KS',
        ],
    },
}

START_DATE = '2021-01-01'
END_DATE = '2026-01-01'
MA_WINDOW = 50  # 50-day Moving Average (Support level)
BREAKOUT_WINDOW = 20  # 20-day Resistance High
MA_PROXIMITY_PCT = 0.03  # Within 3% of MA (LPS / consolidation zone)
FORWARD_DAYS = 10  # 10-day forward return horizon


# --- 2. BACKTEST ENGINE ---
def analyze_country_basket(country, config):
  all_tickers = config['tickers'] + [config['benchmark']]
  data = yf.download(
      all_tickers, start=START_DATE, end=END_DATE, progress=False
  )

  if isinstance(data.columns, pd.MultiIndex):
    close_prices = data['Close']
  else:
    close_prices = data

  benchmark_prices = close_prices[config['benchmark']].dropna()

  results = []

  for ticker in config['tickers']:
    if ticker not in close_prices.columns:
      continue

    df = pd.DataFrame(
        {'Close': close_prices[ticker], 'Benchmark': benchmark_prices}
    ).dropna()

    if len(df) < 100:
      continue

    # 1. Relative Strength Filter: 63-day (3-month) ROC outperformance
    df['Stock_ROC'] = df['Close'].pct_change(63)
    df['Bench_ROC'] = df['Benchmark'].pct_change(63)
    df['Relative_Strength'] = df['Stock_ROC'] > df['Bench_ROC']

    # 2. Moving Average & Wyckoff LPS Proximity Filter
    df['MA'] = df['Close'].rolling(window=MA_WINDOW).mean()
    df['Dist_From_MA'] = (df['Close'] - df['MA']).abs() / df['MA']
    df['Near_MA'] = df['Dist_From_MA'] <= MA_PROXIMITY_PCT

    # 3. Discrete Breakout Trigger (Crosses 20-day high on current bar)
    df['Prev_20D_High'] = (
        df['Close'].shift(1).rolling(window=BREAKOUT_WINDOW).max()
    )
    df['Fresh_Breakout'] = (df['Close'] > df['Prev_20D_High']) & (
        df['Close'].shift(1) <= df['Prev_20D_High'].shift(1)
    )

    # 4. Forward Performance Measurement
    df['Fwd_Return'] = df['Close'].shift(-FORWARD_DAYS) / df['Close'] - 1

    # Filter signals with Relative Strength + Fresh Breakout
    valid = df[df['Relative_Strength'] & df['Fresh_Breakout']].dropna(
        subset=['Fwd_Return']
    )

    near_ma = valid[valid['Near_MA']]
    far_ma = valid[~valid['Near_MA']]

    results.append({
        'Country': country,
        'Ticker': ticker,
        'Near_MA_Trades': len(near_ma),
        'Near_MA_AvgRet': near_ma['Fwd_Return'].mean(),
        'Near_MA_WinRate': (near_ma['Fwd_Return'] > 0).mean(),
        'Far_MA_Trades': len(far_ma),
        'Far_MA_AvgRet': far_ma['Fwd_Return'].mean(),
        'Far_MA_WinRate': (far_ma['Fwd_Return'] > 0).mean(),
    })

  return pd.DataFrame(results)


# --- 3. RUN MULTI-NATIONAL BACKTEST ---
all_country_results = []

for country, config in COUNTRY_BASKETS.items():
  res = analyze_country_basket(country, config)
  all_country_results.append(res)

full_df = pd.concat(all_country_results, ignore_index=True)

# Group by Country
country_summary = (
    full_df.groupby('Country')
    .agg({
        'Near_MA_Trades': 'sum',
        'Near_MA_AvgRet': 'mean',
        'Near_MA_WinRate': 'mean',
        'Far_MA_Trades': 'sum',
        'Far_MA_AvgRet': 'mean',
        'Far_MA_WinRate': 'mean',
    })
    .reset_index()
)

# Format for output
country_summary['Near_MA_AvgRet'] = country_summary['Near_MA_AvgRet'].apply(
    lambda x: f'{x:.2%}' if pd.notnull(x) else 'N/A'
)
country_summary['Near_MA_WinRate'] = country_summary['Near_MA_WinRate'].apply(
    lambda x: f'{x:.2%}' if pd.notnull(x) else 'N/A'
)
country_summary['Far_MA_AvgRet'] = country_summary['Far_MA_AvgRet'].apply(
    lambda x: f'{x:.2%}' if pd.notnull(x) else 'N/A'
)
country_summary['Far_MA_WinRate'] = country_summary['Far_MA_WinRate'].apply(
    lambda x: f'{x:.2%}' if pd.notnull(x) else 'N/A'
)

print('=' * 80)
print(' GLOBAL WYCKOFF LPS / MOVING AVERAGE BREAKOUT SUMMARY BY COUNTRY')
print('=' * 80)
print(country_summary.to_string(index=False))