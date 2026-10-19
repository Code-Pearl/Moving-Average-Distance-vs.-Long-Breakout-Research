MA STRUCTURE / PRICE-SURFING / LPS-PROXY RESEARCH
=================================================

QUICK START (Windows)
1. Extract this ZIP into a normal folder.
2. Double-click RUN_BACKTEST_MA_STRUCTURE.bat.
3. The batch file installs the listed Python packages if needed, runs an offline self-test,
   then starts the live Yahoo Finance backtest. Internet access is needed for first downloads.
4. Results are written into ma_structure_research_output beside the script.

Manual run from Command Prompt in this folder:
  py -m pip install -r requirements_ma_structure.txt
  py ma_structure_research.py --self-test
  py ma_structure_research.py

To set the test dates (end date is exclusive):
  py ma_structure_research.py --start-date 2021-01-01 --end-date 2026-01-01

To force fresh Yahoo downloads (instead of reusing the local cache):
  py ma_structure_research.py --refresh

RESEARCH DEFINITIONS
--------------------
All_Breakouts:
  Daily close breaks above the highest HIGH of the preceding 20 completed sessions.

Bullish_Stack:
  SMA10 > SMA21 > SMA50 > SMA200 on the breakout day, AND the same ordering is present
  on at least 8 of the preceding 10 sessions. This requires the stack to be established,
  rather than appearing for one bar only.

Stacked_and_Surfing:
  Bullish_Stack plus, during the 10 sessions BEFORE the breakout bar, at least 6 closes
  within 3% of at least one of SMA10/SMA21/SMA50. A qualifying close cannot be more than
  3% below SMA50. These thresholds are configurable near the top of the Python file.

LPS_Proxy:
  The lowest intraday LOW during the 10 sessions before breakout is within +/-3% of SMA50
  on the date that low occurred. This is only a mechanical proxy, NOT a verified Wyckoff LPS.
  The script does not establish a preceding SOS, the full Wyckoff structure, or volume
  confirmation.

Stacked_LPS_Proxy:
  Stable Bullish_Stack plus the LPS_Proxy rule above.

EXECUTION MODEL
---------------
- Daily Yahoo Finance OHLCV, auto-adjusted by yfinance (including split/dividend adjustments).
- Breakout is determined at the signal-day close; entry is at the next trading session's open.
- Exit is at the close of the 10th holding session.
- No stop-loss, leverage, or max concurrent-position cap. Trades don't overlap within the
  same ticker and strategy; the same signal can legitimately appear in multiple strategies.
- Costs default to zero. Set for example --cost-bps-per-side 5 to model 5 bps each way.
- Yahoo data are cached per ticker under ma_structure_research_output/yahoo_cache and reused
  for 24 hours. Use --refresh to force a download.

OUTPUTS
-------
performance_summary.csv             Core trade and equity metrics for each strategy.
strategy_comparison.csv             Each filtered strategy vs All_Breakouts baseline.
trade_log.csv                       Trade-by-trade results; strategy is the first column.
signal_diagnostics.csv              Every eligible breakout and all qualification flags.
daily_equity_curves.csv             Daily return, equity, and active-position count by strategy.
equity_curves_all_strategies.png    Overlay of each strategy's equity curve.
equity_curve_*.png                  Individual strategy equity plots.
country_summary.csv                 Strategy results broken down by country.
data_manifest.csv                  Source, coverage, and download/cache errors for each symbol.
research_report.txt                 Rules, caveats, summary metrics, and comparison.

IMPORTANT INTERPRETATION NOTES
------------------------------
- This is a small descriptive study, not proof of statistical significance or future edge.
- Strategies are separate portfolios; trade cohorts can share dates and tickers, so returns are
  not independent samples.
- The equity-curve model averages returns of currently open positions each session and is an
  illustrative comparison, not a fully capital-constrained broker simulation.
- The universe mixes country listings. Returns are in the underlying listing currency and are
  not converted into a single portfolio currency.
- Inspect data_manifest.csv to check for missing or stale Yahoo data before interpreting output.
