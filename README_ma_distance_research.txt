MA DISTANCE VS. LONG BREAKOUT RESEARCH

WHAT IT TESTS
The same long-only breakout rule is compared in two ways:
1) Is the breakout close within 3% of the 50-session SMA, or farther away?
2) Was the lowest low in the 10 sessions before the breakout within 3% of the SMA on that low's date, or farther away?
The second is a reproducible setup-low proxy. It does not claim to identify every genuine Wyckoff LPS pattern.

TRADE RULES
- Signal: close above the highest high of the previous 20 completed daily sessions.
- Entry: next trading session's open (no entry at the signal close).
- Exit: close of the 10th holding session.
- Long only, no stop-loss, no leverage.
- One open trade per ticker within each test/cohort. Different cohorts and the two tests are separate research portfolios.
- MA-distance filter is the only default setup filter. To retain the original script's 63-session relative-strength filter, run with --relative-strength-filter.

INSTALL (Windows PowerShell / CMD)
1) Put ma_distance_research.py, this README, and requirements_ma_distance.txt in the same folder.
2) Install dependencies:
   py -m pip install -r requirements_ma_distance.txt
3) Run the offline code checks:
   py ma_distance_research.py --self-test
4) Run the study (internet needed for first Yahoo Finance download):
   py ma_distance_research.py
5) Force a full data refresh when needed:
   py ma_distance_research.py --refresh

OUTPUTS
- yahoo_cache\: individual Yahoo Finance OHLCV CSV files, reused for up to 24 hours.
- data_manifest.csv: per-symbol data coverage, source, and download/cache errors.
- signal_events.csv: every qualifying breakout event (including signals later skipped due to same-ticker overlap).
- trade_log.csv: non-overlapping trades used in the two cohort comparisons.
- performance_summary.csv: trade-level and daily portfolio statistics for each test/cohort.
- cohort_comparison.csv: near-minus-far differences in average trade return, win rate, and portfolio return. These are descriptive comparisons, not significance tests.
- country_summary.csv: trade statistics split by country.
- ticker_summary.csv: trade statistics for each ticker, test, and cohort.
- daily_equity_curves.csv: daily returns, equity, and number of active positions.
- equity_curve_breakout_close_ma.png and equity_curve_setup_low_ma.png.
- research_report.txt: assumptions and printed summary.

IMPORTANT LIMITATIONS
Yahoo Finance data are auto-adjusted for splits/dividends. No slippage or market impact is modeled. The default cost is 0 bps per side; test costs with e.g. --cost-bps-per-side 5. Portfolio curves equally weight active positions and have no maximum-position cap, so treat them as a research comparison, not a finished execution model. A one-year pre-study warm-up is downloaded for indicator initialization, but signals are counted only inside the requested study period. The countries' returns are not currency-converted. Review data_manifest.csv and avoid interpreting a run that uses many STALE_CACHE_FALLBACK rows or missing symbols.
