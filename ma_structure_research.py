#!/usr/bin/env python3
"""Research daily long-only breakouts using a bullish moving-average stack, price surfing, and an LPS proxy.

Rules are mechanical research definitions, not a claim to identify every valid Wyckoff setup.
Yahoo Finance adjusted daily OHLCV is downloaded and cached locally. Signals are detected at
bar close; entries occur at the next session open; exits occur at the close of the 10th holding
session. Signals are evaluated only inside the requested study dates, with warm-up and exit
buffer data downloaded outside that window. Each strategy is simulated separately.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COUNTRY_BASKETS: dict[str, dict[str, Any]] = {
    "United States": {
        "benchmark": "SPY",
        "tickers": ["AAPL", "MSFT", "NVDA", "AMZN", "JPM", "GOOGL", "META", "AMD", "NFLX", "CAT"],
    },
    "Germany": {
        "benchmark": "^GDAXI",
        "tickers": ["SAP.DE", "SIE.DE", "ALV.DE", "DTG.DE", "MBG.DE", "BMW.DE", "BAS.DE", "BAYN.DE", "ADS.DE", "DTE.DE"],
    },
    "France": {
        "benchmark": "^FCHI",
        "tickers": ["OR.PA", "MC.PA", "TTE.PA", "SAN.PA", "AIR.PA", "SU.PA", "BNP.PA", "EL.PA", "RMS.PA", "DG.PA"],
    },
    "Japan": {
        "benchmark": "^N225",
        "tickers": ["7203.T", "6758.T", "9983.T", "6861.T", "8306.T", "8035.T", "6501.T", "7974.T", "9984.T", "4063.T"],
    },
    "Canada": {
        "benchmark": "^GSPTSE",
        "tickers": ["RY.TO", "TD.TO", "SHOP.TO", "ENB.TO", "CNR.TO", "CP.TO", "BNS.TO", "BMO.TO", "TRI.TO", "SU.TO"],
    },
    "South Korea": {
        "benchmark": "^KS11",
        "tickers": ["005930.KS", "000660.KS", "005380.KS", "000270.KS", "005490.KS", "035420.KS", "035720.KS", "051910.KS", "006400.KS", "105560.KS"],
    },
}
# -------------------- RESEARCH SETTINGS --------------------
DEFAULT_START_DATE = "2021-01-01"
MA_WINDOWS = (10, 21, 50, 200)
BREAKOUT_WINDOW = 20
SURF_LOOKBACK = 10
SURF_TOLERANCE_PCT = 3.0
SURF_MIN_DAYS = 6               # price must be near at least one of SMA10/21/50 on >= 6 of prior 10 sessions
STACK_LOOKBACK = 10
STACK_MIN_DAYS = 8              # bullish MA ordering present on >= 8 of prior 10 sessions
LPS_LOOKBACK = 10
LPS_TOLERANCE_PCT = 3.0         # setup low within +/-3% of SMA50 on the day of that low
HOLDING_DAYS = 10
RELATIVE_STRENGTH_LOOKBACK = 63
USE_RELATIVE_STRENGTH_FILTER_BY_DEFAULT = False
TRANSACTION_COST_BPS_PER_SIDE = 0.0
CACHE_TTL_HOURS = 24
DOWNLOAD_CHUNK_SIZE = 12
EXIT_DATA_BUFFER_DAYS = 35
HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = HERE / "ma_structure_research_output"
PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

STRATEGIES = {
    "All_Breakouts": "Qualifies_All_Breakouts",
    "Bullish_Stack": "Qualifies_Bullish_Stack",
    "Stacked_and_Surfing": "Qualifies_Stacked_and_Surfing",
    "LPS_Proxy": "Qualifies_LPS_Proxy",
    "Stacked_LPS_Proxy": "Qualifies_Stacked_LPS_Proxy",
}

def ticker_file_stem(ticker: str) -> str:
    return ticker.replace("^", "INDEX_").replace(".", "_").replace("/", "_").replace("\\", "_")


def empty_prices() -> pd.DataFrame:
    return pd.DataFrame(columns=PRICE_COLUMNS, index=pd.DatetimeIndex([], name="Date"))


def normalize_price_frame(frame: pd.DataFrame | None) -> pd.DataFrame:
    """Flatten yfinance's version-dependent column layout into daily OHLCV."""
    if frame is None or frame.empty:
        return empty_prices()
    data = frame.copy()
    if isinstance(data.columns, pd.MultiIndex):
        fields = {"Open", "High", "Low", "Close"}
        chosen_level = None
        for level in range(data.columns.nlevels):
            values = {str(x) for x in data.columns.get_level_values(level)}
            if fields.issubset(values):
                chosen_level = level
                break
        if chosen_level is not None:
            data.columns = data.columns.get_level_values(chosen_level)
        else:
            return empty_prices()
    data = data.loc[:, ~pd.Index(data.columns).duplicated(keep="first")]
    if "Close" not in data.columns and "Adj Close" in data.columns:
        data["Close"] = data["Adj Close"]
    required = ["Open", "High", "Low", "Close"]
    if not set(required).issubset(data.columns):
        return empty_prices()
    if "Volume" not in data.columns:
        data["Volume"] = np.nan
    data = data[PRICE_COLUMNS].copy()
    for col in PRICE_COLUMNS:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    idx = pd.DatetimeIndex(pd.to_datetime(data.index, errors="coerce"))
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    data.index = idx.normalize()
    data.index.name = "Date"
    data = data.loc[~data.index.isna()]
    data = data[~data.index.duplicated(keep="last")].sort_index()
    data = data.dropna(subset=["Open", "High", "Low", "Close"])
    data = data[(data["Open"] > 0) & (data["High"] > 0) & (data["Low"] > 0) & (data["Close"] > 0)]
    return data


def read_cache(path: Path) -> pd.DataFrame:
    if not path.exists():
        return empty_prices()
    try:
        frame = pd.read_csv(path, index_col=0, parse_dates=True)
        return normalize_price_frame(frame)
    except Exception as exc:
        print(f"[WARN] Could not read cache {path.name}: {exc}")
        return empty_prices()


def restrict_dates(frame: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)  # Yahoo's end date is exclusive.
    return frame.loc[(frame.index >= start) & (frame.index < end)].copy()


def extract_ticker_from_batch(batch: pd.DataFrame, ticker: str, chunk_size: int) -> pd.DataFrame:
    if batch is None or batch.empty:
        return empty_prices()
    if not isinstance(batch.columns, pd.MultiIndex):
        return normalize_price_frame(batch) if chunk_size == 1 else empty_prices()
    for level in range(batch.columns.nlevels):
        if ticker in set(batch.columns.get_level_values(level)):
            try:
                sub = batch.xs(ticker, axis=1, level=level, drop_level=True)
                return normalize_price_frame(sub)
            except Exception:
                continue
    # Some yfinance versions can return a flat frame for a one-ticker batch.
    if chunk_size == 1:
        return normalize_price_frame(batch)
    return empty_prices()


def load_yahoo_data(
    tickers: list[str], start_date: str, end_date: str, cache_dir: Path, force_refresh: bool = False
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Load fresh cache first, download stale/missing items in batches, fall back to stale cache.

    A full requested-range download replaces the cache to keep adjusted historical prices
    internally consistent when Yahoo revises price adjustments for dividends/splits.
    """
    # Import yfinance only if a download is actually needed. A complete local cache can
    # still be backtested offline, even on a machine where yfinance is currently unavailable.
    yf = None
    cache_dir.mkdir(parents=True, exist_ok=True)
    now = time.time()
    loaded: dict[str, pd.DataFrame] = {}
    stale: dict[str, pd.DataFrame] = {}
    sources: dict[str, str] = {}
    errors: dict[str, str] = {}
    need_download: list[str] = []

    for ticker in tickers:
        path = cache_dir / f"{ticker_file_stem(ticker)}.csv"
        cached = restrict_dates(read_cache(path), start_date, end_date)
        # A cache must cover the requested start (allowing a few days for a newly listed name).
        covers_start = not cached.empty and cached.index.min() <= pd.Timestamp(start_date) + pd.Timedelta(days=7)
        fresh = path.exists() and ((now - path.stat().st_mtime) <= CACHE_TTL_HOURS * 3600)
        if not cached.empty:
            stale[ticker] = cached
        if not force_refresh and fresh and covers_start:
            loaded[ticker] = cached
            sources[ticker] = "CACHE_FRESH"
        else:
            need_download.append(ticker)

    download_errors: dict[str, str] = {}
    downloaded: dict[str, pd.DataFrame] = {}
    if need_download:
        try:
            import yfinance as yf  # type: ignore
        except ImportError:
            for ticker in need_download:
                message = "yfinance is not installed; install requirements_ma_distance.txt to download/refresh data"
                if ticker in stale and not stale[ticker].empty:
                    loaded[ticker] = stale[ticker]
                    sources[ticker] = "STALE_CACHE_FALLBACK"
                    errors[ticker] = message
                else:
                    sources[ticker] = "FAILED"
                    errors[ticker] = message
            need_download = []

    for start in range(0, len(need_download), DOWNLOAD_CHUNK_SIZE):
        chunk = need_download[start : start + DOWNLOAD_CHUNK_SIZE]
        try:
            raw = yf.download(
                tickers=chunk,
                start=start_date,
                end=end_date,
                auto_adjust=True,
                actions=False,
                progress=False,
                group_by="ticker",
                threads=True,
            )
            for ticker in chunk:
                got = restrict_dates(extract_ticker_from_batch(raw, ticker, len(chunk)), start_date, end_date)
                if not got.empty:
                    downloaded[ticker] = got
                else:
                    download_errors[ticker] = "Yahoo returned no usable OHLC rows in batch download"
        except Exception as exc:
            for ticker in chunk:
                download_errors[ticker] = f"Batch download failed: {type(exc).__name__}: {exc}"

    # Retry individual symbols that were missing in a batch; this also handles version-specific
    # yfinance column layouts and some symbols that cause partial batch responses.
    for ticker in list(download_errors):
        try:
            raw_one = yf.download(
                tickers=ticker,
                start=start_date,
                end=end_date,
                auto_adjust=True,
                actions=False,
                progress=False,
                threads=False,
            )
            got = restrict_dates(extract_ticker_from_batch(raw_one, ticker, 1), start_date, end_date)
            if not got.empty:
                downloaded[ticker] = got
                download_errors.pop(ticker, None)
        except Exception as exc:
            download_errors[ticker] = f"Individual download failed: {type(exc).__name__}: {exc}"

    for ticker in need_download:
        path = cache_dir / f"{ticker_file_stem(ticker)}.csv"
        if ticker in downloaded:
            frame = downloaded[ticker]
            try:
                frame.to_csv(path, index_label="Date", date_format="%Y-%m-%d")
                loaded[ticker] = frame
                sources[ticker] = "YAHOO_DOWNLOADED"
            except Exception as exc:
                # The downloaded data are usable for this run even if cache writing fails.
                loaded[ticker] = frame
                sources[ticker] = f"YAHOO_DOWNLOADED_CACHE_WRITE_FAILED: {exc}"
        elif ticker in stale and not stale[ticker].empty:
            loaded[ticker] = stale[ticker]
            sources[ticker] = "STALE_CACHE_FALLBACK"
            errors[ticker] = download_errors.get(ticker, "Download failed; using cached data")
        else:
            sources[ticker] = "FAILED"
            errors[ticker] = download_errors.get(ticker, "No cached data and no Yahoo data returned")

    manifest_rows: list[dict[str, Any]] = []
    for ticker in tickers:
        frame = loaded.get(ticker, empty_prices())
        manifest_rows.append(
            {
                "Ticker": ticker,
                "Source": sources.get(ticker, "FAILED"),
                "Rows": len(frame),
                "First_Date": frame.index.min().date().isoformat() if not frame.empty else "",
                "Last_Date": frame.index.max().date().isoformat() if not frame.empty else "",
                "Error": errors.get(ticker, ""),
            }
        )
    return loaded, pd.DataFrame(manifest_rows)


def build_portfolio_curve(
    selected: list[dict[str, Any]],
    price_by_ticker: dict[str, pd.DataFrame],
    all_dates: pd.DatetimeIndex,
    test_name: str,
    cohort: str,
    cost_bps_per_side: float,
) -> pd.DataFrame:
    """Equal-weight daily sleeve of open positions, flat in cash when no positions are open."""
    daily_returns: dict[pd.Timestamp, list[float]] = {}
    fee = max(float(cost_bps_per_side), 0.0) / 10000.0
    for trade in selected:
        ticker = str(trade["Ticker"])
        bars = price_by_ticker.get(ticker)
        if bars is None or bars.empty:
            continue
        bars = normalize_price_frame(bars)
        entry_i, exit_i = int(trade["_Entry_Index"]), int(trade["_Exit_Index"])
        if entry_i >= len(bars) or exit_i >= len(bars) or entry_i > exit_i:
            continue
        for j in range(entry_i, exit_i + 1):
            if j == entry_i:
                multiplier = float(bars["Close"].iloc[j] / bars["Open"].iloc[j])
                multiplier /= (1.0 + fee)  # buy-side cost on entry
            else:
                previous_close = float(bars["Close"].iloc[j - 1])
                if previous_close <= 0:
                    continue
                multiplier = float(bars["Close"].iloc[j] / previous_close)
            if j == exit_i:
                multiplier *= (1.0 - fee)  # sell-side cost on exit
            day = pd.Timestamp(bars.index[j]).normalize()
            daily_returns.setdefault(day, []).append(multiplier - 1.0)

    frame = pd.DataFrame(index=all_dates)
    frame.index.name = "Date"
    frame["Daily_Return"] = [float(np.mean(daily_returns[d])) if d in daily_returns else 0.0 for d in frame.index]
    frame["Active_Positions"] = [len(daily_returns[d]) if d in daily_returns else 0 for d in frame.index]
    frame["Equity"] = 100.0 * (1.0 + frame["Daily_Return"]).cumprod()
    frame = frame.reset_index()
    frame["Test"] = test_name
    frame["Cohort"] = cohort
    return frame[["Date", "Test", "Cohort", "Daily_Return", "Equity", "Active_Positions"]]


def trade_metrics(returns: pd.Series | np.ndarray) -> dict[str, Any]:
    vals = pd.Series(returns, dtype=float).dropna()
    if vals.empty:
        return {
            "Trades": 0, "Win_Rate_Pct": np.nan, "Avg_Trade_Return_Pct": np.nan,
            "Median_Trade_Return_Pct": np.nan, "Avg_Win_Pct": np.nan,
            "Avg_Loss_Pct": np.nan, "Profit_Factor": np.nan,
        }
    wins = vals[vals > 0]
    losses = vals[vals < 0]
    loss_abs = abs(float(losses.sum()))
    pf = float(wins.sum() / loss_abs) if loss_abs > 0 else (math.inf if not wins.empty else np.nan)
    return {
        "Trades": int(len(vals)),
        "Win_Rate_Pct": float((vals > 0).mean() * 100.0),
        "Avg_Trade_Return_Pct": float(vals.mean() * 100.0),
        "Median_Trade_Return_Pct": float(vals.median() * 100.0),
        "Avg_Win_Pct": float(wins.mean() * 100.0) if not wins.empty else np.nan,
        "Avg_Loss_Pct": float(losses.mean() * 100.0) if not losses.empty else np.nan,
        "Profit_Factor": pf,
    }


def portfolio_metrics(curve: pd.DataFrame) -> dict[str, Any]:
    if curve.empty:
        return {"Final_Equity": 100.0, "Portfolio_Total_Return_Pct": 0.0, "CAGR_Pct": np.nan,
                "Max_Drawdown_Pct": np.nan, "Sharpe_252": np.nan, "Avg_Active_Positions": 0.0}
    equity = curve["Equity"].astype(float)
    daily = curve["Daily_Return"].astype(float)
    final_equity = float(equity.iloc[-1])
    first_date, last_date = pd.Timestamp(curve["Date"].iloc[0]), pd.Timestamp(curve["Date"].iloc[-1])
    span_days = max((last_date - first_date).days, 1)
    cagr = ((final_equity / 100.0) ** (365.25 / span_days) - 1.0) * 100.0 if final_equity > 0 else np.nan
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    vol = float(daily.std(ddof=1)) if len(daily) > 1 else np.nan
    sharpe = float(daily.mean() / vol * np.sqrt(252.0)) if np.isfinite(vol) and vol > 0 else np.nan
    return {
        "Final_Equity": final_equity,
        "Portfolio_Total_Return_Pct": (final_equity / 100.0 - 1.0) * 100.0,
        "CAGR_Pct": cagr,
        "Max_Drawdown_Pct": float(drawdown.min() * 100.0),
        "Sharpe_252": sharpe,
        "Avg_Active_Positions": float(curve["Active_Positions"].mean()),
    }


def distance_pct(price: float, average: float) -> float:
    if not np.isfinite(price) or not np.isfinite(average) or average <= 0:
        return float("nan")
    return abs(price / average - 1.0) * 100.0


def build_breakout_events(
    country: str,
    ticker: str,
    prices: pd.DataFrame,
    benchmark_prices: pd.DataFrame | None,
    use_relative_strength_filter: bool,
    cost_bps_per_side: float,
    study_start_date: str,
    study_end_date: str,
) -> list[dict[str, Any]]:
    """Build one diagnostic row per eligible breakout, with non-lookahead setup measures."""
    bars = normalize_price_frame(prices)
    minimum_bars = max(MA_WINDOWS) + max(SURF_LOOKBACK, STACK_LOOKBACK, LPS_LOOKBACK) + HOLDING_DAYS + 2
    if len(bars) < minimum_bars:
        return []

    close, high, low = bars["Close"], bars["High"], bars["Low"]
    ma10 = close.rolling(10, min_periods=10).mean()
    ma21 = close.rolling(21, min_periods=21).mean()
    ma50 = close.rolling(50, min_periods=50).mean()
    ma200 = close.rolling(200, min_periods=200).mean()
    # Breakout uses the prior 20 completed sessions' HIGHS; today's bar is excluded.
    prior_20_high = high.shift(1).rolling(BREAKOUT_WINDOW, min_periods=BREAKOUT_WINDOW).max()
    breakout = close > prior_20_high

    stack_today = (ma10 > ma21) & (ma21 > ma50) & (ma50 > ma200)
    if use_relative_strength_filter:
        stock_roc = close.pct_change(RELATIVE_STRENGTH_LOOKBACK)
        if benchmark_prices is None or benchmark_prices.empty:
            bench_roc = pd.Series(np.nan, index=bars.index)
            relative_ok = pd.Series(False, index=bars.index)
        else:
            bench_close = benchmark_prices["Close"].reindex(bars.index).ffill()
            bench_roc = bench_close.pct_change(RELATIVE_STRENGTH_LOOKBACK)
            relative_ok = stock_roc > bench_roc
    else:
        stock_roc = close.pct_change(RELATIVE_STRENGTH_LOOKBACK)
        bench_roc = pd.Series(np.nan, index=bars.index)
        relative_ok = pd.Series(True, index=bars.index)

    start_ts, end_ts = pd.Timestamp(study_start_date), pd.Timestamp(study_end_date)
    fee = max(float(cost_bps_per_side), 0.0) / 10000.0
    candidates: list[dict[str, Any]] = []
    index = bars.index
    ma_bundle = pd.concat([ma10, ma21, ma50], axis=1)

    for i in np.flatnonzero((breakout & relative_ok).fillna(False).to_numpy()):
        signal_date = pd.Timestamp(index[i]).normalize()
        if signal_date < start_ts or signal_date >= end_ts:
            continue
        entry_i = int(i) + 1
        exit_i = entry_i + HOLDING_DAYS - 1
        if i < max(MA_WINDOWS) + max(SURF_LOOKBACK, STACK_LOOKBACK, LPS_LOOKBACK) or exit_i >= len(bars):
            continue
        averages = [ma10.iloc[i], ma21.iloc[i], ma50.iloc[i], ma200.iloc[i]]
        if any(pd.isna(x) or not np.isfinite(float(x)) or float(x) <= 0 for x in averages):
            continue

        # The setup window ends BEFORE the breakout bar to avoid counting the breakout candle as "surfing".
        surf_positions = np.arange(i - SURF_LOOKBACK, i)
        surf_close = close.iloc[surf_positions].to_numpy(dtype=float)
        surf_averages = ma_bundle.iloc[surf_positions].to_numpy(dtype=float)
        denom_ok = np.isfinite(surf_averages) & (surf_averages > 0)
        distances = np.abs(surf_close[:, None] / np.where(denom_ok, surf_averages, np.nan) - 1.0) * 100.0
        nearest_distance = np.nanmin(distances, axis=1)
        surf_floor_ok = surf_close >= ma50.iloc[surf_positions].to_numpy(dtype=float) * (1.0 - SURF_TOLERANCE_PCT / 100.0)
        surf_day_flags = (nearest_distance <= SURF_TOLERANCE_PCT) & surf_floor_ok
        surf_days = int(np.nansum(surf_day_flags))
        qualifies_surfing = surf_days >= SURF_MIN_DAYS

        stack_positions = np.arange(i - STACK_LOOKBACK, i)
        stack_days_prior = int(stack_today.iloc[stack_positions].fillna(False).sum())
        stack_on_signal = bool(stack_today.iloc[i])
        stable_stack = stack_on_signal and stack_days_prior >= STACK_MIN_DAYS

        setup_positions = np.arange(i - LPS_LOOKBACK, i)
        setup = bars.iloc[setup_positions]
        setup_low_date = setup["Low"].idxmin()
        setup_low_i = int(bars.index.get_loc(setup_low_date))
        setup_low_price = float(bars["Low"].iloc[setup_low_i])
        setup_low_ma50 = float(ma50.iloc[setup_low_i]) if pd.notna(ma50.iloc[setup_low_i]) else np.nan
        lps_distance = distance_pct(setup_low_price, setup_low_ma50)
        lps_proxy = bool(np.isfinite(lps_distance) and lps_distance <= LPS_TOLERANCE_PCT)

        entry_open = float(bars["Open"].iloc[entry_i])
        exit_close = float(bars["Close"].iloc[exit_i])
        if not np.isfinite(entry_open) or not np.isfinite(exit_close) or entry_open <= 0 or exit_close <= 0:
            continue
        gross_return = exit_close / entry_open - 1.0
        net_return = (exit_close * (1.0 - fee)) / (entry_open * (1.0 + fee)) - 1.0
        surf_pct = 100.0 * surf_days / SURF_LOOKBACK
        row = {
            "Country": country,
            "Ticker": ticker,
            "Signal_Date": index[i].date().isoformat(),
            "Entry_Date": index[entry_i].date().isoformat(),
            "Exit_Date": index[exit_i].date().isoformat(),
            "Signal_Close": float(close.iloc[i]),
            "Prior_20_Session_High": float(prior_20_high.iloc[i]),
            "Entry_Open": entry_open,
            "Exit_Close": exit_close,
            "Holding_Sessions": HOLDING_DAYS,
            "Gross_Return_Pct": gross_return * 100.0,
            "Net_Return_Pct": net_return * 100.0,
            "SMA10_On_Signal": float(ma10.iloc[i]),
            "SMA21_On_Signal": float(ma21.iloc[i]),
            "SMA50_On_Signal": float(ma50.iloc[i]),
            "SMA200_On_Signal": float(ma200.iloc[i]),
            "Bullish_Stack_On_Signal": stack_on_signal,
            "Stack_Days_Previous_10": stack_days_prior,
            "Stable_Bullish_Stack": stable_stack,
            "Price_Surf_Days_Previous_10": surf_days,
            "Price_Surf_Pct_Previous_10": surf_pct,
            "Meets_Surf_Criterion": qualifies_surfing,
            "Setup_Low_Date": setup_low_date.date().isoformat(),
            "Setup_Low_Price": setup_low_price,
            "SMA50_On_Setup_Low": setup_low_ma50,
            "Setup_Low_MA_Distance_Pct": lps_distance,
            "Meets_LPS_Proxy": lps_proxy,
            "Stock_ROC_63D_Pct": float(stock_roc.iloc[i] * 100.0) if pd.notna(stock_roc.iloc[i]) else np.nan,
            "Benchmark_ROC_63D_Pct": float(bench_roc.iloc[i] * 100.0) if pd.notna(bench_roc.iloc[i]) else np.nan,
            "Qualifies_All_Breakouts": True,
            "Qualifies_Bullish_Stack": stable_stack,
            "Qualifies_Stacked_and_Surfing": stable_stack and qualifies_surfing,
            "Qualifies_LPS_Proxy": lps_proxy,
            "Qualifies_Stacked_LPS_Proxy": stable_stack and lps_proxy,
            "_Signal_Index": int(i),
            "_Entry_Index": entry_i,
            "_Exit_Index": exit_i,
        }
        candidates.append(row)
    return candidates


def select_strategy_trades(candidates: list[dict[str, Any]], strategy: str) -> list[dict[str, Any]]:
    """Select qualifying trades, avoiding overlapping positions within each ticker/strategy."""
    flag = STRATEGIES[strategy]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in candidates:
        if bool(row.get(flag, False)):
            grouped.setdefault(str(row["Ticker"]), []).append(row)
    selected: list[dict[str, Any]] = []
    for ticker_rows in grouped.values():
        ticker_rows.sort(key=lambda item: int(item["_Signal_Index"]))
        last_exit_index = -1
        for row in ticker_rows:
            # Signal on the old position's exit date may enter next session after that close.
            if int(row["_Signal_Index"]) < last_exit_index:
                continue
            selected.append(row)
            last_exit_index = int(row["_Exit_Index"])
    selected.sort(key=lambda item: (item["Signal_Date"], item["Ticker"]))
    return selected


def write_strategy_equity_plot(curves: pd.DataFrame, output_path: Path, only_strategy: str | None = None) -> None:
    fig, ax = plt.subplots(figsize=(13, 7))
    if curves.empty:
        ax.text(0.5, 0.5, "No qualifying trades / no downloaded data", ha="center", va="center", transform=ax.transAxes)
    else:
        selected_strategies = [only_strategy] if only_strategy else list(STRATEGIES)
        for strategy in selected_strategies:
            part = curves[curves["Strategy"] == strategy].sort_values("Date")
            if not part.empty:
                ax.plot(pd.to_datetime(part["Date"]), part["Equity"], label=strategy.replace("_", " "), linewidth=1.8)
    ax.set_title((only_strategy or "Strategy comparison").replace("_", " ") + " — long breakout equity curve")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity (start = 100; illustrative equal-weight active-position model)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def build_comparison(performance: pd.DataFrame) -> pd.DataFrame:
    rows = []
    baseline = performance[performance["Strategy"] == "All_Breakouts"]
    if baseline.empty:
        return pd.DataFrame()
    base = baseline.iloc[0]
    for strategy in list(STRATEGIES)[1:]:
        part = performance[performance["Strategy"] == strategy]
        if part.empty:
            continue
        current = part.iloc[0]
        rows.append({
            "Strategy": strategy,
            "Baseline_Trades": int(base["Trades"]),
            "Strategy_Trades": int(current["Trades"]),
            "Avg_Trade_Return_Pct": current["Avg_Trade_Return_Pct"],
            "Baseline_Avg_Trade_Return_Pct": base["Avg_Trade_Return_Pct"],
            "Avg_Return_Difference_Pct_Points": current["Avg_Trade_Return_Pct"] - base["Avg_Trade_Return_Pct"],
            "Win_Rate_Pct": current["Win_Rate_Pct"],
            "Baseline_Win_Rate_Pct": base["Win_Rate_Pct"],
            "Win_Rate_Difference_Pct_Points": current["Win_Rate_Pct"] - base["Win_Rate_Pct"],
            "Profit_Factor": current["Profit_Factor"],
            "Baseline_Profit_Factor": base["Profit_Factor"],
            "Portfolio_Total_Return_Pct": current["Portfolio_Total_Return_Pct"],
            "Baseline_Portfolio_Total_Return_Pct": base["Portfolio_Total_Return_Pct"],
            "Portfolio_Return_Difference_Pct_Points": current["Portfolio_Total_Return_Pct"] - base["Portfolio_Total_Return_Pct"],
            "Max_Drawdown_Pct": current["Max_Drawdown_Pct"],
            "Baseline_Max_Drawdown_Pct": base["Max_Drawdown_Pct"],
        })
    return pd.DataFrame(rows)


def run_research(start_date: str, end_date: str, output_dir: Path, refresh_cache: bool,
                 use_relative_strength_filter: bool, cost_bps_per_side: float) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "yahoo_cache"
    all_tickers: list[str] = []
    ticker_country: dict[str, str] = {}
    for country, config in COUNTRY_BASKETS.items():
        for ticker in config["tickers"]:
            if ticker not in all_tickers:
                all_tickers.append(ticker)
            ticker_country[ticker] = country
        if config["benchmark"] not in all_tickers:
            all_tickers.append(config["benchmark"])

    warmup_start = (pd.Timestamp(start_date) - pd.Timedelta(days=365)).date().isoformat()
    data_end_date = (pd.Timestamp(end_date) + pd.Timedelta(days=EXIT_DATA_BUFFER_DAYS)).date().isoformat()
    print(f"Loading {len(all_tickers)} Yahoo Finance symbols; signal dates {start_date} to {end_date} (end exclusive).")
    price_by_ticker, manifest = load_yahoo_data(
        all_tickers, warmup_start, data_end_date, cache_dir, force_refresh=refresh_cache
    )
    manifest.to_csv(output_dir / "data_manifest.csv", index=False)
    usable = [t for t in ticker_country if t in price_by_ticker and not price_by_ticker[t].empty]
    if not usable:
        print("ERROR: No stock price histories are available. Check internet access, yfinance installation, and data_manifest.csv.")
        print("Install dependencies: py -m pip install -r requirements_ma_structure.txt")
        return 2

    events: list[dict[str, Any]] = []
    for country, config in COUNTRY_BASKETS.items():
        benchmark = price_by_ticker.get(config["benchmark"])
        for ticker in config["tickers"]:
            stock = price_by_ticker.get(ticker)
            if stock is None or stock.empty:
                continue
            events.extend(build_breakout_events(
                country, ticker, stock, benchmark, use_relative_strength_filter,
                cost_bps_per_side, start_date, end_date,
            ))

    public_columns = [
        "Country", "Ticker", "Signal_Date", "Entry_Date", "Exit_Date", "Signal_Close", "Prior_20_Session_High",
        "Entry_Open", "Exit_Close", "Holding_Sessions", "Gross_Return_Pct", "Net_Return_Pct",
        "SMA10_On_Signal", "SMA21_On_Signal", "SMA50_On_Signal", "SMA200_On_Signal",
        "Bullish_Stack_On_Signal", "Stack_Days_Previous_10", "Stable_Bullish_Stack",
        "Price_Surf_Days_Previous_10", "Price_Surf_Pct_Previous_10", "Meets_Surf_Criterion",
        "Setup_Low_Date", "Setup_Low_Price", "SMA50_On_Setup_Low", "Setup_Low_MA_Distance_Pct", "Meets_LPS_Proxy",
        "Stock_ROC_63D_Pct", "Benchmark_ROC_63D_Pct", *STRATEGIES.values(),
    ]
    diagnostics = pd.DataFrame([{col: event.get(col, np.nan) for col in public_columns} for event in events], columns=public_columns)
    diagnostics.to_csv(output_dir / "signal_diagnostics.csv", index=False)

    # Extend the equity curve only as far as available data, including exits for late-window signals.
    study_start_ts = pd.Timestamp(start_date)
    all_stock_indices = [frame.index for ticker, frame in price_by_ticker.items() if ticker in ticker_country and not frame.empty]
    if all_stock_indices:
        all_dates = pd.DatetimeIndex(sorted({pd.Timestamp(d).normalize() for idx in all_stock_indices for d in idx
                                             if study_start_ts <= pd.Timestamp(d).normalize() < pd.Timestamp(data_end_date)}), name="Date")
    else:
        all_dates = pd.DatetimeIndex([], name="Date")

    selected_rows: list[dict[str, Any]] = []
    curve_frames: list[pd.DataFrame] = []
    performance_rows: list[dict[str, Any]] = []
    for strategy in STRATEGIES:
        selected = select_strategy_trades(events, strategy)
        for row in selected:
            public = {col: row.get(col, np.nan) for col in public_columns}
            public["Strategy"] = strategy
            selected_rows.append(public)
        curve = build_portfolio_curve(selected, price_by_ticker, all_dates, strategy, "All", cost_bps_per_side)
        curve = curve.rename(columns={"Test": "Strategy"}).drop(columns=["Cohort"])
        curve_frames.append(curve)
        returns = pd.Series([float(row["Net_Return_Pct"]) / 100.0 for row in selected], dtype=float)
        performance_rows.append({
            "Strategy": strategy,
            **trade_metrics(returns),
            **portfolio_metrics(curve.rename(columns={"Strategy": "Test"}).assign(Cohort="All")[["Date", "Test", "Cohort", "Daily_Return", "Equity", "Active_Positions"]]),
        })

    trades = pd.DataFrame(selected_rows, columns=["Strategy", *public_columns])
    if not trades.empty:
        trades = trades.sort_values(["Strategy", "Signal_Date", "Ticker"]).reset_index(drop=True)
    trades.to_csv(output_dir / "trade_log.csv", index=False)
    curves = pd.concat(curve_frames, ignore_index=True) if curve_frames else pd.DataFrame(
        columns=["Date", "Strategy", "Daily_Return", "Equity", "Active_Positions"]
    )
    curves.to_csv(output_dir / "daily_equity_curves.csv", index=False, date_format="%Y-%m-%d")

    performance = pd.DataFrame(performance_rows)
    performance.to_csv(output_dir / "performance_summary.csv", index=False)
    comparison = build_comparison(performance)
    comparison.to_csv(output_dir / "strategy_comparison.csv", index=False)
    write_strategy_equity_plot(curves, output_dir / "equity_curves_all_strategies.png")
    for strategy in STRATEGIES:
        write_strategy_equity_plot(curves, output_dir / f"equity_curve_{strategy.lower()}.png", strategy)

    country_rows = []
    if not trades.empty:
        for (strategy, country), group in trades.groupby(["Strategy", "Country"], dropna=False):
            country_rows.append({"Strategy": strategy, "Country": country, **trade_metrics(group["Net_Return_Pct"] / 100.0)})
    pd.DataFrame(country_rows).to_csv(output_dir / "country_summary.csv", index=False)

    rows_with_data = int((manifest["Rows"] > 0).sum()) if not manifest.empty else 0
    report = [
        "MA STRUCTURE / PRICE-SURFING / LPS-PROXY BREAKOUT RESEARCH",
        "===========================================================",
        f"Signal date range: {start_date} to {end_date} (end date exclusive)",
        f"Price history warm-up starts: {warmup_start}; data downloaded through: {data_end_date}",
        f"Breakout: signal close > highest HIGH of prior {BREAKOUT_WINDOW} completed sessions",
        f"MAs: simple moving averages of {MA_WINDOWS}; bullish order is SMA10 > SMA21 > SMA50 > SMA200",
        f"Stable stack: bullish order on signal day and on >= {STACK_MIN_DAYS} of the previous {STACK_LOOKBACK} sessions",
        f"Surfing: previous {SURF_LOOKBACK} sessions; close within +/-{SURF_TOLERANCE_PCT:.1f}% of at least one of SMA10/SMA21/SMA50 and not >{SURF_TOLERANCE_PCT:.1f}% below SMA50 on >= {SURF_MIN_DAYS} sessions",
        f"LPS proxy: lowest low in the previous {LPS_LOOKBACK} sessions is within +/-{LPS_TOLERANCE_PCT:.1f}% of SMA50 on the day of that low",
        f"Entry: next session open; exit: close of holding session {HOLDING_DAYS}; no stop-loss or leverage",
        f"Relative-strength filter: {'ON' if use_relative_strength_filter else 'OFF'}; transaction costs: {cost_bps_per_side:.2f} bps per side",
        "Each strategy is simulated separately; same-ticker positions do not overlap within a strategy.",
        "Equity curves use equal-weight average returns across that strategy's currently open positions; this is illustrative, not a capped-capital broker simulation.",
        "Returns are local-currency price returns; no FX conversion, slippage, taxes, or market impact. Yahoo adjusted data are subject to vendor revisions.",
        "LPS is a mechanical proxy only. The script does not verify a preceding Wyckoff SOS/structure or volume/effort-result confirmation.",
        "Comparisons are descriptive, not proof of statistical significance. Cohorts can share dates/tickers and trades are not independent.",
        "",
        "PERFORMANCE SUMMARY",
        performance.to_string(index=False) if not performance.empty else "No strategy results generated.",
        "",
        "STRATEGIES VS BASELINE",
        comparison.to_string(index=False) if not comparison.empty else "No comparison available.",
        "",
        f"Symbols with downloaded/cached data: {rows_with_data}/{len(manifest)}",
        f"Eligible breakout events: {len(diagnostics)}; selected trades across all strategies (duplicates across strategies are intentional): {len(trades)}",
        f"Outputs: {output_dir}",
    ]
    (output_dir / "research_report.txt").write_text("\n".join(report), encoding="utf-8")

    print("\nPERFORMANCE SUMMARY")
    if performance.empty:
        print("No results. Check data_manifest.csv and the ticker data coverage.")
    else:
        columns = ["Strategy", "Trades", "Win_Rate_Pct", "Avg_Trade_Return_Pct", "Profit_Factor", "Portfolio_Total_Return_Pct", "Max_Drawdown_Pct", "Sharpe_252"]
        print(performance[columns].round(2).to_string(index=False))
    print(f"\nEligible breakout events: {len(diagnostics)}")
    print(f"Symbols with data: {rows_with_data}/{len(manifest)}")
    print(f"Outputs saved to: {output_dir}")
    return 0


def run_self_test() -> int:
    """Offline deterministic test of breakout timing, stack/surf detection, and output plumbing."""
    n = 330
    dates = pd.bdate_range("2023-01-02", periods=n)
    close = 100.0 + np.arange(n, dtype=float) * 0.4 + 0.15 * np.sin(np.arange(n) / 3.0)
    open_ = close - 0.05
    high = close + 0.12
    low = close - 0.12
    bars = pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": 10000}, index=dates)
    normalized = normalize_price_frame(bars)
    events = build_breakout_events("Test", "SYN", normalized, None, False, 0.0,
                                   dates[220].date().isoformat(), dates[-20].date().isoformat())
    assert events, "Synthetic uptrend did not generate breakout events"
    assert all(e["Entry_Date"] > e["Signal_Date"] for e in events)
    assert all(e["Exit_Date"] > e["Entry_Date"] for e in events)
    assert all(e["Holding_Sessions"] == HOLDING_DAYS for e in events)
    assert any(e["Qualifies_Bullish_Stack"] for e in events), "Synthetic data did not satisfy the stable stack"
    assert any(e["Qualifies_Stacked_and_Surfing"] for e in events), "Synthetic data did not satisfy stack + surfing"
    selected = select_strategy_trades(events, "Stacked_and_Surfing")
    assert selected, "Strategy trade selection returned no trades"
    assert all(a["_Exit_Index"] <= b["_Signal_Index"] for a, b in zip(selected, selected[1:]) if a["Ticker"] == b["Ticker"])
    print("SELF-TEST PASS")
    print(f"Generated {len(events)} synthetic breakout diagnostics; {len(selected)} non-overlapping stack+surf trades.")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default=DEFAULT_START_DATE, help="YYYY-MM-DD; default 2021-01-01")
    parser.add_argument("--end-date", default=date.today().isoformat(), help="YYYY-MM-DD; exclusive; default today")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output/cache directory")
    parser.add_argument("--refresh", action="store_true", help="Force fresh Yahoo Finance downloads")
    parser.add_argument("--relative-strength-filter", action="store_true", help="Require 63-session stock return to exceed its country benchmark")
    parser.add_argument("--no-relative-strength-filter", action="store_true", help="Disable relative-strength filter (default)")
    parser.add_argument("--cost-bps-per-side", type=float, default=TRANSACTION_COST_BPS_PER_SIDE, help="Estimated transaction cost per side in basis points; default 0")
    parser.add_argument("--self-test", action="store_true", help="Run offline synthetic checks without downloading data")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        return run_self_test()
    if args.start_date >= args.end_date:
        print("ERROR: --start-date must be earlier than --end-date.", file=sys.stderr)
        return 2
    if args.cost_bps_per_side < 0:
        print("ERROR: --cost-bps-per-side must be >= 0.", file=sys.stderr)
        return 2
    use_rs = USE_RELATIVE_STRENGTH_FILTER_BY_DEFAULT
    if args.relative_strength_filter:
        use_rs = True
    if args.no_relative_strength_filter:
        use_rs = False
    return run_research(args.start_date, args.end_date, Path(args.output_dir).expanduser().resolve(),
                        args.refresh, use_rs, args.cost_bps_per_side)


if __name__ == "__main__":
    raise SystemExit(main())
