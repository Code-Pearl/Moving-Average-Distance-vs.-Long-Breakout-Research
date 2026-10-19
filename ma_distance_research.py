#!/usr/bin/env python3
"""Research whether long breakouts perform differently near vs. far from a moving average.

Yahoo Finance adjusted daily OHLCV is cached locally. The study runs two classification
variants so the result is not dependent on one interpretation of “LPS location”:
  1) distance of the breakout close from the 50-session SMA;
  2) distance of the lowest low in the 10 sessions before the breakout from the SMA
     on that low's date (a mechanical setup-low proxy, not an automatic Wyckoff label).

Signals are close > the highest high of the previous 20 completed sessions. Entry is at
next session's open; exit is at the close of the 10th holding session. Per ticker and
per cohort, trades may not overlap. No stop-loss or leverage is used. Transaction costs
are configurable and default to zero; reported results are research estimates, not a
claim of real-world fill quality.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# -------------------- CONFIGURATION --------------------
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

DEFAULT_START_DATE = "2021-01-01"
MA_WINDOW = 50
BREAKOUT_WINDOW = 20
NEAR_MA_THRESHOLD_PCT = 3.0  # Absolute distance, ±3%, to preserve the original script's idea.
SETUP_LOW_LOOKBACK = 10
HOLDING_DAYS = 10
RELATIVE_STRENGTH_LOOKBACK = 63
USE_RELATIVE_STRENGTH_FILTER_BY_DEFAULT = False  # Off to isolate MA distance as the variable.
TRANSACTION_COST_BPS_PER_SIDE = 0.0
CACHE_TTL_HOURS = 24
DOWNLOAD_CHUNK_SIZE = 12
EXIT_DATA_BUFFER_DAYS = 35  # Download beyond the study end so late signals can complete their holding period.

HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = HERE / "ma_distance_research_output"

PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
TESTS = {
    "Breakout_Close_MA": "Breakout_Close_MA_Distance_Pct",
    "Setup_Low_MA": "Setup_Low_MA_Distance_Pct",
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


def pct_distance(price: float, ma: float) -> float:
    if not np.isfinite(price) or not np.isfinite(ma) or ma <= 0:
        return float("nan")
    return abs(price / ma - 1.0) * 100.0


def near_far(distance_pct: float) -> str:
    if not np.isfinite(distance_pct):
        return "Unclassified"
    return "Near_MA" if distance_pct <= NEAR_MA_THRESHOLD_PCT else "Far_MA"


def make_signal_events(
    country: str,
    ticker: str,
    prices: pd.DataFrame,
    benchmark_prices: pd.DataFrame | None,
    use_relative_strength_filter: bool,
    cost_bps_per_side: float,
    study_start_date: str | None = None,
    study_end_date: str | None = None,
) -> list[dict[str, Any]]:
    """Create breakout events with next-open entry and fixed 10-session exit."""
    bars = normalize_price_frame(prices)
    if len(bars) < max(MA_WINDOW, BREAKOUT_WINDOW, SETUP_LOW_LOOKBACK) + HOLDING_DAYS + 2:
        return []

    close = bars["Close"]
    ma = close.rolling(MA_WINDOW, min_periods=MA_WINDOW).mean()
    prior_high = bars["High"].shift(1).rolling(BREAKOUT_WINDOW, min_periods=BREAKOUT_WINDOW).max()
    breakout = close > prior_high

    if use_relative_strength_filter:
        if benchmark_prices is None or benchmark_prices.empty:
            relative_ok = pd.Series(False, index=bars.index)
            stock_roc = close.pct_change(RELATIVE_STRENGTH_LOOKBACK)
            bench_roc = pd.Series(np.nan, index=bars.index)
        else:
            benchmark_close = benchmark_prices["Close"].reindex(bars.index).ffill()
            stock_roc = close.pct_change(RELATIVE_STRENGTH_LOOKBACK)
            bench_roc = benchmark_close.pct_change(RELATIVE_STRENGTH_LOOKBACK)
            relative_ok = stock_roc > bench_roc
    else:
        relative_ok = pd.Series(True, index=bars.index)
        stock_roc = close.pct_change(RELATIVE_STRENGTH_LOOKBACK)
        bench_roc = pd.Series(np.nan, index=bars.index)

    candidates: list[dict[str, Any]] = []
    index = bars.index
    fee = max(float(cost_bps_per_side), 0.0) / 10000.0

    study_start = pd.Timestamp(study_start_date) if study_start_date else None
    study_end = pd.Timestamp(study_end_date) if study_end_date else None
    for i in np.flatnonzero((breakout & relative_ok).fillna(False).to_numpy()):
        signal_day = pd.Timestamp(index[i]).normalize()
        if study_start is not None and signal_day < study_start:
            continue
        if study_end is not None and signal_day >= study_end:
            continue
        entry_i = int(i) + 1
        exit_i = entry_i + HOLDING_DAYS - 1
        # Require full pre-breakout structure and a complete future holding period.
        if i < SETUP_LOW_LOOKBACK or exit_i >= len(bars):
            continue
        signal_ma = float(ma.iloc[i]) if pd.notna(ma.iloc[i]) else np.nan
        if not np.isfinite(signal_ma) or signal_ma <= 0:
            continue

        setup = bars.iloc[i - SETUP_LOW_LOOKBACK : i]
        setup_low_idx = setup["Low"].idxmin()
        setup_low_i = int(bars.index.get_loc(setup_low_idx))
        setup_low_price = float(bars.loc[setup_low_idx, "Low"])
        setup_low_ma = float(ma.iloc[setup_low_i]) if pd.notna(ma.iloc[setup_low_i]) else np.nan
        if not np.isfinite(setup_low_ma) or setup_low_ma <= 0:
            continue

        entry_price = float(bars["Open"].iloc[entry_i])
        exit_price = float(bars["Close"].iloc[exit_i])
        if not np.isfinite(entry_price) or not np.isfinite(exit_price) or entry_price <= 0:
            continue
        gross_return = exit_price / entry_price - 1.0
        net_return = (exit_price * (1.0 - fee)) / (entry_price * (1.0 + fee)) - 1.0

        close_dist = pct_distance(float(close.iloc[i]), signal_ma)
        low_dist = pct_distance(setup_low_price, setup_low_ma)
        candidates.append(
            {
                "Country": country,
                "Ticker": ticker,
                "Signal_Date": index[i].date().isoformat(),
                "Entry_Date": index[entry_i].date().isoformat(),
                "Exit_Date": index[exit_i].date().isoformat(),
                "Signal_Close": float(close.iloc[i]),
                "SMA50_On_Signal": signal_ma,
                "Breakout_Close_MA_Distance_Pct": close_dist,
                "Breakout_MA_Side": "Above" if close.iloc[i] >= signal_ma else "Below",
                "Setup_Low_Date": setup_low_idx.date().isoformat(),
                "Setup_Low_Price": setup_low_price,
                "SMA50_On_Setup_Low": setup_low_ma,
                "Setup_Low_MA_Distance_Pct": low_dist,
                "Setup_Low_MA_Side": "Above" if setup_low_price >= setup_low_ma else "Below",
                "Entry_Open": entry_price,
                "Exit_Close": exit_price,
                "Holding_Sessions": HOLDING_DAYS,
                "Gross_Return_Pct": gross_return * 100.0,
                "Net_Return_Pct": net_return * 100.0,
                "Stock_ROC_63D_Pct": float(stock_roc.iloc[i] * 100.0) if pd.notna(stock_roc.iloc[i]) else np.nan,
                "Benchmark_ROC_63D_Pct": float(bench_roc.iloc[i] * 100.0) if pd.notna(bench_roc.iloc[i]) else np.nan,
                "_Signal_Index": int(i),
                "_Entry_Index": entry_i,
                "_Exit_Index": exit_i,
            }
        )
    return candidates


def select_non_overlapping(candidates: list[dict[str, Any]], distance_field: str, cohort: str) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    by_ticker: dict[str, list[dict[str, Any]]] = {}
    for row in candidates:
        dist = row.get(distance_field, np.nan)
        if near_far(float(dist)) == cohort:
            by_ticker.setdefault(str(row["Ticker"]), []).append(row)
    for ticker_rows in by_ticker.values():
        ticker_rows.sort(key=lambda r: r["_Signal_Index"])
        last_exit_index = -1
        for row in ticker_rows:
            # If a breakout prints on the prior trade's exit date, it enters the next day,
            # after the old position has exited at that day's close; that is non-overlapping.
            if int(row["_Signal_Index"]) < last_exit_index:
                continue
            selected.append(row)
            last_exit_index = int(row["_Exit_Index"])
    selected.sort(key=lambda r: (r["Signal_Date"], r["Ticker"]))
    return selected


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


def write_equity_plot(curves: pd.DataFrame, test: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6.5))
    subset = curves[curves["Test"] == test] if not curves.empty else pd.DataFrame()
    if subset.empty:
        ax.text(0.5, 0.5, "No qualifying trades / no downloaded data", ha="center", va="center", transform=ax.transAxes)
    else:
        for cohort in ["Near_MA", "Far_MA"]:
            part = subset[subset["Cohort"] == cohort].sort_values("Date")
            if not part.empty:
                ax.plot(pd.to_datetime(part["Date"]), part["Equity"], label=cohort, linewidth=1.8)
    ax.set_title(f"Long breakout equity curve — {test.replace('_', ' ')}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity (start = 100)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def summarize_country(trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if trades.empty:
        return pd.DataFrame(columns=["Test", "Cohort", "Country", *trade_metrics([]).keys()])
    for (test, cohort, country), group in trades.groupby(["Test", "Cohort", "Country"], dropna=False):
        rows.append({"Test": test, "Cohort": cohort, "Country": country, **trade_metrics(group["Net_Return_Pct"] / 100.0)})
    return pd.DataFrame(rows).sort_values(["Test", "Cohort", "Country"]).reset_index(drop=True)


def compare_cohorts(trades: pd.DataFrame, performance: pd.DataFrame) -> pd.DataFrame:
    """Put near/far effect sizes side by side; differences are descriptive, not proof of significance."""
    rows: list[dict[str, Any]] = []
    for test_name in TESTS:
        near = trades[(trades["Test"] == test_name) & (trades["Cohort"] == "Near_MA")] if not trades.empty else pd.DataFrame()
        far = trades[(trades["Test"] == test_name) & (trades["Cohort"] == "Far_MA")] if not trades.empty else pd.DataFrame()
        near_ret = near["Net_Return_Pct"].astype(float) if not near.empty else pd.Series(dtype=float)
        far_ret = far["Net_Return_Pct"].astype(float) if not far.empty else pd.Series(dtype=float)
        near_win = float((near_ret > 0).mean() * 100.0) if len(near_ret) else np.nan
        far_win = float((far_ret > 0).mean() * 100.0) if len(far_ret) else np.nan
        near_avg = float(near_ret.mean()) if len(near_ret) else np.nan
        far_avg = float(far_ret.mean()) if len(far_ret) else np.nan
        perf_near = performance[(performance["Test"] == test_name) & (performance["Cohort"] == "Near_MA")]
        perf_far = performance[(performance["Test"] == test_name) & (performance["Cohort"] == "Far_MA")]
        near_total = float(perf_near["Portfolio_Total_Return_Pct"].iloc[0]) if not perf_near.empty else np.nan
        far_total = float(perf_far["Portfolio_Total_Return_Pct"].iloc[0]) if not perf_far.empty else np.nan
        rows.append({
            "Test": test_name,
            "Near_Trades": int(len(near_ret)),
            "Far_Trades": int(len(far_ret)),
            "Near_Avg_Trade_Return_Pct": near_avg,
            "Far_Avg_Trade_Return_Pct": far_avg,
            "Near_Minus_Far_Avg_Return_Pct_Points": near_avg - far_avg if np.isfinite(near_avg) and np.isfinite(far_avg) else np.nan,
            "Near_Win_Rate_Pct": near_win,
            "Far_Win_Rate_Pct": far_win,
            "Near_Minus_Far_Win_Rate_Pct_Points": near_win - far_win if np.isfinite(near_win) and np.isfinite(far_win) else np.nan,
            "Near_Portfolio_Total_Return_Pct": near_total,
            "Far_Portfolio_Total_Return_Pct": far_total,
            "Near_Minus_Far_Portfolio_Return_Pct_Points": near_total - far_total if np.isfinite(near_total) and np.isfinite(far_total) else np.nan,
        })
    return pd.DataFrame(rows)


def run_research(
    start_date: str,
    end_date: str,
    output_dir: Path,
    refresh_cache: bool,
    use_relative_strength_filter: bool,
    cost_bps_per_side: float,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "yahoo_cache"
    all_tickers: list[str] = []
    ticker_country: dict[str, str] = {}
    ticker_benchmark: dict[str, str] = {}
    for country, config in COUNTRY_BASKETS.items():
        for ticker in config["tickers"]:
            if ticker not in all_tickers:
                all_tickers.append(ticker)
            ticker_country[ticker] = country
        benchmark = config["benchmark"]
        if benchmark not in all_tickers:
            all_tickers.append(benchmark)
        ticker_benchmark[country] = benchmark

    # Pull pre-study warm-up and post-study bars so the last signals can finish their
    # 10-session holding period without being dropped just because the study window ended.
    warmup_start = (pd.Timestamp(start_date) - pd.Timedelta(days=365)).date().isoformat()
    data_end_date = (pd.Timestamp(end_date) + pd.Timedelta(days=EXIT_DATA_BUFFER_DAYS)).date().isoformat()
    print(f"Loading {len(all_tickers)} Yahoo Finance symbols (warm-up {warmup_start}; study signals {start_date} to {end_date}, end exclusive)...")
    try:
        price_by_ticker, manifest = load_yahoo_data(all_tickers, warmup_start, data_end_date, cache_dir, refresh_cache)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        print("Install dependencies with: py -m pip install -r requirements_ma_distance.txt")
        return 2
    manifest_path = output_dir / "data_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    valid_stocks = [t for t in ticker_country if t in price_by_ticker and not price_by_ticker[t].empty]
    if not valid_stocks:
        print("ERROR: No stock price histories were available. No performance results were generated.")
        print("The current runtime could not access Yahoo data. Run this script on your PC with internet access and yfinance installed.")
        print(f"Data manifest: {manifest_path}")
        return 2

    candidate_rows: list[dict[str, Any]] = []
    for country, config in COUNTRY_BASKETS.items():
        benchmark = price_by_ticker.get(config["benchmark"])
        for ticker in config["tickers"]:
            stock = price_by_ticker.get(ticker)
            if stock is None or stock.empty:
                continue
            events = make_signal_events(
                country, ticker, stock, benchmark, use_relative_strength_filter,
                cost_bps_per_side, study_start_date=start_date, study_end_date=end_date,
            )
            candidate_rows.extend(events)

    public_signal_columns = [
        "Country", "Ticker", "Signal_Date", "Entry_Date", "Exit_Date", "Signal_Close", "SMA50_On_Signal",
        "Breakout_Close_MA_Distance_Pct", "Breakout_MA_Side", "Setup_Low_Date", "Setup_Low_Price",
        "SMA50_On_Setup_Low", "Setup_Low_MA_Distance_Pct", "Setup_Low_MA_Side", "Entry_Open", "Exit_Close",
        "Holding_Sessions", "Gross_Return_Pct", "Net_Return_Pct", "Stock_ROC_63D_Pct", "Benchmark_ROC_63D_Pct",
    ]
    events_df = pd.DataFrame([{k: r.get(k, np.nan) for k in public_signal_columns} for r in candidate_rows], columns=public_signal_columns)
    events_df.to_csv(output_dir / "signal_events.csv", index=False)

    selected_records: list[dict[str, Any]] = []
    curve_frames: list[pd.DataFrame] = []
    stock_dates = [frame.index for ticker, frame in price_by_ticker.items() if ticker in ticker_country and not frame.empty]
    study_start_ts, curve_end_ts = pd.Timestamp(start_date), pd.Timestamp(data_end_date)
    all_dates = pd.DatetimeIndex(
        sorted(set(
            pd.Timestamp(d).normalize()
            for idx in stock_dates for d in idx
            if study_start_ts <= pd.Timestamp(d).normalize() < curve_end_ts
        )),
        name="Date",
    )
    for test_name, distance_field in TESTS.items():
        for cohort in ["Near_MA", "Far_MA"]:
            selected = select_non_overlapping(candidate_rows, distance_field, cohort)
            for row in selected:
                public = {k: row.get(k, np.nan) for k in public_signal_columns}
                public["Test"] = test_name
                public["Cohort"] = cohort
                selected_records.append(public)
            curve_frames.append(build_portfolio_curve(selected, price_by_ticker, all_dates, test_name, cohort, cost_bps_per_side))

    trade_columns = ["Test", "Cohort", *public_signal_columns]
    trades = pd.DataFrame(selected_records, columns=trade_columns)
    if not trades.empty:
        trades = trades.sort_values(["Test", "Cohort", "Signal_Date", "Ticker"]).reset_index(drop=True)
    trades.to_csv(output_dir / "trade_log.csv", index=False)

    curves = pd.concat(curve_frames, ignore_index=True) if curve_frames else pd.DataFrame(
        columns=["Date", "Test", "Cohort", "Daily_Return", "Equity", "Active_Positions"]
    )
    curves.to_csv(output_dir / "daily_equity_curves.csv", index=False, date_format="%Y-%m-%d")
    for test_name in TESTS:
        slug = test_name.lower()
        write_equity_plot(curves, test_name, output_dir / f"equity_curve_{slug}.png")

    performance_rows: list[dict[str, Any]] = []
    for test_name in TESTS:
        for cohort in ["Near_MA", "Far_MA"]:
            group_trades = trades[(trades["Test"] == test_name) & (trades["Cohort"] == cohort)] if not trades.empty else pd.DataFrame()
            group_curve = curves[(curves["Test"] == test_name) & (curves["Cohort"] == cohort)] if not curves.empty else pd.DataFrame()
            returns = group_trades["Net_Return_Pct"] / 100.0 if not group_trades.empty else pd.Series(dtype=float)
            performance_rows.append({
                "Test": test_name,
                "Cohort": cohort,
                **trade_metrics(returns),
                **portfolio_metrics(group_curve),
            })
    performance = pd.DataFrame(performance_rows)
    performance.to_csv(output_dir / "performance_summary.csv", index=False)
    comparison = compare_cohorts(trades, performance)
    comparison.to_csv(output_dir / "cohort_comparison.csv", index=False)
    country_summary = summarize_country(trades)
    country_summary.to_csv(output_dir / "country_summary.csv", index=False)
    if trades.empty:
        ticker_summary = pd.DataFrame(columns=["Test", "Cohort", "Country", "Ticker", *trade_metrics([]).keys()])
    else:
        ticker_rows = []
        for (test_name, cohort, country, ticker), group in trades.groupby(
            ["Test", "Cohort", "Country", "Ticker"], dropna=False
        ):
            ticker_rows.append({
                "Test": test_name, "Cohort": cohort, "Country": country, "Ticker": ticker,
                **trade_metrics(group["Net_Return_Pct"] / 100.0),
            })
        ticker_summary = pd.DataFrame(ticker_rows).sort_values(
            ["Test", "Cohort", "Country", "Ticker"]
        ).reset_index(drop=True)
    ticker_summary.to_csv(output_dir / "ticker_summary.csv", index=False)

    report = [
        "MOVING-AVERAGE DISTANCE VS. LONG BREAKOUT RESEARCH",
        "===================================================",
        f"Study date range: {start_date} to {end_date} (end date exclusive)",
        f"Historical indicator warm-up begins: {warmup_start}",
        f"MA: {MA_WINDOW}-session simple moving average",
        f"Breakout: daily close above the highest high of the previous {BREAKOUT_WINDOW} completed sessions",
        f"Near threshold: absolute distance <= {NEAR_MA_THRESHOLD_PCT:.2f}% of the MA; far = > threshold",
        f"Setup-low proxy: lowest low during the {SETUP_LOW_LOOKBACK} sessions before signal, measured from the MA on that low's date",
        f"Entry: next session open; exit: close of holding session {HOLDING_DAYS}",
        f"Relative-strength filter: {'ON' if use_relative_strength_filter else 'OFF'}",
        f"Transaction costs: {cost_bps_per_side:.2f} bps per side",
        "No stop-loss, slippage, market impact, or currency conversion. Corporate actions are adjusted by Yahoo Finance.",
        "Portfolio curves equal-weight active positions each day and hold cash when flat; there is no maximum concurrent-position limit.",
        "The setup-low metric is a repeatable proxy, not a verified Wyckoff LPS pattern detector.",
        "Use the CSVs and data_manifest.csv to check data coverage and stale-cache fallbacks before interpreting results.",
        "",
        "PERFORMANCE SUMMARY",
        performance.to_string(index=False) if not performance.empty else "No trades generated.",
        "",
        "NEAR VS FAR DIFFERENCE (descriptive; not a statistical-significance test)",
        comparison.to_string(index=False) if not comparison.empty else "No cohort comparison available.",
        "",
        f"Symbols with data: {len([x for x in manifest.to_dict('records') if x['Rows'] > 0])}/{len(manifest)}",
        f"Signals: {len(events_df)}; selected trades across both test variants: {len(trades)}",
    ]
    (output_dir / "research_report.txt").write_text("\n".join(report), encoding="utf-8")

    print("\nPERFORMANCE SUMMARY")
    if performance.empty:
        print("No trades generated. Check data_manifest.csv and the configured tickers/date range.")
    else:
        show = ["Test", "Cohort", "Trades", "Win_Rate_Pct", "Avg_Trade_Return_Pct", "Portfolio_Total_Return_Pct", "Max_Drawdown_Pct", "Sharpe_252"]
        print(performance[show].round(2).to_string(index=False))
        print("\nNEAR VS FAR DIFFERENCE (percentage points; descriptive only)")
        print(comparison.round(2).to_string(index=False))
    print(f"\nSymbols with data: {sum(manifest['Rows'] > 0)}/{len(manifest)}")
    print(f"Signals: {len(events_df)}; selected trades across both variants: {len(trades)}")
    print(f"Outputs saved to: {output_dir}")
    return 0


def run_self_test() -> int:
    """Offline deterministic test of column handling, signal timing, returns, and cohort labels."""
    rng = np.random.default_rng(17)
    n = 180
    dates = pd.bdate_range("2023-01-02", periods=n)
    close = 50.0 + np.cumsum(rng.normal(0.05, 0.5, n))
    # Create a clear breakout after enough history; adjust OHLC so it crosses prior highs.
    close[120] = max(close[100:120]) + 3.0
    for j in range(121, n):
        close[j] = close[j - 1] + 0.15 + rng.normal(0, 0.15)
    high = close + 0.4
    low = close - 0.4
    open_ = close - rng.normal(0, 0.15, n)
    bars = pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": 1000}, index=dates)
    # Confirm multi-index yfinance-style columns normalize correctly.
    multi = bars.copy()
    multi.columns = pd.MultiIndex.from_product([["SYN"], multi.columns])
    normalized = normalize_price_frame(multi)
    assert list(normalized.columns) == PRICE_COLUMNS
    assert len(normalized) == n

    events = make_signal_events("Test", "SYN", normalized, None, False, 0.0)
    assert events, "Synthetic data did not produce any breakout event"
    event = events[0]
    assert event["Entry_Date"] > event["Signal_Date"]
    assert event["Exit_Date"] > event["Entry_Date"]
    assert event["Holding_Sessions"] == HOLDING_DAYS
    assert event["Breakout_Close_MA_Distance_Pct"] >= 0
    assert event["Setup_Low_MA_Distance_Pct"] >= 0

    # Test both cohort definitions and the exact fields configured in TESTS.
    for test_name, distance_field in TESTS.items():
        assert distance_field in event, f"Configured distance field missing: {distance_field}"
        cohort = near_far(float(event[distance_field]))
        selected = select_non_overlapping(events, distance_field, cohort)
        assert selected, f"No selected trade for {test_name}/{cohort}"
        curve = build_portfolio_curve(selected, {"SYN": normalized}, normalized.index, test_name, cohort, 0.0)
        assert len(curve) == n
        assert np.isfinite(curve["Equity"]).all()

    print("SELF-TEST PASS")
    print(f"Normalized {len(normalized)} synthetic daily bars; generated {len(events)} breakout event(s).")
    print("Entry/exit timing, distance labels, and equity-curve generation passed.")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default=DEFAULT_START_DATE, help="YYYY-MM-DD; default 2021-01-01")
    parser.add_argument("--end-date", default=date.today().isoformat(), help="YYYY-MM-DD exclusive; default today")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output/cache directory")
    parser.add_argument("--refresh", action="store_true", help="Force full Yahoo Finance refresh, bypassing fresh cache")
    parser.add_argument("--relative-strength-filter", action="store_true", help="Require stock 63-session return > country benchmark return")
    parser.add_argument("--no-relative-strength-filter", action="store_true", help="Disable the relative-strength filter (default)")
    parser.add_argument("--cost-bps-per-side", type=float, default=TRANSACTION_COST_BPS_PER_SIDE, help="Estimated transaction cost per side in basis points; default 0")
    parser.add_argument("--self-test", action="store_true", help="Run offline deterministic checks without downloading data")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        return run_self_test()
    if args.start_date >= args.end_date:
        print("ERROR: --start-date must be earlier than --end-date.", file=sys.stderr)
        return 2
    use_rs = USE_RELATIVE_STRENGTH_FILTER_BY_DEFAULT
    if args.relative_strength_filter:
        use_rs = True
    if args.no_relative_strength_filter:
        use_rs = False
    return run_research(
        start_date=args.start_date,
        end_date=args.end_date,
        output_dir=Path(args.output_dir).expanduser().resolve(),
        refresh_cache=args.refresh,
        use_relative_strength_filter=use_rs,
        cost_bps_per_side=args.cost_bps_per_side,
    )


if __name__ == "__main__":
    raise SystemExit(main())
