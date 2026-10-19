# Moving-Average Distance vs. Long Breakout Research

A reproducible, research-grade backtest of **long-only daily breakouts** conditioned on **moving-average distance** and **bullish MA structure** across six national equity baskets.
![image](.png)
> **Not investment advice.** All results are research estimates on adjusted, local-currency Yahoo Finance data. No slippage, taxes, FX conversion, or market impact is modelled. Transaction costs default to zero.

---

## What This Repo Does

Two independent research scripts answer two related questions:

| Question | Script | Output directory |
|---|---|---|
| Do breakouts behave differently when price is **near vs. far from the 50-SMA**? | `ma_distance_research.py` | `outputs/ma_distance_research_output/` |
| Do **bullish MA stacks, price surfing, and LPS-style setup lows** add value over raw breakouts? | `ma_structure_research.py` | `outputs/ma_structure_research_output/` |

Both scripts use the same data pipeline, same six country baskets, same breakout definition, same 10-session holding period, and same non-overlapping trade rule — so results are directly comparable.

---

## Strategy Rules (Mechanical, Not Discretionary)

### Common rules (both scripts)

- **Breakout signal:** daily close > highest **high** of the previous 20 completed sessions (today's bar excluded).
- **Entry:** next session's **open**.
- **Exit:** close of the **10th holding session**.
- **No stop-loss, no leverage.**
- **No overlapping trades** within a ticker/cohort/strategy.
- **Signals evaluated only inside the study window**; warm-up and exit-buffer data are downloaded outside it.
- **Relative-strength filter:** off by default (isolates MA distance/structure as the variable). Optional 63-session stock vs. benchmark ROC.

### `ma_distance_research.py` — two MA-distance definitions

1. **Breakout-close distance:** absolute distance of the breakout close from the 50-SMA on signal day.
2. **Setup-low distance:** absolute distance of the lowest low in the 10 sessions before the breakout from the 50-SMA **on that low's date** — a mechanical, no-lookahead proxy for an LPS/support zone, **not** an automated Wyckoff label.

Cohorts: `Near_MA` (≤ 3%), `Far_MA` (> 3%).

### `ma_structure_research.py` — five nested strategies

| Strategy | Rule |
|---|---|
| `All_Breakouts` | Baseline — every breakout |
| `Bullish_Stack` | SMA10 > SMA21 > SMA50 > SMA200 on signal day **and** on ≥ 8 of the previous 10 sessions |
| `Stacked_and_Surfing` | Bullish stack **and** close within ±3% of SMA10/21/50 and not >3% below SMA50 on ≥ 6 of the previous 10 sessions |
| `LPS_Proxy` | Lowest low in the previous 10 sessions within ±3% of SMA50 on that low's date |
| `Stacked_LPS_Proxy` | Bullish stack **and** LPS proxy |

---

## Country Baskets

Ten liquid large-caps per basket, with a local benchmark for the optional relative-strength filter:

| Country | Benchmark | Tickers |
|---|---|---|
| United States | `SPY` | AAPL, MSFT, NVDA, AMZN, JPM, GOOGL, META, AMD, NFLX, CAT |
| Germany | `^GDAXI` | SAP.DE, SIE.DE, ALV.DE, DTG.DE, MBG.DE, BMW.DE, BAS.DE, BAYN.DE, ADS.DE, DTE.DE |
| France | `^FCHI` | OR.PA, MC.PA, TTE.PA, SAN.PA, AIR.PA, SU.PA, BNP.PA, EL.PA, RMS.PA, DG.PA |
| Japan | `^N225` | 7203.T, 6758.T, 9983.T, 6861.T, 8306.T, 8035.T, 6501.T, 7974.T, 9984.T, 4063.T |
| Canada | `^GSPTSE` | RY.TO, TD.TO, SHOP.TO, ENB.TO, CNR.TO, CP.TO, BNS.TO, BMO.TO, TRI.TO, SU.TO |
| South Korea | `^KS11` | 005930.KS, 000660.KS, 005380.KS, 000270.KS, 005490.KS, 035420.KS, 035720.KS, 051910.KS, 006400.KS, 105560.KS |

---

## Quick Start

### 1. Install dependencies

```bash
python -m pip install -r requirements.txt
