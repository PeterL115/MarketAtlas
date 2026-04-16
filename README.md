# MarketAtlas

A quantitative market analytics platform for options traders, focused on intraday and end-of-day insights for SPX and major US equities. Built entirely in Python with a modular 13-step data pipeline, machine learning regime classification, pivot level computation, and a real-time Streamlit dashboard with full Chinese localization.

---

## Overview

MarketAtlas ingests live market data, computes volatility and trend features, classifies market regimes, forecasts expected price ranges, computes pivot levels, and generates actionable options trade guidance — all from a single CLI or web interface.

It covers **SPX, SPY, VIX, VXX**, and the **Magnificent 7** (AAPL, MSFT, AMZN, GOOGL, META, NVDA, TSLA).

---

## Key Features

### Data Pipeline
- Automated ingestion of daily OHLCV bars (back to 2015) and 5-minute intraday bars via Yahoo Finance
- Robust retry/backoff logic with rate-limit detection and exponential jitter
- Atomic file writes and timestamp-based deduplication to prevent data corruption
- 60-day intraday data retention with automatic cleanup
- Full timezone awareness (America/New_York) throughout

### Session State & Market Structure
- Incremental VWAP computation per session (PRE / RTH / AFT)
- Opening range tracking (first 30 minutes of RTH)
- Session high/low/close aggregation updated bar-by-bar

### Support & Resistance Levels
- Prior day and prior week high/low/close
- Swing pivot fractals on 120-day lookback
- Intraday anchors: RTH VWAP, opening range, session extremes
- Zones sized by ATR; ranked by strength and proximity; capped at 40 levels

### Pivot Levels (CPR + Classic)
- **Central Pivot Range (CPR):** Pivot, Bottom Central (BC), Top Central (TC) — computed from prior-day H/L/C
- **Classic Daily Pivots:** R3, R2, R1, Pivot, S1, S2, S3
- Automatically attached to every intraday insight and EOD plan payload
- UI displays the session the levels apply to ("For 2026-04-16") alongside the source date ("Based on 2026-04-15")

### Feature Engineering
- **Intraday (5-minute):** Realized volatility, Efficiency Ratio (ER), bar overlap, level pressure, range position, relative strength vs SPY
- **Daily:** ATR (14), 20-day realized vol, ER, linear regression slope, gap size, HH/LL position, volume z-score

### Regime Classification
- **Rule-based:** `trend_up` / `trend_down` / `range` / `high_vol` / `transition` using ER, normalized slope, and range/ATR thresholds
- **ML layer:** Logistic regression models trained per ticker (SPX + M7) for 3-class regime prediction
- Models serialized with joblib; versioned output directory

### Volatility Forecasting
- **Intraday remaining-session range** — blends 5-minute realized vol, daily ATR, and live options IV using variance blending with U-shaped intraday weights (higher vol at open/close, lower midday)
- **Tomorrow's range forecast** — blends IV (55%), RV (20%), ATR (25%) with regime-dependent multipliers; outputs 45% / 68% / 95% confidence bands using proper normal distribution z-scores

### Options Guidance Engine
- Compares implied volatility (IV) against realized volatility (RV) to classify premium as **rich / neutral / cheap**
- Recommends specific option structures: Iron Condor, Bull/Bear Put/Call Spread, Short Strangle, or Debit Spread based on regime, premium environment, and directional bias
- Generates strike placement guidance directly from forecasted range bands
- Flags gamma risk (≤6 bars remaining), elevated vol regimes, and gap risk

### Output & Logging
- All outputs written as structured JSON with bilingual support (English / Chinese)
- Append-only JSONL logs for full intraday and EOD audit trails
- Snapshot + latest pattern for every data type (always queryable without date)

### Dashboard (Streamlit)
- **Intraday Live** screen: range forecast, movement likelihoods, pivot levels, key levels, options guidance, diagnostics
- **Tomorrow Plan (EOD)** screen: next-day range forecast, pivot levels, options guidance
- Pivot levels panel displays CPR (中枢轴心区间) and Classic Pivots (经典轴心点位) side-by-side with the range forecast
- Run controls: **Run Intraday Pipeline** and **Run EOD Pipeline** both automatically execute Step 1 (ingest-daily) before the main pipeline
- Full **Chinese localization**: switching the language selector to `zh` or `both` renders all UI labels in professional Chinese financial terminology (阻力位、支撑位、轴心点、铁鹰式价差, etc.)

### CLI
13 composable subcommands covering every pipeline stage individually or end-to-end:

```
python -m MarketAtlas.app.cli intraday-run --date 2024-01-08
python -m MarketAtlas.app.cli eod-run --date 2024-01-08 --ticker SPX
```

---

## Tech Stack

| Layer | Technologies |
|---|---|
| Language | Python 3.10+ |
| Data | pandas, numpy, yfinance, pyarrow |
| ML | scikit-learn (LogisticRegression), joblib |
| UI | Streamlit |
| Storage | Parquet (time-series), JSON (snapshots/insights) |
| Config | YAML |

---

## Architecture

```
MarketAtlas/
├── core/               # Config, I/O, session state, levels, pivot_levels, providers
├── features/           # Intraday and daily feature pipelines
├── insights/           # Range forecasting (intraday + EOD), i18n views
├── regimes/            # Rule-based regime labeler
├── training/           # ML model training (SPX + M7)
├── orchestrators/      # Pipeline step orchestration (13 steps)
├── app/                # CLI entrypoint + UI helpers
└── config/             # app.yaml
```

### Pipeline (13 Steps)

```
Step 1  ingest-daily          Daily OHLCV bars (2015–present)
Step 2  ingest-5m             5-minute intraday bars
Step 3  session-state         VWAP, opening range, session H/L/C
Step 4  levels                Support/resistance zone computation
Step 5  intraday-features     Per-bar feature vectors
Step 6  today-range           Remaining-session range forecast (45/68/95%)
Step 7  intraday-insights     Full intraday insight payload + pivot levels
Step 8  daily-features        Daily feature vectors
Step 9  regime-labels         Rule-based regime classification
Step 10 train-spx-regime      Logistic regression on SPX
Step 11 train-m7-regime       Logistic regression per M7 ticker
Step 12 tomorrow-range        Next-day range forecast (45/68/95%)
Step 13 eod-run               EOD plan: options guidance + pivot levels + strike placement
```

---

## Getting Started

### Requirements
```
pip install pandas numpy yfinance pyarrow scikit-learn streamlit pyyaml pandas-market-calendars
```

### Configuration
Edit `config/app.yaml` to set your data paths and asset universe.

### Run the intraday pipeline
```bash
python -m MarketAtlas.app.cli ingest-daily
python -m MarketAtlas.app.cli intraday-run --date 2024-01-08
```

### Run the EOD pipeline
```bash
python -m MarketAtlas.app.cli ingest-daily
python -m MarketAtlas.app.cli eod-run --date 2024-01-08
```

### Run a single step for one ticker
```bash
python -m MarketAtlas.app.cli levels --date 2024-01-08 --ticker SPX
python -m MarketAtlas.app.cli regime-labels --date 2024-01-08 --ticker SPX
```

### Launch the dashboard
```bash
run_ui.bat
# or
streamlit run MarketAtlas/app/ui_streamlit.py
```

---

## Notes

- The Streamlit UI file (`app/ui_streamlit.py`) is excluded from this repository.
- Generated data directories (`data/`, `features/`, `outputs/`, `models/`) are excluded and must be populated by running the pipeline locally.

---

## Skills Demonstrated

- End-to-end system design: data ingestion → feature engineering → ML → actionable output
- Time-series data engineering with production-grade patterns (atomic writes, deduplication, retention)
- Quantitative finance: volatility modeling, pivot level computation, options pricing intuition, regime detection
- Machine learning pipeline: feature construction, model training, versioned serialization
- CLI and API design with clean separation of concerns across 13 modules
- Bilingual (English/Chinese) UI with professional financial terminology localization
