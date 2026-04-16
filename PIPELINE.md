# MarketAtlas Pipeline Reference

This document is an operator/developer index for:
- which CLI step generates which data
- input dependencies per step
- where outputs are written (resolved via config paths)
- how to verify artifacts exist for a given ticker/date

## Universe + Symbols

Configured in: `config/app.yaml` (or your config file)
- `assets.daily`: mapping { ticker -> provider_symbol }
- `assets.intraday_5m`: mapping { ticker -> provider_symbol }

Examples:
- SPX -> ^GSPC
- VIX -> ^VIX
- VXX -> VXX

## Path Conventions (from `AppConfig.path(key)`)

Common keys (names may vary by your implementation):
- `daily_dir`: daily parquet store (e.g., `data/daily/`)
- `intraday_dir`: intraday 5m parquet store (e.g., `data/intraday_5m/`)
- `outputs_dir` / `reports_dir`: JSON outputs

When in doubt, run:
`python tools/locate_artifacts.py --root MarketAtlas --date YYYY-MM-DD --ticker SPX`

---

## Step 1: ingest-daily

CLI:
- `MarketAtlas ingest-daily [--start YYYY-MM-DD] [--end YYYY-MM-DD]`

Code:
- `orchestrators/ingest_daily.py`
- Provider: `core/providers/yfinance_provider.py`

Inputs:
- none (hits provider)

Outputs:
- `{daily_dir}/{TICKER}.parquet`

Validation:
- parquet exists
- last row timestamp is recent

---

## Step 2: ingest-5m

CLI:
- `MarketAtlas ingest-5m --date YYYY-MM-DD [--lookback N] [--no-extended]`

Code:
- `orchestrators/ingest_intraday_5m.py`

Inputs:
- none (hits provider)

Outputs:
- `{intraday_dir}/{TICKER}/{YYYY-MM-DD}.parquet` (+ lookback days)

Notes:
- This step must NOT pass ISO strings like `YYYY-MM-DDT00:00:00-05:00` to yfinance.
  Use `YYYY-MM-DD HH:MM:SS` (naive) to avoid parsing errors.

---

## Step 3: session-state

CLI:
- `MarketAtlas session-state --date YYYY-MM-DD`

Code:
- `orchestrators/update_session_state.py`
- Core: `core/session_state.py`

Inputs:
- intraday 5m parquet for the day

Outputs:
- session state JSON (exact path depends on your config/core writer)

---

## Step 4: levels

CLI:
- `MarketAtlas levels --date YYYY-MM-DD`

Code:
- `orchestrators/compute_levels.py`
- Core: `core/levels.py`

Inputs:
- daily parquet
- session state JSON

Outputs:
- levels JSON:
  - "latest" path (per ticker)
  - "snapshot" path (per day)
  (exact filenames returned by `write_levels()`)

---

## Step 5: intraday-features

CLI:
- `MarketAtlas intraday-features --date YYYY-MM-DD [--ticker TICKER]`

Code:
- `orchestrators/intraday_features.py`

Inputs:
- intraday 5m parquet
- session state (if used)
- levels (if used)

Outputs:
- feature artifacts (parquet/json depending on implementation)

---

## Step 6: today-range

CLI:
- `MarketAtlas today-range --date YYYY-MM-DD [--ticker TICKER]`

Code:
- `orchestrators/today_range.py`

Inputs:
- intraday features
- intraday bars (possibly)
- levels (possibly)

Outputs:
- today range JSON artifacts (latest/snapshot depending on writer)

---

## Step 7a: intraday-insights

CLI:
- `MarketAtlas intraday-insights --date YYYY-MM-DD [--ticker TICKER]`

Code:
- `orchestrators/intraday_insights.py`

Inputs:
- today-range outputs
- diagnostics/features/levels (depending on implementation)

Outputs:
- intraday insights JSON (consumed by UI as “Intraday Latest”)

---

## Step 7: intraday-run (full intraday pipeline)

CLI:
- `MarketAtlas intraday-run --date YYYY-MM-DD [--lookback N] [--no-extended] [--ticker TICKER]`

Code:
- `orchestrators/intraday_run.py`

Behavior:
- runs steps 2 → 7a in order
- if `--ticker` is provided, ticker-scoped stages should run ONLY that ticker

---

## Step 8: daily-features

CLI:
- `MarketAtlas daily-features --date YYYY-MM-DD [--ticker TICKER]`

Code:
- `orchestrators/daily_features.py`

Inputs:
- daily parquet

Outputs:
- daily features artifacts for training/EOD

---

## Step 9: regime-labels

CLI:
- `MarketAtlas regime-labels --date YYYY-MM-DD [--ticker TICKER]`

Code:
- `orchestrators/regime_labels.py`

Inputs:
- daily features

Outputs:
- labels artifacts

---

## Step 10: train-spx-regime

CLI:
- `MarketAtlas train-spx-regime --date YYYY-MM-DD --version v1`

Code:
- `orchestrators/train_spx_regime.py`

Inputs:
- SPX daily features + labels

Outputs:
- model artifacts under a versioned folder

---

## Step 11: train-m7-regime

CLI:
- `MarketAtlas train-m7-regime --date YYYY-MM-DD --version v1 [--ticker NVDA]`

Code:
- `orchestrators/train_m7_regime.py`

Inputs:
- M7 tickers daily features + labels

Outputs:
- model artifacts per ticker/version

---

## Step 12: tomorrow-range

CLI:
- `MarketAtlas tomorrow-range --date YYYY-MM-DD [--ticker TICKER]`

Code:
- `orchestrators/tomorrow_range.py`

Inputs:
- daily features
- regime models

Outputs:
- tomorrow range forecast artifacts

---

## Step 13: eod-run

CLI:
- `MarketAtlas eod-run --date YYYY-MM-DD [--ticker TICKER]`

Code:
- `orchestrators/eod_run.py`

Inputs:
- tomorrow-range outputs

Outputs:
- EOD plan JSON (consumed by UI Tomorrow Plan screen)

---

## Debug Checklist

1) Verify intraday parquet exists: `{intraday_dir}/{TICKER}/{DATE}.parquet`
2) Inspect last timestamp in parquet (is it stale?)
3) Verify “latest” JSON exists for intraday/eod outputs
4) If single-ticker still runs all:
   - ensure all orchestrators accept `tickers=...` and filter
   - ensure CLI subcommand supports `--ticker` where needed
