from __future__ import annotations

from typing import Optional

import pandas as pd
import yfinance as yf

from ..timeutils import ensure_tz

def _clean_symbol(symbol: str) -> str:
    # yfinance does NOT want "$SPY" style symbols
    return str(symbol).strip().lstrip("$")


def _coerce_yf_date(x):
    if x is None:
        return None
    if isinstance(x, str) and "T" in x:
        return x.split("T")[0]
    return x


def _normalize_yf(df: pd.DataFrame, market_tz: str) -> pd.DataFrame:
    """
    Normalize yfinance download() output to canonical schema.
    Handles MultiIndex columns like ('Open','AAPL') or ('AAPL','Open').

    NOTE: This version is fine for DAILY bars where ts conversion is simpler.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()

    # --- Flatten MultiIndex columns (critical fix) ---
    if isinstance(out.columns, pd.MultiIndex):
        wanted = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
        flat_cols = []
        for col in out.columns:
            # col is a tuple like ('Open','AAPL') or ('AAPL','Open')
            if any(level in wanted for level in col):
                # choose the level that matches OHLCV label
                label = next(level for level in col if level in wanted)
                flat_cols.append(label)
            else:
                flat_cols.append("_".join(str(x) for x in col))
        out.columns = flat_cols

    out = out.reset_index()

    # Timestamp column can be named 'Date' or 'Datetime'
    if "Datetime" in out.columns:
        out.rename(columns={"Datetime": "ts"}, inplace=True)
    elif "Date" in out.columns:
        out.rename(columns={"Date": "ts"}, inplace=True)
    else:
        out.rename(columns={out.columns[0]: "ts"}, inplace=True)

    rename_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    }
    out.rename(columns=rename_map, inplace=True)

    keep = ["ts", "open", "high", "low", "close", "volume"]
    out = out[[c for c in keep if c in out.columns]].copy()

    # Coerce numeric safely (now guaranteed to be Series)
    for c in ["open", "high", "low", "close", "volume"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    # For daily, your existing ensure_tz logic is acceptable
    out["ts"] = ensure_tz(out["ts"], market_tz)
    out = out.dropna(subset=["ts", "open", "high", "low", "close"]).copy()

    return out.sort_values("ts").reset_index(drop=True)


def _normalize_yf_intraday(df: pd.DataFrame, market_tz: str) -> pd.DataFrame:
    """
    Normalize yfinance intraday (5m) output to canonical schema.

    CRITICAL FIX:
      - Force utc=True when parsing timestamps to avoid:
        "Tz-aware datetime.datetime cannot be converted to datetime64 unless utc=True"
      - Then convert to market_tz.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()

    # --- Flatten MultiIndex columns (same logic as daily) ---
    if isinstance(out.columns, pd.MultiIndex):
        wanted = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
        flat_cols = []
        for col in out.columns:
            if any(level in wanted for level in col):
                label = next(level for level in col if level in wanted)
                flat_cols.append(label)
            else:
                flat_cols.append("_".join(str(x) for x in col))
        out.columns = flat_cols

    out = out.reset_index()

    # Timestamp column can be named 'Date' or 'Datetime'
    if "Datetime" in out.columns:
        out.rename(columns={"Datetime": "ts"}, inplace=True)
    elif "Date" in out.columns:
        out.rename(columns={"Date": "ts"}, inplace=True)
    else:
        out.rename(columns={out.columns[0]: "ts"}, inplace=True)

    rename_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    }
    out.rename(columns=rename_map, inplace=True)

    keep = ["ts", "open", "high", "low", "close", "volume"]
    out = out[[c for c in keep if c in out.columns]].copy()

    # Coerce numeric safely
    for c in ["open", "high", "low", "close", "volume"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    # --- CRITICAL FIX: utc=True then convert to market_tz ---
    ts = pd.to_datetime(out["ts"], utc=True, errors="coerce")
    ts = ts.dt.tz_convert(market_tz)
    out["ts"] = ts

    out = out.dropna(subset=["ts", "open", "high", "low", "close"]).copy()
    return out.sort_values("ts").reset_index(drop=True)


class YFinanceProvider:
    def __init__(self, market_tz: str):
        self.market_tz = market_tz

    def fetch_daily(self, symbol: str, start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
        symbol = _clean_symbol(symbol)
        df = yf.download(
            tickers=symbol,
            start=start,
            end=end,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=True,
        )
        return _normalize_yf(df, self.market_tz)

    def fetch_intraday_5m(
        self,
        symbol: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        include_extended_hours: bool = True,
    ) -> pd.DataFrame:
        symbol = _clean_symbol(symbol)
        start = _coerce_yf_date(start)
        end = _coerce_yf_date(end)

        df = yf.download(
            tickers=symbol,
            start=start,
            end=end,
            interval="5m",
            prepost=bool(include_extended_hours),
            auto_adjust=False,
            progress=False,
            threads=True,
        )
        return _normalize_yf_intraday(df, self.market_tz)
