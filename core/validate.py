from __future__ import annotations

from typing import Dict, List

import pandas as pd

from .constants import REQUIRED_BAR_COLS


def validate_bars(df: pd.DataFrame, require_volume: bool = True) -> Dict[str, object]:
    """
    Validate canonical OHLCV bars.
    Raises ValueError on hard failures; returns a small report dict for logging.
    """
    if df is None or df.empty:
        raise ValueError("Bars DataFrame is empty.")

    if "ts" not in df.columns:
        raise ValueError("Bars DataFrame must include 'ts' column.")

    missing = [c for c in REQUIRED_BAR_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if require_volume and df["volume"].isna().any():
        raise ValueError("Volume contains NaNs.")

    # Timestamp checks
    ts = pd.DatetimeIndex(df["ts"])
    if ts.tz is None:
        raise ValueError("Timestamps must be timezone-aware.")
    if not ts.is_monotonic_increasing:
        # Not fatal if provider returns unsorted; we will sort, but flag it.
        monotonic = False
    else:
        monotonic = True

    dupes = df["ts"].duplicated().sum()

    # OHLC sanity
    bad_ohlc = int(((df["high"] < df["low"]) | (df["open"] > df["high"]) | (df["open"] < df["low"]) |
                    (df["close"] > df["high"]) | (df["close"] < df["low"])).sum())

    report = {
        "rows": int(len(df)),
        "monotonic_ts": bool(monotonic),
        "duplicate_ts": int(dupes),
        "bad_ohlc_rows": int(bad_ohlc),
    }

    if bad_ohlc > 0:
        raise ValueError(f"Found {bad_ohlc} rows with invalid OHLC relationships.")

    return report
