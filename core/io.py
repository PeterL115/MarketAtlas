from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    """
    Atomic write to avoid partial files.
    """
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def merge_dedupe_by_ts(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """
    Merge two bar frames and de-duplicate on 'ts' (keep last).
    Always returns sorted by ts.
    """
    if existing is None or existing.empty:
        out = new.copy()
    else:
        out = pd.concat([existing, new], axis=0, ignore_index=True)

    out = out.drop_duplicates(subset=["ts"], keep="last")
    out = out.sort_values("ts").reset_index(drop=True)
    return out


def last_timestamp(df: pd.DataFrame, ts_col: str = "ts", market_tz: str = "America/New_York") -> Optional[pd.Timestamp]:
    """
    Return latest timestamp in df[ts_col] as a tz-aware pandas Timestamp in market_tz.

    Robust to:
      - tz-aware datetime.datetime objects
      - pandas Timestamp
      - ISO strings (with or without offset)
      - mixed/object dtype
      - naive datetimes (assumed UTC during coercion, then converted)
    """
    if df is None or df.empty or ts_col not in df.columns:
        return None

    s = df[ts_col]

    # If already tz-aware datetime64[ns, tz], max() is safe
    if pd.api.types.is_datetime64tz_dtype(s):
        mx = s.max()
        if pd.isna(mx):
            return None
        try:
            return pd.Timestamp(mx).tz_convert(market_tz)
        except Exception:
            return pd.Timestamp(mx)

    # Otherwise coerce via UTC to avoid:
    # "Tz-aware datetime.datetime cannot be converted to datetime64 unless utc=True"
    ts_utc = pd.to_datetime(s, errors="coerce", utc=True)
    if ts_utc is None or ts_utc.isna().all():
        return None

    mx_utc = ts_utc.max()
    if pd.isna(mx_utc):
        return None

    try:
        return pd.Timestamp(mx_utc).tz_convert(market_tz)
    except Exception:
        return pd.Timestamp(mx_utc)
