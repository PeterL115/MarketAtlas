from __future__ import annotations

from pathlib import Path
from typing import Optional
import pandas as pd

from .io import ensure_dir


def cleanup_intraday_retention(intraday_dir: Path, retention_days: int, as_of_day: str) -> int:
    """
    Delete intraday parquet files older than (as_of_day - retention_days).
    Expects structure: intraday_dir/<ticker>/<YYYY-MM-DD>.parquet
    Returns number of files deleted.
    """
    if retention_days is None or retention_days <= 0:
        return 0

    cutoff = (pd.Timestamp(as_of_day) - pd.Timedelta(days=retention_days)).strftime("%Y-%m-%d")
    deleted = 0

    if not intraday_dir.exists():
        return 0

    for ticker_dir in intraday_dir.iterdir():
        if not ticker_dir.is_dir():
            continue

        for f in ticker_dir.glob("*.parquet"):
            # filename is YYYY-MM-DD.parquet
            day = f.stem
            if day < cutoff:
                try:
                    f.unlink()
                    deleted += 1
                except Exception:
                    # fail-soft; do not crash ingestion
                    pass

    return deleted
