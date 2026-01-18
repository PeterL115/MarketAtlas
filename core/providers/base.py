from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd


class MarketDataProvider(ABC):
    """
    Provider interface. Implementations must return canonical schema:

    Columns: ts, open, high, low, close, volume (optional vwap)
    ts must be timezone-aware and in market timezone after normalization.
    """

    @abstractmethod
    def fetch_daily(
        self,
        symbol: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def fetch_intraday_5m(
        self,
        symbol: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        include_extended_hours: bool = True,
    ) -> pd.DataFrame:
        raise NotImplementedError
