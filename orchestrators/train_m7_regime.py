from __future__ import annotations

from typing import Any, Dict, List, Optional

from MarketAtlas.core.config import AppConfig
from MarketAtlas.training.m7_regime.train_m7_regime import train_one_ticker_regime, TrainParams


M7 = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA"]


def train_m7_regime_models(
    cfg: AppConfig,
    day: str,
    version: str,
    ticker: Optional[str] = None,
    params: Optional[TrainParams] = None,
) -> Dict[str, Any]:
    tickers: List[str] = [ticker] if ticker else list(M7)

    out: Dict[str, Any] = {"day": day, "version": version, "results": {}}

    for t in tickers:
        try:
            res = train_one_ticker_regime(cfg, ticker=t, day=day, version=version, params=params)
            out["results"][t] = res
        except Exception as e:
            out["results"][t] = {"status": "error", "ticker": t, "day": day, "error": str(e)}

    return out
