from __future__ import annotations

from typing import Any, Dict, Optional

from MarketAtlas.core.config import AppConfig
from MarketAtlas.features.intraday.pipeline import (
    IntradayFeatureParams,
    compute_intraday_features_one,
    write_intraday_features,
)


def build_intraday_features_one(cfg: AppConfig, ticker: str, day: str) -> Dict[str, Any]:
    df, snap = compute_intraday_features_one(cfg, ticker=ticker, day=day, params=IntradayFeatureParams())
    feat_path, latest_path = write_intraday_features(cfg, ticker, day, df, snap)
    return {
        "ticker": ticker,
        "day": day,
        "rows": int(len(df)),
        "asof_ts": snap.get("asof_ts"),
        "feature_parquet": feat_path,
        "latest_json": latest_path,
        "status": "ok",
    }


def build_intraday_features_all(cfg: AppConfig, day: str) -> Dict[str, Any]:
    tickers = getattr(cfg, "intraday_assets", None) or getattr(cfg, "intraday_tickers", None)
    if not tickers:
        raise ValueError("No intraday tickers configured (expected cfg.intraday_assets).")

    out: Dict[str, Any] = {"day": day, "results": {}}
    for t in tickers:
        try:
            out["results"][t] = build_intraday_features_one(cfg, t, day)
        except Exception as e:
            out["results"][t] = {"ticker": t, "day": day, "status": "error", "error": str(e)}
    return out
