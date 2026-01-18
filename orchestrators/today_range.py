from __future__ import annotations

from typing import Any, Dict

from MarketAtlas.core.config import AppConfig
from MarketAtlas.insights.intraday.today_range import nowcast_today_range_one, write_today_range


def nowcast_today_range_all(cfg: AppConfig, day: str) -> Dict[str, Any]:
    assets = cfg.intraday_assets  # dict {ticker: provider_symbol}
    out: Dict[str, Any] = {"day": day, "results": {}}

    for ticker in assets.keys():
        try:
            payload = nowcast_today_range_one(cfg, ticker=ticker, day=day)
            out_path, log_path = write_today_range(cfg, payload)
            out["results"][ticker] = {
                "ticker": ticker,
                "day": day,
                "status": "ok",
                "asof_ts": payload.get("asof_ts"),
                "out_json": out_path,
                "log_jsonl": log_path,
            }
        except Exception as e:
            out["results"][ticker] = {"ticker": ticker, "day": day, "status": "error", "error": str(e)}

    return out


def nowcast_today_range_one_ticker(cfg: AppConfig, ticker: str, day: str) -> Dict[str, Any]:
    payload = nowcast_today_range_one(cfg, ticker=ticker, day=day)
    out_path, log_path = write_today_range(cfg, payload)
    return {
        "ticker": ticker,
        "day": day,
        "status": "ok",
        "asof_ts": payload.get("asof_ts"),
        "out_json": out_path,
        "log_jsonl": log_path,
    }
