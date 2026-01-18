from __future__ import annotations

from typing import Any, Dict, Tuple

import json
from datetime import datetime, timezone


from MarketAtlas.core.config import AppConfig
from MarketAtlas.core.jsonio import atomic_write_json
from MarketAtlas.insights.intraday.insight_generator import generate_intraday_insight_one


def _json_default(o):
    # makes Timestamp / numpy / other odd types serializable
    try:
        # pandas Timestamp has .isoformat()
        return o.isoformat()
    except Exception:
        return str(o)

def _append_jsonl(path, obj: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, default=_json_default))
        f.write("\n")



def write_intraday_outputs(cfg: AppConfig, payload: Dict[str, Any]) -> Tuple[str, str]:
    ticker = str(payload["ticker"])
    day = str(payload["day"])

    latest_dir = cfg.path("outputs_intraday_latest_dir")
    log_dir = cfg.path("outputs_intraday_log_dir")
    latest_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    out_path = (latest_dir / f"{ticker}.json").resolve()
    log_path = (log_dir / f"{day}.jsonl").resolve()

    atomic_write_json(payload, out_path)
    _append_jsonl(log_path, payload)

    return str(out_path), str(log_path)


def intraday_insights_all(cfg: AppConfig, day: str) -> Dict[str, Any]:
    tickers = list(cfg.intraday_assets.keys())
    out: Dict[str, Any] = {"day": day, "results": {}}

    for t in tickers:
        try:
            payload = generate_intraday_insight_one(cfg, ticker=t, day=day)
            out_json, log_jsonl = write_intraday_outputs(cfg, payload)
            out["results"][t] = {
                "ticker": t,
                "day": day,
                "status": "ok",
                "asof_ts": payload.get("asof_ts"),
                "out_json": out_json,
                "log_jsonl": log_jsonl,
            }
        except Exception as e:
            out["results"][t] = {"ticker": t, "day": day, "status": "error", "error": str(e)}

    return out


def intraday_insights_one(cfg: AppConfig, ticker: str, day: str) -> Dict[str, Any]:
    payload = generate_intraday_insight_one(cfg, ticker=ticker, day=day)
    out_json, log_jsonl = write_intraday_outputs(cfg, payload)
    return {
        "ticker": ticker,
        "day": day,
        "status": "ok",
        "asof_ts": payload.get("asof_ts"),
        "out_json": out_json,
        "log_jsonl": log_jsonl,
    }
