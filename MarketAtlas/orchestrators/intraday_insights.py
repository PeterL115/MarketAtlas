from __future__ import annotations

from typing import Any, Dict, Tuple
import json

from MarketAtlas.core.config import AppConfig
from MarketAtlas.core.jsonio import atomic_write_json
from MarketAtlas.insights.intraday.insight_generator import generate_intraday_insight_one

from MarketAtlas.core.providers.yfinance_provider import get_iv_snapshot_yahoo


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


def _inject_iv_into_payload(cfg: AppConfig, payload: Dict[str, Any]) -> None:
    """
    Best-effort:
    - compute IV from Yahoo options chain
    - write into payload["diagnostics"]["iv"] and payload["features"]["iv"]
    - keep payload["views"] (zh/both/en) consistent if present

    If IV unavailable (e.g. some indices), write None and move on.
    """
    try:
        ticker = str(payload.get("ticker") or "")
        if not ticker:
            return

        # intraday_assets: mapping { ticker -> provider_symbol }
        provider_symbol = None
        try:
            provider_symbol = (getattr(cfg, "intraday_assets", {}) or {}).get(ticker)
        except Exception:
            provider_symbol = None

        symbol = str(provider_symbol or ticker)

        iv = get_iv_snapshot_yahoo(symbol, target_dte=30)

        # diagnostics
        diagnostics = payload.get("diagnostics")
        if not isinstance(diagnostics, dict):
            diagnostics = {}
            payload["diagnostics"] = diagnostics
        diagnostics["iv"] = iv  # 0..1 or None

        # features
        features = payload.get("features")
        if not isinstance(features, dict):
            features = {}
            payload["features"] = features
        features["iv"] = iv

        # views
        views = payload.get("views")
        if isinstance(views, dict):
            zh = views.get("zh")
            if isinstance(zh, dict):
                zh_diag = zh.get("诊断指标")
                if isinstance(zh_diag, dict):
                    zh_diag["隐含波动率(IV)"] = iv

            both = views.get("both")
            if isinstance(both, dict):
                both_diag = both.get("Diagnostics/诊断指标")
                if isinstance(both_diag, dict):
                    both_diag["Implied Vol (IV)/隐含波动率(IV)"] = iv

            en = views.get("en")
            if isinstance(en, dict):
                en_diag = en.get("Diagnostics")
                if isinstance(en_diag, dict):
                    en_diag["Implied Vol (IV)"] = iv

    except Exception:
        # IV 不应影响主流程（insights 仍然要能产出）
        diagnostics = payload.get("diagnostics")
        if not isinstance(diagnostics, dict):
            diagnostics = {}
            payload["diagnostics"] = diagnostics
        diagnostics.setdefault("iv", None)

        features = payload.get("features")
        if not isinstance(features, dict):
            features = {}
            payload["features"] = features
        features.setdefault("iv", None)

        views = payload.get("views")
        if isinstance(views, dict):
            zh = views.get("zh")
            if isinstance(zh, dict):
                zh_diag = zh.get("诊断指标")
                if isinstance(zh_diag, dict):
                    zh_diag["隐含波动率(IV)"] = None

            both = views.get("both")
            if isinstance(both, dict):
                both_diag = both.get("Diagnostics/诊断指标")
                if isinstance(both_diag, dict):
                    both_diag["Implied Vol (IV)/隐含波动率(IV)"] = None

            en = views.get("en")
            if isinstance(en, dict):
                en_diag = en.get("Diagnostics")
                if isinstance(en_diag, dict):
                    en_diag["Implied Vol (IV)"] = None


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

            # inject IV (and keep views consistent)
            _inject_iv_into_payload(cfg, payload)

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

    # inject IV (and keep views consistent)
    _inject_iv_into_payload(cfg, payload)

    out_json, log_jsonl = write_intraday_outputs(cfg, payload)
    return {
        "ticker": ticker,
        "day": day,
        "status": "ok",
        "asof_ts": payload.get("asof_ts"),
        "out_json": out_json,
        "log_jsonl": log_jsonl,
    }