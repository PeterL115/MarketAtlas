from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, List

from ..core.config import AppConfig
from ..orchestrators.ingest_intraday_5m import ingest_intraday_for_date
from ..orchestrators.update_session_state import update_session_state_all
from ..orchestrators.compute_levels import compute_levels_all
from ..orchestrators.intraday_features import build_intraday_features_all, build_intraday_features_one
from ..orchestrators.today_range import nowcast_today_range_all, nowcast_today_range_one_ticker
from ..orchestrators.intraday_insights import intraday_insights_all, intraday_insights_one


@dataclass
class IntradayRunResult:
    day: str
    status: str
    error: Optional[str]
    stages: Dict[str, Any]


def intraday_run(
    cfg: AppConfig,
    day: str,
    lookback_days: Optional[int] = None,
    include_extended_hours: bool = True,
    ticker: Optional[str] = None,
) -> IntradayRunResult:
    """
    Full intraday pipeline.
    If ticker is provided, run ONLY that ticker for ticker-scoped stages.
    """
    stages: Dict[str, Any] = {}

    try:
        lb = lookback_days
        if lb is None:
            lb = int(cfg.raw.get("ingestion", {}).get("intraday_5m", {}).get("default_lookback_days", 5))

        tickers: Optional[List[str]] = [ticker.strip()] if (ticker and ticker.strip()) else None

        # Step 2: ingest 5m
        stages["ingest_5m"] = {
            "lookback_days": lb,
            "include_extended_hours": include_extended_hours,
            "ticker": tickers[0] if tickers else None,
        }
        ingest_res = ingest_intraday_for_date(
            cfg,
            day=day,
            lookback_days=lb,
            include_extended_hours=include_extended_hours,
            tickers=tickers,
        )
        stages["ingest_5m_result_n"] = len(ingest_res)

        # Step 3: session state
        stages["session_state"] = {"ticker": tickers[0] if tickers else None}
        ss_res = update_session_state_all(cfg, day=day, tickers=tickers)
        stages["session_state_result_n"] = len(ss_res) if hasattr(ss_res, "__len__") else "ok"

        # Step 4: levels
        stages["levels"] = {"ticker": tickers[0] if tickers else None}
        lv_res = compute_levels_all(cfg, day=day, tickers=tickers)
        stages["levels_result_n"] = len(lv_res) if hasattr(lv_res, "__len__") else "ok"

        # Step 5: intraday features
        if tickers:
            stages["intraday_features"] = {"mode": "one", "ticker": tickers[0]}
            feat_res = build_intraday_features_one(cfg, ticker=tickers[0], day=day)
        else:
            stages["intraday_features"] = {"mode": "all"}
            feat_res = build_intraday_features_all(cfg, day=day)
        stages["intraday_features_result"] = str(feat_res)

        # Step 6: today range
        if tickers:
            stages["today_range"] = {"mode": "one", "ticker": tickers[0]}
            tr_res = nowcast_today_range_one_ticker(cfg, ticker=tickers[0], day=day)
        else:
            stages["today_range"] = {"mode": "all"}
            tr_res = nowcast_today_range_all(cfg, day=day)
        stages["today_range_result"] = str(tr_res)

        # Step 7a: intraday insights
        if tickers:
            stages["intraday_insights"] = {"mode": "one", "ticker": tickers[0]}
            ins_res = intraday_insights_one(cfg, ticker=tickers[0], day=day)
        else:
            stages["intraday_insights"] = {"mode": "all"}
            ins_res = intraday_insights_all(cfg, day=day)
        stages["intraday_insights_result"] = str(ins_res)

        return IntradayRunResult(day=day, status="ok", error=None, stages=stages)

    except Exception as e:
        return IntradayRunResult(day=day, status="error", error=str(e), stages=stages)
