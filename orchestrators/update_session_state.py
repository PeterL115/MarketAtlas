from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

from ..core.config import AppConfig
from ..core.session_state import update_session_state


@dataclass(frozen=True)
class SessionStateRunResult:
    ticker: str
    day: str
    status: str
    last_update_ts: str | None


def update_session_state_all(cfg: AppConfig, day: str) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    for ticker in cfg.intraday_assets.keys():
        st = update_session_state(cfg, ticker=ticker, day=day)
        results[ticker] = SessionStateRunResult(
            ticker=ticker,
            day=day,
            status=str(st.get("status")),
            last_update_ts=st.get("last_update_ts"),
        )
    return {k: vars(v) for k, v in results.items()}
