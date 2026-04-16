from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

from ..core.config import AppConfig
from ..core.session_state import update_session_state


@dataclass(frozen=True)
class SessionStateRunResult:
    ticker: str
    day: str
    status: str
    last_update_ts: str | None


def _resolve_tickers(cfg: AppConfig, tickers: Optional[Iterable[str]] = None) -> list[str]:
    if tickers is None:
        return list(cfg.intraday_assets.keys())
    # 过滤掉不在配置里的 ticker，避免 KeyError / 跑到不存在的标的
    want = [str(t).strip() for t in tickers if str(t).strip()]
    have = set(cfg.intraday_assets.keys())
    return [t for t in want if t in have]


def update_session_state_all(cfg: AppConfig, day: str, tickers: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    """
    Build/update session state JSON from intraday 5m bars.

    - tickers=None  => all configured intraday assets
    - tickers=[...] => only those tickers (must exist in cfg.intraday_assets)
    """
    results: Dict[str, Any] = {}
    universe = _resolve_tickers(cfg, tickers=tickers)

    for ticker in universe:
        st = update_session_state(cfg, ticker=ticker, day=day)
        results[ticker] = SessionStateRunResult(
            ticker=ticker,
            day=day,
            status=str(st.get("status")),
            last_update_ts=st.get("last_update_ts"),
        )

    return {k: vars(v) for k, v in results.items()}
