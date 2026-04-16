from __future__ import annotations

import random
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union

import pandas as pd
import yfinance as yf

# yfinance 的限流异常在不同版本里位置不完全一致：做一个兼容导入
try:
    from yfinance.exceptions import YFRateLimitError  # type: ignore
except Exception:  # pragma: no cover
    class YFRateLimitError(Exception):
        pass


__all__ = [
    "YFPolicy",
    "YFinanceProvider",
    "get_yf_provider",
    "get_iv_snapshot_yahoo",
]


@dataclass
class YFPolicy:
    # 重试次数
    max_attempts: int = 6

    # 指数退避：sleep = base_backoff_s * 2**attempt + rand(0, jitter_s)
    base_backoff_s: float = 1.25
    jitter_s: float = 0.75
    max_backoff_s: float = 20.0

    # 每个 provider 调用之间的“最低间隔”（避免瞬间连发）
    min_interval_s: float = 0.2

    # 单 ticker 跑完后，额外 sleep（在 orchestrator 循环里也会做，这里是 provider 级兜底）
    post_call_sleep_s: float = 0.0


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if pd.isna(v):
            return None
        return v
    except Exception:
        return None


def _pick_expiry(options: List[str], target_dte: int = 30, *, market_tz: str = "America/New_York") -> Optional[str]:
    """
    Pick expiry closest to target_dte (days-to-expiry).
    If parsing fails, fallback to first expiry.
    """
    if not options:
        return None

    try:
        # 用市场时区的“今天”来算 DTE，避免 UTC 跨日偏差
        today = pd.Timestamp.now(tz=market_tz).normalize()
        best: Optional[str] = None
        best_abs: Optional[int] = None

        for exp in options:
            d = pd.Timestamp(exp)
            # exp 是日期字符串（YYYY-MM-DD），默认 tz-naive；这里按 market_tz 的日期处理
            d = d.tz_localize(market_tz, nonexistent="shift_forward", ambiguous="NaT") if d.tzinfo is None else d.tz_convert(market_tz)
            dte = (d.normalize() - today).days
            if dte < 0:
                continue
            a = abs(dte - int(target_dte))
            if best_abs is None or a < best_abs:
                best_abs = a
                best = exp

        return best or options[0]
    except Exception:
        return options[0]


class YFinanceProvider:
    """
    Yahoo Finance (yfinance) provider with:
      - retry + exponential backoff + jitter
      - simple throttle (min interval between calls)

    Plus:
      - IV snapshot from options chain (ATM, nearest target DTE)
    """

    def __init__(self, market_tz: str = "America/New_York", policy: Optional[YFPolicy] = None) -> None:
        self.market_tz = market_tz
        self.policy = policy or YFPolicy()
        self._last_call_ts: float = 0.0

    # -----------------------------
    # Internal helpers
    # -----------------------------
    def _throttle(self) -> None:
        p = self.policy
        now = time.time()
        elapsed = now - self._last_call_ts
        if elapsed < p.min_interval_s:
            time.sleep(max(0.0, p.min_interval_s - elapsed))
        self._last_call_ts = time.time()

    def _sleep_backoff(self, attempt: int) -> None:
        p = self.policy
        raw = p.base_backoff_s * (2 ** attempt) + random.random() * p.jitter_s
        time.sleep(min(raw, p.max_backoff_s))

    def _is_rate_limit(self, e: Exception) -> bool:
        # 兼容不同版本/不同报错字符串
        s = str(e).lower()
        if isinstance(e, YFRateLimitError):
            return True
        return ("rate limit" in s) or ("too many requests" in s) or ("429" in s)

    def _coerce_user_dt_for_yf(self, x: Optional[Union[str, datetime]]) -> Optional[Union[datetime, str]]:
        """
        把各种 start/end 输入（尤其是 'YYYY-MM-DDTHH:MM:SS-05:00' 这种）
        转换成 yfinance 更稳定接受的 python datetime（tz-naive）。

        yfinance 的某些版本对字符串 start/end 会走 '%Y-%m-%d' 的 strptime，
        遇到带 'T...' 就会 ValueError: unconverted data remains。
        传 datetime 对象可以绕开这条分支。
        """
        if x is None:
            return None

        if isinstance(x, datetime):
            dt0 = x
        else:
            try:
                ts = pd.to_datetime(x, errors="raise")
                dt0 = ts.to_pydatetime()
            except Exception:
                # 解析不了就原样返回（让 yfinance 自己处理）
                return x

        # 如果 tz-aware：先转到 market_tz，再去掉 tzinfo（yfinance 最稳）
        if dt0.tzinfo is not None:
            try:
                dt2 = pd.Timestamp(dt0).tz_convert(self.market_tz).to_pydatetime()
                dt0 = dt2.replace(tzinfo=None)
            except Exception:
                dt0 = dt0.replace(tzinfo=None)

        return dt0

    def _call_with_retry(self, fn: Callable[[], Any]) -> Any:
        """
        对非 download 的 yfinance 调用（如 Ticker.options / option_chain / fast_info）也做同样的 retry/backoff。
        """
        p = self.policy
        last_err: Optional[Exception] = None

        for attempt in range(p.max_attempts):
            try:
                self._throttle()
                out = fn()
                if p.post_call_sleep_s > 0:
                    time.sleep(p.post_call_sleep_s)
                return out
            except Exception as e:
                last_err = e
                if self._is_rate_limit(e) and attempt < p.max_attempts - 1:
                    self._sleep_backoff(attempt)
                    continue
                raise

        if last_err:
            raise last_err
        return None

    def _download_with_retry(
        self,
        symbol: str,
        *,
        interval: str,
        start: Optional[Union[str, datetime]] = None,
        end: Optional[Union[str, datetime]] = None,
        auto_adjust: bool = True,
        actions: bool = False,
        prepost: bool = True,
    ) -> pd.DataFrame:
        p = self.policy

        # 关键修复：把带 T / 带 offset 的字符串转成 datetime（绕开 yfinance 的 '%Y-%m-%d' 解析）
        start_arg = self._coerce_user_dt_for_yf(start)
        end_arg = self._coerce_user_dt_for_yf(end)

        last_err: Optional[Exception] = None
        for attempt in range(p.max_attempts):
            try:
                self._throttle()
                df = yf.download(
                    tickers=symbol,
                    start=start_arg,
                    end=end_arg,
                    interval=interval,
                    auto_adjust=auto_adjust,
                    actions=actions,
                    prepost=prepost,
                    progress=False,
                    threads=False,  # 强烈建议关掉 threads，减少并发触发限流
                )
                if p.post_call_sleep_s > 0:
                    time.sleep(p.post_call_sleep_s)
                return df

            except Exception as e:
                last_err = e
                if self._is_rate_limit(e) and attempt < p.max_attempts - 1:
                    self._sleep_backoff(attempt)
                    continue
                raise

        if last_err:
            raise last_err
        return pd.DataFrame()

    # -----------------------------
    # Public APIs: prices
    # -----------------------------
    def fetch_daily(self, symbol: str, start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
        raw = self._download_with_retry(
            symbol,
            interval="1d",
            start=start,
            end=end,
            auto_adjust=True,
            actions=False,
            prepost=False,
        )
        return self._normalize_ohlc(raw, tz=self.market_tz)

    def fetch_intraday_5m(
        self,
        symbol: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        include_extended_hours: bool = True,
    ) -> pd.DataFrame:
        raw = self._download_with_retry(
            symbol,
            interval="5m",
            start=start,
            end=end,
            auto_adjust=True,
            actions=False,
            prepost=bool(include_extended_hours),
        )
        return self._normalize_ohlc(raw, tz=self.market_tz)

    # -----------------------------
    # Public APIs: options IV
    # -----------------------------
    def fetch_iv_snapshot(self, symbol: str, *, target_dte: int = 30) -> Optional[float]:
        """
        Best-effort implied vol snapshot (0..1) from Yahoo options chain.

        Method:
          - pick expiry closest to target_dte
          - pick ATM call + ATM put (closest strike to spot)
          - return average(call_iv, put_iv)

        Notes:
          - Some symbols (esp. indices like ^GSPC) may have no options -> None
          - This is a snapshot, not "historical IV"
        """
        try:
            tk = yf.Ticker(symbol)

            exps = self._call_with_retry(lambda: list(getattr(tk, "options", None) or []))
            if not exps:
                return None

            exp = _pick_expiry(exps, target_dte=int(target_dte), market_tz=self.market_tz)
            if not exp:
                return None

            oc = self._call_with_retry(lambda: tk.option_chain(exp))
            calls = getattr(oc, "calls", None)
            puts = getattr(oc, "puts", None)
            if calls is None or puts is None or calls.empty or puts.empty:
                return None

            # spot price
            spot = None
            try:
                fi = self._call_with_retry(lambda: getattr(tk, "fast_info", None) or {})
                spot = _safe_float(fi.get("last_price"))
            except Exception:
                spot = None

            if spot is None:
                try:
                    info = self._call_with_retry(lambda: getattr(tk, "info", None) or {})
                    spot = _safe_float(info.get("regularMarketPrice"))
                except Exception:
                    spot = None

            if spot is None:
                return None

            calls2 = calls.copy()
            puts2 = puts.copy()

            # Some chains may have NaNs/strings in strike/iv columns
            calls2["strike"] = pd.to_numeric(calls2.get("strike"), errors="coerce")
            puts2["strike"] = pd.to_numeric(puts2.get("strike"), errors="coerce")
            if "impliedVolatility" in calls2.columns:
                calls2["impliedVolatility"] = pd.to_numeric(calls2["impliedVolatility"], errors="coerce")
            if "impliedVolatility" in puts2.columns:
                puts2["impliedVolatility"] = pd.to_numeric(puts2["impliedVolatility"], errors="coerce")

            calls2 = calls2.dropna(subset=["strike"]).copy()
            puts2 = puts2.dropna(subset=["strike"]).copy()
            if calls2.empty or puts2.empty:
                return None

            calls2["d"] = (calls2["strike"] - float(spot)).abs()
            puts2["d"] = (puts2["strike"] - float(spot)).abs()

            c = calls2.sort_values("d").head(1)
            p = puts2.sort_values("d").head(1)

            c_iv = _safe_float(c.iloc[0].get("impliedVolatility")) if not c.empty else None
            p_iv = _safe_float(p.iloc[0].get("impliedVolatility")) if not p.empty else None

            if c_iv is None and p_iv is None:
                return None
            if c_iv is None:
                return p_iv
            if p_iv is None:
                return c_iv

            return float((c_iv + p_iv) / 2.0)

        except Exception:
            return None

    # -----------------------------
    # Normalization
    # -----------------------------
    def _normalize_ohlc(self, df: pd.DataFrame, tz: str) -> pd.DataFrame:
        """
        Normalize yfinance output into canonical schema:
          ts, open, high, low, close, volume
        """
        if df is None or df.empty:
            return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])

        out = df.copy()

        # yfinance download sometimes returns multiindex columns when tickers list is used; here symbol is single, but safe-guard.
        if isinstance(out.columns, pd.MultiIndex):
            out.columns = [c[0] for c in out.columns]

        # index -> ts
        out = out.reset_index()

        # yfinance uses "Date" or "Datetime" depending on interval/version
        if "Datetime" in out.columns:
            out = out.rename(columns={"Datetime": "ts"})
        elif "Date" in out.columns:
            out = out.rename(columns={"Date": "ts"})
        else:
            out = out.rename(columns={out.columns[0]: "ts"})

        colmap = {}
        for k in ["Open", "High", "Low", "Close", "Volume"]:
            if k in out.columns:
                colmap[k] = k.lower()
        out = out.rename(columns=colmap)

        # Ensure required columns exist
        for c in ["open", "high", "low", "close"]:
            if c not in out.columns:
                out[c] = pd.NA
        if "volume" not in out.columns:
            out["volume"] = pd.NA

        # Normalize ts timezone
        ts = pd.to_datetime(out["ts"], errors="coerce")
        # yfinance often returns tz-aware for intraday, tz-naive for daily
        if getattr(ts.dt, "tz", None) is None:
            ts = ts.dt.tz_localize(tz)
        else:
            ts = ts.dt.tz_convert(tz)

        out["ts"] = ts
        out = out.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)

        # Keep only canonical columns
        out = out[["ts", "open", "high", "low", "close", "volume"]].copy()
        return out


# ============================================================
# Module-level singleton(s) + wrapper exports
# ============================================================

# 复用 provider：保持 throttle / retry 的状态，避免每次 new 导致短时间连发
_PROVIDERS: Dict[str, YFinanceProvider] = {}


def get_yf_provider(*, market_tz: str = "America/New_York", policy: Optional[YFPolicy] = None) -> YFinanceProvider:
    """
    Get (and cache) a YFinanceProvider by market_tz.

    If you pass a policy, it will create/replace the provider for that tz.
    (So use policy only when you know what you're doing.)
    """
    global _PROVIDERS
    if policy is not None:
        p = YFinanceProvider(market_tz=market_tz, policy=policy)
        _PROVIDERS[market_tz] = p
        return p

    p0 = _PROVIDERS.get(market_tz)
    if p0 is None:
        p0 = YFinanceProvider(market_tz=market_tz)
        _PROVIDERS[market_tz] = p0
    return p0


def get_iv_snapshot_yahoo(symbol: str, target_dte: int = 30, *, market_tz: str = "America/New_York") -> Optional[float]:
    """
    ✅ Wrapper function expected by orchestrators:
        from MarketAtlas.core.providers.yfinance_provider import get_iv_snapshot_yahoo

    Returns IV in 0..1 (or None).
    """
    p = get_yf_provider(market_tz=market_tz)
    return p.fetch_iv_snapshot(symbol, target_dte=int(target_dte))