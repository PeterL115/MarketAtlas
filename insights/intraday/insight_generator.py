from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

from MarketAtlas.core.config import AppConfig
from MarketAtlas.core.jsonio import read_json


# ============================================================
# i18n mode
# ============================================================

def _i18n_mode(cfg: AppConfig) -> str:
    """
    Output language mode for human-facing strings.
    Supported: "en" | "zh" | "both"
    Defaults to "both" if not configured.
    """
    try:
        mode = (cfg.raw.get("i18n", {}) or {}).get("mode", "both")
        mode = str(mode).strip().lower()
        if mode not in ("en", "zh", "both"):
            return "both"
        return mode
    except Exception:
        return "both"


def _fmt_num(x: Any, ndp: int = 3) -> str:
    try:
        v = float(x)
        if pd.isna(v):
            return "NA"
        return f"{v:.{ndp}f}"
    except Exception:
        return "NA"


# ============================================================
# i18n rationale messages (stable keys)
# ============================================================

_I18N_MESSAGES: Dict[str, Dict[str, str]] = {
    "R_LATE_SESSION": {
        "en": "Late session: bars_remaining={bars_remaining}",
        "zh": "临近收盘：剩余K线数={bars_remaining}",
    },
    "R_LOW_REMAINING_MOVE": {
        "en": "Low remaining movement: sigma_rem_atr={sigma_rem_atr}",
        "zh": "剩余波动偏低：sigma_rem_atr={sigma_rem_atr}",
    },
    "R_CHOPPY_TAPE": {
        "en": "Choppy / overlapping tape; trend signal weak",
        "zh": "行情偏震荡、重叠度高；趋势信号偏弱",
    },
    "R_OVERLAP_ER": {
        "en": "overlap={overlap} er={er}",
        "zh": "重叠度={overlap} ER={er}",
    },
    "R_BREAKOUT_UP_ELEVATED": {
        "en": "Breakout-up likelihood elevated ({p})",
        "zh": "向上突破概率偏高({p})",
    },
    "R_BREAKOUT_DOWN_ELEVATED": {
        "en": "Breakout-down likelihood elevated ({p})",
        "zh": "向下突破概率偏高({p})",
    },
    "R_HIGH_VOL_UNCLEAR_DIR": {
        "en": "High remaining volatility with unclear direction; defined-risk structures preferred",
        "zh": "剩余波动率高且方向不明确；更适合使用风险限定的结构(例如价差/铁鹰等)",
    },
    "R_STALL_HIGH_NEAR_LEVELS": {
        "en": "Stall likelihood high ({p}) near key levels",
        "zh": "在关键价位附近横盘/停滞概率较高({p})",
    },
}


def _i18n_item(key: str, **kwargs: Any) -> Dict[str, Any]:
    tpl = _I18N_MESSAGES.get(key, {"en": key, "zh": key})
    en_tpl = tpl.get("en", key)
    zh_tpl = tpl.get("zh", key)

    def safe_format(s: str) -> str:
        try:
            return s.format(**kwargs)
        except Exception:
            return s

    return {
        "key": key,
        "en": safe_format(en_tpl),
        "zh": safe_format(zh_tpl),
        "vars": kwargs or {},
    }


def _i18n_display_list(cfg: AppConfig, items: List[Dict[str, Any]]) -> List[str]:
    """
    Convert i18n items -> list[str] for backward-compatible display (CLI/log).
    """
    mode = _i18n_mode(cfg)
    if mode == "en":
        return [str(x.get("en", "")) for x in items]
    if mode == "zh":
        return [str(x.get("zh", "")) for x in items]
    # both
    out: List[str] = []
    for x in items:
        en = str(x.get("en", ""))
        zh = str(x.get("zh", ""))
        if en and zh and en != zh:
            out.append(f"{en} / {zh}")
        else:
            out.append(en or zh)
    return out


def _trade_bias_i18n(trade_bias: str) -> Dict[str, str]:
    tb = str(trade_bias or "neutral")
    mapping = {
        "neutral": {"en": "Neutral", "zh": "中性"},
        "range/mean_reversion": {"en": "Range / Mean Reversion", "zh": "区间/均值回归"},
        "bullish_breakout": {"en": "Bullish Breakout", "zh": "向上突破"},
        "bearish_breakout": {"en": "Bearish Breakout", "zh": "向下突破"},
    }
    return {"key": tb, **mapping.get(tb, {"en": tb, "zh": tb})}


# ============================================================
# Key-translation maps for "views"
# ============================================================

def _build_keymap(extra_range_bands: List[float]) -> Dict[str, Dict[str, Any]]:
    en: Dict[str, Any] = {
        "ticker": "Ticker",
        "day": "Date",
        "market_tz": "Market TZ",
        "asof_ts": "As-of Timestamp",
        "session": "Session",
        "last_price": "Last Price",

        "diagnostics": {
            "_": "Diagnostics",
            "atr": "Daily ATR",
            "rv_5m": "5m RV",
            "er": "Efficiency Ratio (ER)",
            "overlap": "Overlap",
            "sigma_pts": "Remaining Sigma (pts)",
            "sigma_rem_atr": "Remaining Sigma / ATR",
            "bars_remaining": "Bars Remaining (5m)",
            "dist_vwap_atr": "Distance to VWAP (ATR)",
            "pos_rth_range": "Position in RTH Range",
        },

        "levels_context": {
            "_": "Levels Context",
            "nearest_support": {
                "_": "Nearest Support",
                "id": "ID",
                "kind": "Kind",
                "zone_low": "Zone Low",
                "zone_high": "Zone High",
                "strength": "Strength",
                "dist_atr": "Distance (ATR)",
                "source": "Source",
            },
            "nearest_resistance": {
                "_": "Nearest Resistance",
                "id": "ID",
                "kind": "Kind",
                "zone_low": "Zone Low",
                "zone_high": "Zone High",
                "strength": "Strength",
                "dist_atr": "Distance (ATR)",
                "source": "Source",
            },
        },

        "range_context": {"_": "Range Context"},
        "likelihoods": {"_": "Likelihoods", "stall": "Stall", "breakout_up": "Breakout Up", "breakout_down": "Breakout Down"},

        "options_guidance": {
            "_": "Options Guidance",
            "trade_bias": "Trade Bias",
            "trade_bias_i18n": "Trade Bias (i18n)",
            "directional_premium_allowed": "Directional Premium Allowed",
            "prefer_spreads": "Prefer Spreads / Defined Risk",
            "mean_reversion_preferred": "Mean Reversion Preferred",
            "avoid_trading": "Avoid Trading",
            "rationale": "Rationale (display)",
            "rationale_i18n": "Rationale (i18n)",
        },
    }

    zh: Dict[str, Any] = {
        # -----------------------------
        # Top-level
        # -----------------------------
        "ticker": "股票代码",
        "day": "日期",
        "market_tz": "市场时区",
        "asof_ts": "数据截至时间",
        "session": "交易时段",
        "last_price": "最新价",

        # -----------------------------
        # Diagnostics
        # -----------------------------
        "diagnostics": {
            "_": "诊断指标",
            "atr": "ATR(日线,14)",
            "rv_5m": "5分钟实现波动(RV)",
            "er": "效率比(ER)",
            "overlap": "K线重叠度(Overlap)",
            "sigma_pts": "剩余波动σ(点)",
            "sigma_rem_atr": "剩余σ/ATR",
            "bars_remaining": "剩余5分钟K线(根)",
            "dist_vwap_atr": "距VWAP(ATR)",
            "pos_rth_range": "RTH区间位置",
        },

        # -----------------------------
        # Levels context
        # -----------------------------
        "levels_context": {
            "_": "关键价位上下文",
            "nearest_support": {
                "_": "最近支撑区",
                "id": "标识",
                "kind": "类型",
                "zone_low": "区间下沿",
                "zone_high": "区间上沿",
                "strength": "强度",
                "dist_atr": "距离(ATR)",
                "source": "来源",
            },
            "nearest_resistance": {
                "_": "最近阻力区",
                "id": "标识",
                "kind": "类型",
                "zone_low": "区间下沿",
                "zone_high": "区间上沿",
                "strength": "强度",
                "dist_atr": "距离(ATR)",
                "source": "来源",
            },
        },

        # -----------------------------
        # Range context
        # -----------------------------
        "range_context": {
            "_": "区间预测",
            # remaining_session_{k} / pos_in_remaining_{k} are appended later in the function
        },

        # -----------------------------
        # Likelihoods
        # -----------------------------
        "likelihoods": {
            "_": "概率评估",
            "stall": "横盘/受阻",
            "breakout_up": "上破概率",
            "breakout_down": "下破概率",
        },

        # -----------------------------
        # Options guidance
        # -----------------------------
        "options_guidance": {
            "_": "期权建议",
            "trade_bias": "交易倾向",
            "trade_bias_i18n": "交易倾向(中英)",
            "directional_premium_allowed": "允许方向性策略",
            "prefer_spreads": "优先价差/限定风险结构",
            "mean_reversion_preferred": "偏好均值回归",
            "avoid_trading": "建议观望",
            "rationale": "理由(展示)",
            "rationale_i18n": "理由(中英结构化)",
        },
    }


    probs = []
    for central in extra_range_bands:
        k = str(int(round(float(central) * 100)))
        probs.append(k)

    for k in ["45", "68", "95"]:
        if k not in probs:
            probs.append(k)

    for k in probs:
        en["range_context"][f"remaining_session_{k}"] = f"Remaining Session ({k}%)"
        en["range_context"][f"pos_in_remaining_{k}"] = f"Position in {k}% Band"
        zh["range_context"][f"remaining_session_{k}"] = f"剩余时段波动区间({k}%)"
        zh["range_context"][f"pos_in_remaining_{k}"] = f"当前价在{k}%区间位置(0=下沿,1=上沿)"


    return {"en": en, "zh": zh}


def _translate_keys(obj: Any, km_zh: Dict[str, Any], km_en: Dict[str, Any], mode: str) -> Any:
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            if k == "views":
                continue

            k_str = str(k)
            node_zh = km_zh.get(k_str)
            node_en = km_en.get(k_str)

            lbl_zh = node_zh.get("_") if isinstance(node_zh, dict) else (node_zh if isinstance(node_zh, str) else None)
            lbl_en = node_en.get("_") if isinstance(node_en, dict) else (node_en if isinstance(node_en, str) else None)

            if mode == "zh":
                new_key = lbl_zh or k_str
            else:
                if lbl_en or lbl_zh:
                    new_key = f"{(lbl_en or k_str)}/{(lbl_zh or k_str)}"
                else:
                    new_key = k_str

            child_km_zh = node_zh if isinstance(node_zh, dict) else {}
            child_km_en = node_en if isinstance(node_en, dict) else {}

            if "_" in child_km_zh:
                child_km_zh = {kk: vv for kk, vv in child_km_zh.items() if kk != "_"}
            if "_" in child_km_en:
                child_km_en = {kk: vv for kk, vv in child_km_en.items() if kk != "_"}

            out[new_key] = _translate_keys(v, child_km_zh, child_km_en, mode)
        return out

    if isinstance(obj, list):
        return [_translate_keys(x, km_zh, km_en, mode) for x in obj]

    return obj


# ============================================================
# Parameters / thresholds
# ============================================================

@dataclass
class InsightParams:
    eps: float = 1e-9

    near_level_atr: float = 0.25
    within_zone_bonus: float = 0.20

    trend_er_hi: float = 0.60
    trend_er_lo: float = 0.45
    chop_overlap_hi: float = 0.62

    sigma_rem_atr_low: float = 0.18
    sigma_rem_atr_hi: float = 0.45

    bars_remaining_avoid: int = 6

    allow_directional_if: float = 0.62
    avoid_if: float = 0.70

    extra_range_bands: List[float] = None


def _sf(x) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if pd.isna(v):
            return None
        return v
    except Exception:
        return None


def _clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _level_distance_to_zone(last: float, zlo: float, zhi: float) -> float:
    if zlo <= last <= zhi:
        return 0.0
    if last < zlo:
        return zlo - last
    return last - zhi


def _nearest_levels(levels: List[Dict[str, Any]], last: float) -> Dict[str, Any]:
    best_sup = None
    best_res = None
    best_sup_dist = float("inf")
    best_res_dist = float("inf")

    for lv in levels:
        kind = str(lv.get("kind", "")).lower()
        zlo = _sf(lv.get("zone_low"))
        zhi = _sf(lv.get("zone_high"))
        if zlo is None or zhi is None:
            continue

        dist = _level_distance_to_zone(last, zlo, zhi)

        is_support_like = kind in ("support", "pivot", "anchor")
        is_res_like = kind in ("resistance", "pivot", "anchor")

        if is_support_like:
            if zhi <= last or (zlo <= last <= zhi):
                if dist < best_sup_dist:
                    best_sup_dist = dist
                    best_sup = lv

        if is_res_like:
            if zlo >= last or (zlo <= last <= zhi):
                if dist < best_res_dist:
                    best_res_dist = dist
                    best_res = lv

    return {
        "nearest_support": best_sup,
        "nearest_support_dist": None if best_sup is None else float(best_sup_dist),
        "nearest_resistance": best_res,
        "nearest_resistance_dist": None if best_res is None else float(best_res_dist),
    }


def _zone_strength(lv: Optional[Dict[str, Any]]) -> float:
    if not lv:
        return 0.0
    s = _sf(lv.get("strength"))
    if s is None:
        return 0.0
    return _clamp01((s - 1.0) / 4.0)


def _level_name(lv: Optional[Dict[str, Any]]) -> Optional[str]:
    if not lv:
        return None
    return str(lv.get("id") or lv.get("source") or "level")


def _band_position(last: float, lo: float, hi: float) -> Optional[float]:
    w = hi - lo
    if w <= 0:
        return None
    return float((last - lo) / w)


def _norm_ppf(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        raise ValueError("p must be in (0,1)")

    a = [
        -3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
        1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00
    ]
    b = [
        -5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
        6.680131188771972e01, -1.328068155288572e01
    ]
    c = [
        -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
        -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00
    ]
    d = [
        7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
        3.754408661907416e00
    ]

    plow = 0.02425
    phigh = 1 - plow

    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                 ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)

    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def _z_for_central_prob(central: float) -> float:
    return float(_norm_ppf(0.5 + 0.5 * float(central)))


def _make_band(last: float, sigma_pts: float, central: float) -> Dict[str, float]:
    z = _z_for_central_prob(central)
    lo = float(last - z * sigma_pts)
    hi = float(last + z * sigma_pts)
    return {"low": lo, "high": hi, "width": float(hi - lo)}


def generate_intraday_insight_one(
    cfg: AppConfig,
    ticker: str,
    day: str,
    params: Optional[InsightParams] = None
) -> Dict[str, Any]:
    p = params or InsightParams()
    if p.extra_range_bands is None:
        # ✅ Only 45/68/95
        p.extra_range_bands = [0.45, 0.68, 0.95]

    feat = read_json(cfg.path("features_intraday_latest_dir") / f"{ticker}.json") or {}
    tr = read_json(cfg.path("today_range_dir") / f"{ticker}.json") or {}
    lv = read_json(cfg.path("levels_dir") / f"{ticker}.json") or {}

    if feat.get("ticker") != ticker:
        raise ValueError(f"Missing/invalid Step 4 intraday_latest for {ticker}.")
    if tr.get("ticker") != ticker:
        raise ValueError(f"Missing Step 5 today_range for {ticker}.")
    if lv.get("ticker") != ticker:
        raise ValueError(f"Missing Step 3 levels for {ticker}.")

    asof_ts = str(feat.get("asof_ts"))
    last = float(feat["last_price"])
    session = str(feat.get("session") or tr.get("session") or "UNK")

    atr = _sf((lv.get("atr") or {}).get("value"))
    if atr is None or atr <= 0:
        raise ValueError(f"ATR missing/invalid for {ticker}.")

    f = feat.get("features") or {}
    rv = _sf(f.get("rv"))
    er = _sf(f.get("er"))
    overlap = _sf(f.get("overlap"))
    dist_vwap_atr = _sf(f.get("dist_vwap_atr"))
    pos_rth_range = _sf(f.get("pos_rth_range"))

    rem = tr.get("remaining_session") or {}
    band68 = rem.get("band_68") or {}
    band95 = rem.get("band_95") or {}
    rem_lo = _sf(band68.get("low"))
    rem_hi = _sf(band68.get("high"))
    rem95_lo = _sf(band95.get("low"))
    rem95_hi = _sf(band95.get("high"))

    bars_remaining = int((tr.get("inputs") or {}).get("bars_remaining") or 0)
    sigma_pts = _sf(rem.get("sigma_pts"))

    sigma_rem_atr = None
    if sigma_pts is not None and atr > 0:
        sigma_rem_atr = float(sigma_pts / atr)

    levels = list(lv.get("levels") or [])
    near = _nearest_levels(levels, last)
    sup = near["nearest_support"]
    res = near["nearest_resistance"]

    sup_dist_pts = near["nearest_support_dist"]
    res_dist_pts = near["nearest_resistance_dist"]

    sup_dist_atr = None if sup_dist_pts is None else float(sup_dist_pts / atr)
    res_dist_atr = None if res_dist_pts is None else float(res_dist_pts / atr)

    sup_str = _zone_strength(sup)
    res_str = _zone_strength(res)

    pos_rem_68 = None
    if rem_lo is not None and rem_hi is not None:
        pos_rem_68 = _band_position(last, rem_lo, rem_hi)

    trend = 0.0 if er is None else _clamp01((er - p.trend_er_lo) / max(p.eps, (p.trend_er_hi - p.trend_er_lo)))
    chop = 0.0 if overlap is None else _clamp01((overlap - 0.45) / max(p.eps, (p.chop_overlap_hi - 0.45)))

    vol = 0.0
    if sigma_rem_atr is not None:
        vol = _clamp01((sigma_rem_atr - p.sigma_rem_atr_low) / max(p.eps, (p.sigma_rem_atr_hi - p.sigma_rem_atr_low)))

    def near_score(dist_atr: Optional[float], strength01: float) -> float:
        if dist_atr is None:
            return 0.0
        x = max(0.0, 1.0 - (dist_atr / max(p.eps, p.near_level_atr)))
        return _clamp01(x * (0.5 + 0.5 * strength01))

    near_sup = near_score(sup_dist_atr, sup_str)
    near_res = near_score(res_dist_atr, res_str)
    near_any = max(near_sup, near_res)

    inside_zone = 0.0
    if (sup_dist_atr is not None and sup_dist_atr <= 0.0) or (res_dist_atr is not None and res_dist_atr <= 0.0):
        inside_zone = p.within_zone_bonus

    stall_raw = (1.10 * chop + 0.70 * near_any + 0.50 * (1.0 - vol) + inside_zone)
    stall = _clamp01(_sigmoid(stall_raw - 1.0))

    up_raw = 1.10 * trend + 0.80 * vol + 0.65 * near_res - 0.75 * chop
    dn_raw = 1.10 * trend + 0.80 * vol + 0.65 * near_sup - 0.75 * chop

    if pos_rem_68 is not None:
        up_raw += 0.35 * (pos_rem_68 - 0.5)
        dn_raw += 0.35 * (0.5 - pos_rem_68)

    breakout_up = _clamp01(_sigmoid(up_raw))
    breakout_down = _clamp01(_sigmoid(dn_raw))

    breakout_up = float(breakout_up * (1.0 - 0.55 * stall))
    breakout_down = float(breakout_down * (1.0 - 0.55 * stall))

    rationale_i18n: List[Dict[str, Any]] = []
    guidance: Dict[str, Any] = {
        "trade_bias": "neutral",
        "trade_bias_i18n": _trade_bias_i18n("neutral"),
        "directional_premium_allowed": False,
        "prefer_spreads": False,
        "mean_reversion_preferred": False,
        "avoid_trading": False,
        "rationale": [],
        "rationale_i18n": rationale_i18n,
    }

    if bars_remaining > 0 and bars_remaining <= p.bars_remaining_avoid:
        guidance["avoid_trading"] = True
        rationale_i18n.append(_i18n_item("R_LATE_SESSION", bars_remaining=bars_remaining))

    if sigma_rem_atr is not None and sigma_rem_atr < p.sigma_rem_atr_low:
        guidance["avoid_trading"] = True
        rationale_i18n.append(_i18n_item("R_LOW_REMAINING_MOVE", sigma_rem_atr=_fmt_num(sigma_rem_atr, 3)))

    if not guidance["avoid_trading"]:
        if (overlap is not None and overlap >= p.chop_overlap_hi) and (er is None or er < p.trend_er_lo):
            guidance["mean_reversion_preferred"] = True
            guidance["prefer_spreads"] = True
            guidance["trade_bias"] = "range/mean_reversion"
            guidance["trade_bias_i18n"] = _trade_bias_i18n("range/mean_reversion")
            rationale_i18n.append(_i18n_item("R_CHOPPY_TAPE"))
            rationale_i18n.append(_i18n_item("R_OVERLAP_ER", overlap=_fmt_num(overlap, 3), er=("NA" if er is None else _fmt_num(er, 3))))

    if not guidance["avoid_trading"] and not guidance["mean_reversion_preferred"]:
        if breakout_up >= p.allow_directional_if and breakout_up > breakout_down:
            guidance["directional_premium_allowed"] = True
            guidance["trade_bias"] = "bullish_breakout"
            guidance["trade_bias_i18n"] = _trade_bias_i18n("bullish_breakout")
            rationale_i18n.append(_i18n_item("R_BREAKOUT_UP_ELEVATED", p=_fmt_num(breakout_up, 2)))
        elif breakout_down >= p.allow_directional_if and breakout_down > breakout_up:
            guidance["directional_premium_allowed"] = True
            guidance["trade_bias"] = "bearish_breakout"
            guidance["trade_bias_i18n"] = _trade_bias_i18n("bearish_breakout")
            rationale_i18n.append(_i18n_item("R_BREAKOUT_DOWN_ELEVATED", p=_fmt_num(breakout_down, 2)))

    if not guidance["avoid_trading"]:
        if sigma_rem_atr is not None and sigma_rem_atr >= p.sigma_rem_atr_hi and not guidance["directional_premium_allowed"]:
            guidance["prefer_spreads"] = True
            rationale_i18n.append(_i18n_item("R_HIGH_VOL_UNCLEAR_DIR"))

    if stall >= p.avoid_if:
        guidance["directional_premium_allowed"] = False
        if not guidance["mean_reversion_preferred"]:
            guidance["avoid_trading"] = True
        rationale_i18n.append(_i18n_item("R_STALL_HIGH_NEAR_LEVELS", p=_fmt_num(stall, 2)))

    guidance["rationale"] = _i18n_display_list(cfg, rationale_i18n)

    # extra bands
    extra_bands: Dict[str, Any] = {}
    extra_positions: Dict[str, Any] = {}
    if sigma_pts is not None and sigma_pts > 0:
        for central in p.extra_range_bands:
            k = str(int(round(float(central) * 100)))  # "45","68","95"
            bd = _make_band(last, sigma_pts, float(central))
            extra_bands[k] = {"low": bd["low"], "high": bd["high"]}
            extra_positions[k] = _band_position(last, bd["low"], bd["high"])

    # -----------------------------
    # range_context ORDER: 45 / 68 / 95
    # -----------------------------
    range_order = ["45", "68", "95"]
    extras = sorted([k for k in extra_bands.keys() if k not in range_order], key=lambda x: float(x))
    final_order = range_order + extras

    range_context: Dict[str, Any] = {}

    # bands
    for k in final_order:
        if k == "68":
            if rem_lo is not None and rem_hi is not None:
                range_context["remaining_session_68"] = {"low": rem_lo, "high": rem_hi}
            elif "68" in extra_bands:
                range_context["remaining_session_68"] = extra_bands["68"]
        elif k == "95":
            if rem95_lo is not None and rem95_hi is not None:
                range_context["remaining_session_95"] = {"low": rem95_lo, "high": rem95_hi}
            elif "95" in extra_bands:
                range_context["remaining_session_95"] = extra_bands["95"]
        else:
            if k in extra_bands:
                range_context[f"remaining_session_{k}"] = extra_bands[k]

    # positions in same order
    for k in final_order:
        band = range_context.get(f"remaining_session_{k}")
        if not isinstance(band, dict):
            continue
        lo = _sf(band.get("low"))
        hi = _sf(band.get("high"))
        if lo is None or hi is None:
            continue

        if k == "68" and pos_rem_68 is not None:
            pos = pos_rem_68
        else:
            pos = extra_positions.get(k)
            if pos is None:
                pos = _band_position(last, lo, hi)

        range_context[f"pos_in_remaining_{k}"] = pos

    # canonical output
    base_out: Dict[str, Any] = {
        "ticker": ticker,
        "day": day,
        "market_tz": cfg.market_tz,
        "asof_ts": asof_ts,
        "session": session,
        "last_price": last,

        "diagnostics": {
            "atr": atr,
            "rv_5m": rv,
            "er": er,
            "overlap": overlap,
            "sigma_pts": sigma_pts,
            "sigma_rem_atr": sigma_rem_atr,
            "bars_remaining": bars_remaining,
            "dist_vwap_atr": dist_vwap_atr,
            "pos_rth_range": pos_rth_range,
        },

        "levels_context": {
            "nearest_support": {
                "id": _level_name(sup),
                "kind": None if not sup else sup.get("kind"),
                "zone_low": None if not sup else sup.get("zone_low"),
                "zone_high": None if not sup else sup.get("zone_high"),
                "strength": None if not sup else sup.get("strength"),
                "dist_atr": sup_dist_atr,
                "source": None if not sup else sup.get("source"),
            },
            "nearest_resistance": {
                "id": _level_name(res),
                "kind": None if not res else res.get("kind"),
                "zone_low": None if not res else res.get("zone_low"),
                "zone_high": None if not res else res.get("zone_high"),
                "strength": None if not res else res.get("strength"),
                "dist_atr": res_dist_atr,
                "source": None if not res else res.get("source"),
            },
        },

        "range_context": range_context,

        "likelihoods": {
            "stall": float(stall),
            "breakout_up": float(breakout_up),
            "breakout_down": float(breakout_down),
        },

        "options_guidance": guidance,
    }

    # views
    km = _build_keymap(p.extra_range_bands)
    km_en = km["en"]
    km_zh = km["zh"]

    mode = _i18n_mode(cfg)
    views: Dict[str, Any] = {}

    if mode in ("zh", "both"):
        views["zh"] = _translate_keys(base_out, km_zh, km_en, mode="zh")
    if mode == "both":
        views["both"] = _translate_keys(base_out, km_zh, km_en, mode="both")

    out = dict(base_out)
    if views:
        out["views"] = views

    return out
