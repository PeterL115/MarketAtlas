from __future__ import annotations

from typing import Any, Dict, Optional
import re


# ---------------------------------------------------------------------
# Display labels for EOD output views
# ---------------------------------------------------------------------
EN = {
    # common top-level
    "ticker": "Ticker",
    "day": "Plan Day",
    "asof_day": "As-of Trading Day",
    "predicted_day": "Predicted Session Date",
    "market_tz": "Market TZ",
    "asof_ts": "As-of Timestamp",

    # EOD plan
    "today_close": "Close (As-of Day)",

    "regime_context": "Regime Context",
    "asof_label_day": "Label As-of Day",
    "asof_regime": "Regime",

    "tomorrow_range": "Tomorrow Range",
    "band_45": "Band (45%)",
    "band_68": "Band (68%)",
    "band_95": "Band (95%)",
    "low": "Low",
    "high": "High",

    "tomorrow_key_levels": "Key Levels (Next Session)",
    "supports": "Support Zones",
    "resistances": "Resistance Zones",
    "id": "ID",
    "kind": "Type",
    "zone_low": "Zone Low",
    "zone_high": "Zone High",
    "strength": "Strength",
    "source": "Source",
    "dist_pts": "Distance (pts)",

    "risk_notes": "Risk Notes",

    "sources": "Sources",
    "daily_latest": "Daily Latest",
    "regime_labels_latest": "Regime Labels Latest",
    "levels_latest": "Levels Latest",
}

ZH = {
    # common top-level (market wording)
    "ticker": "股票代码",
    "day": "计划生成日（EOD）",
    "asof_day": "数据对应交易日",
    "predicted_day": "预测交易日",
    "market_tz": "市场时区",
    "asof_ts": "数据时间戳（as-of）",

    # EOD plan
    "today_close": "基准收盘价（as-of）",

    "regime_context": "市场状态（Regime）",
    "asof_label_day": "状态标签对应交易日",
    "asof_regime": "市场状态",

    "tomorrow_range": "次日价格区间（概率带）",
    "band_45": "概率带（45%）",
    "band_68": "概率带（68%）",
    "band_95": "概率带（95%）",
    "low": "下界",
    "high": "上界",

    "tomorrow_key_levels": "次日关键价位区（支撑/阻力带）",
    "supports": "支撑带（需求区）",
    "resistances": "阻力带（供给区）",
    "id": "标识",
    "kind": "类型",
    "zone_low": "区间下沿",
    "zone_high": "区间上沿",
    "strength": "强度",
    "source": "来源",
    "dist_pts": "距离（点）",

    "risk_notes": "风险提示",

    "sources": "数据来源",
    "daily_latest": "日线特征（latest）",
    "regime_labels_latest": "状态标签（latest）",
    "levels_latest": "关键价位（latest）",
}


# Regime VALUE translations (views only)
REGIME_ZH = {
    "trend_up": "上行趋势",
    "trend_down": "下行趋势",
    "range": "区间震荡",
    "high_vol": "高波动（方向不明）",
    "transition": "状态切换期",
    "unknown": "未知",
}

# Level kind VALUE translations (views only)
KIND_ZH = {
    "pivot": "枢轴位",
    "support": "支撑",
    "resistance": "阻力",
    "anchor": "锚点",
}

# Level source VALUE translations (views only)
SOURCE_ZH = {
    "prior_day": "前一交易日",
    "prior_week": "上一周",
    "swing_pivot": "摆动枢轴",
    "opening_range": "开盘区间（OR）",
    "vwap": "成交量加权均价（VWAP）",
    "session_extreme": "时段极值",
    "premarket_context": "盘前上下文",
}


def _norm(s: str) -> str:
    return str(s or "").strip()


def _norm_lc(s: str) -> str:
    return _norm(s).lower()


def _label_en(key: str) -> Optional[str]:
    return EN.get(key)


def _label_zh(key: str) -> Optional[str]:
    return ZH.get(key)


def _label_both(key: str) -> str:
    en = _label_en(key) or key
    zh = _label_zh(key)
    return f"{en}/{zh}" if zh else en


def _regime_to_zh(v: str) -> Optional[str]:
    return REGIME_ZH.get(_norm_lc(v))


def _kind_to_zh(v: str) -> Optional[str]:
    return KIND_ZH.get(_norm_lc(v))


def _source_to_zh(v: str) -> Optional[str]:
    return SOURCE_ZH.get(_norm_lc(v))


def _translate_risk_note(note: str, *, mode: str) -> str:
    """
    Translate the specific templates produced by MarketAtlas.orchestrators.eod_run._risk_notes().
    Deterministic, no external calls.
    """
    if not isinstance(note, str):
        return str(note)

    n = note.strip()

    # Regime=xxx: consider wider error bars and defined-risk structures.
    m = re.match(r"^Regime=(\w+):\s*(.*)$", n)
    if m:
        reg = m.group(1)
        zh_reg = _regime_to_zh(reg) or reg
        # Make this desk-friendly Chinese
        zh_rest = "建议扩大误差带，优先采用限定风险结构（如价差、铁鹰等）。"
        zh = f"市场状态={zh_reg}：{zh_rest}"
        if mode == "zh":
            return zh
        if mode == "both":
            return f"{n} / {zh}"
        return n

    # Large gap vs ATR (gap_atr=...): open may be discontinuous.
    m = re.match(r"^Large gap vs ATR\s*\(gap_atr=([-\d\.]+)\):\s*(.*)$", n)
    if m:
        gap = m.group(1)
        zh = f"跳空幅度相对ATR偏大（gap_atr={gap}）：开盘可能出现价格不连续跳变。"
        if mode == "zh":
            return zh
        if mode == "both":
            return f"{n} / {zh}"
        return n

    # Elevated realized vol proxy (rv=...).
    m = re.match(r"^Elevated realized vol proxy\s*\(rv=([-\d\.]+)\)\.?\s*$", n)
    if m:
        rv = m.group(1)
        zh = f"短期实现波动偏高（rv={rv}）。"
        if mode == "zh":
            return zh
        if mode == "both":
            return f"{n} / {zh}"
        return n

    # ATR invalid (should not happen).
    if n == "ATR invalid (should not happen).":
        zh = "ATR异常（理论上不应发生）。"
        if mode == "zh":
            return zh
        if mode == "both":
            return f"{n} / {zh}"
        return n

    return n


def _transform(obj: Any, key_label_fn, *, mode: str, parent_key: Optional[str] = None) -> Any:
    """
    Recursively transform dict keys using key_label_fn(key)->new_key.
    Also translates selected VALUES in views only:
      - regime/asof_regime
      - kind/source
      - risk_notes strings
    """
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            ks = str(k)
            new_k = key_label_fn(ks)
            out[new_k] = _transform(v, key_label_fn, mode=mode, parent_key=ks)
        return out

    if isinstance(obj, list):
        if parent_key == "risk_notes":
            return [_translate_risk_note(x, mode=mode) for x in obj]
        return [_transform(x, key_label_fn, mode=mode, parent_key=parent_key) for x in obj]

    # value-level translation for views only
    if isinstance(obj, str):
        if parent_key in ("regime", "asof_regime"):
            zh_val = _regime_to_zh(obj)
            if zh_val:
                if mode == "zh":
                    return zh_val
                if mode == "both":
                    return f"{obj}/{zh_val}"
            return obj

        if parent_key == "kind":
            zh_val = _kind_to_zh(obj)
            if zh_val:
                if mode == "zh":
                    return zh_val
                if mode == "both":
                    return f"{obj}/{zh_val}"
            return obj

        if parent_key == "source":
            zh_val = _source_to_zh(obj)
            if zh_val:
                if mode == "zh":
                    return zh_val
                if mode == "both":
                    return f"{obj}/{zh_val}"
            return obj

    return obj


def attach_eod_views(payload: Dict[str, Any], mode: str) -> Dict[str, Any]:
    """
    Add payload["views"] based on mode:
      - "en": do nothing
      - "zh": add views.zh
      - "both": add views.zh and views.both

    Canonical payload keys remain unchanged.
    """
    m = (mode or "en").strip().lower()
    if m not in ("en", "zh", "both"):
        m = "en"

    if m == "en":
        return payload

    base = dict(payload)
    base.pop("views", None)

    views: Dict[str, Any] = {}

    if m in ("zh", "both"):
        views["zh"] = _transform(base, lambda k: _label_zh(k) or k, mode="zh")

    if m == "both":
        views["both"] = _transform(base, _label_both, mode="both")

    out = dict(payload)
    out["views"] = views
    return out
