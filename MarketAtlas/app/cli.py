from __future__ import annotations

import argparse
from datetime import date, timedelta
from typing import Optional, List

from ..core.config import load_config
from ..orchestrators.ingest_daily import ingest_daily_all
from ..orchestrators.ingest_intraday_5m import ingest_intraday_for_date
from ..orchestrators.ingest_all import ingest_all
from ..orchestrators.update_session_state import update_session_state_all
from ..orchestrators.compute_levels import compute_levels_all
from ..orchestrators.intraday_features import build_intraday_features_all, build_intraday_features_one
from ..orchestrators.today_range import nowcast_today_range_all, nowcast_today_range_one_ticker
from ..orchestrators.intraday_insights import intraday_insights_all, intraday_insights_one
from ..orchestrators.intraday_run import intraday_run
from ..orchestrators.daily_features import daily_features_all, daily_features_one
from ..orchestrators.regime_labels import regime_labels_all, regime_labels_one
from ..orchestrators.train_spx_regime import train_spx_regime_model
from ..orchestrators.train_m7_regime import train_m7_regime_models
from ..orchestrators.tomorrow_range import tomorrow_range_all, tomorrow_range_one
from ..orchestrators.eod_run import eod_run_all


def _parse_iso_day(s: str) -> str:
    date.fromisoformat(s)
    return s


def _norm_ticker(t: Optional[str]) -> Optional[str]:
    if not t:
        return None
    x = str(t).strip().upper()
    return x if x else None


def _ticker_list_or_none(t: Optional[str]) -> Optional[List[str]]:
    nt = _norm_ticker(t)
    return [nt] if nt else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="MarketAtlas", description="MarketAtlas CLI")

    parser.add_argument(
        "--root",
        type=str,
        default=".",
        help="Project root (folder containing config/ and data/).",
    )

    sub = parser.add_subparsers(dest="cmd", required=True)

    # ingest-daily
    p_daily = sub.add_parser("ingest-daily", help="Ingest daily bars for configured universe.")
    p_daily.add_argument("--start", type=str, default=None, help="Start date YYYY-MM-DD (optional).")
    p_daily.add_argument("--end", type=str, default=None, help="End date YYYY-MM-DD (optional).")
    p_daily.add_argument("--date", type=_parse_iso_day, default=None, help="Single day YYYY-MM-DD (optional).")
    p_daily.add_argument("--ticker", type=str, default=None, help="Optional single ticker (e.g., AAPL).")

    # ingest-5m
    p_5m = sub.add_parser("ingest-5m", help="Ingest intraday 5-minute bars for configured universe.")
    p_5m.add_argument("--date", type=str, required=True, help="Trading day YYYY-MM-DD.")
    p_5m.add_argument("--lookback", type=int, default=None, help="Lookback days (default from config).")
    p_5m.add_argument("--no-extended", action="store_true", help="Disable pre/after-hours where supported.")
    p_5m.add_argument("--ticker", type=str, default=None, help="Optional single ticker (e.g., AAPL)")

    # ingest-all
    p_all = sub.add_parser("ingest-all", help="Ingest both daily and intraday 5m.")
    p_all.add_argument("--daily-start", type=str, default=None)
    p_all.add_argument("--daily-end", type=str, default=None)
    p_all.add_argument("--date", type=str, required=True, help="Trading day YYYY-MM-DD for intraday.")
    p_all.add_argument("--lookback", type=int, default=None)

    # session-state
    p_ss = sub.add_parser("session-state", help="Build/update session state JSON from intraday 5m bars.")
    p_ss.add_argument("--date", type=str, required=True, help="Trading day YYYY-MM-DD.")
    p_ss.add_argument("--ticker", type=str, default=None, help="Optional single ticker")

    # levels
    p_lv = sub.add_parser("levels", help="Compute support/resistance zones from daily bars + session state.")
    p_lv.add_argument("--date", type=str, required=True, help="Trading day YYYY-MM-DD.")
    p_lv.add_argument("--ticker", type=str, default=None, help="Optional single ticker")

    # intraday-features
    p_feat = sub.add_parser("intraday-features", help="Step 4: build intraday features (5m) for tickers")
    p_feat.add_argument("--date", type=str, required=True, help="YYYY-MM-DD")
    p_feat.add_argument("--ticker", type=str, default=None, help="Optional single ticker (e.g., TSLA)")

    # today-range
    p = sub.add_parser("today-range", help="Step 5: nowcast today's remaining-session and conditional full-day range")
    p.add_argument("--date", required=True, help="YYYY-MM-DD")
    p.add_argument("--ticker", default=None, help="optional single ticker")

    # intraday-insights
    p_ins = sub.add_parser("intraday-insights", help="Step 6: generate intraday insights")
    p_ins.add_argument("--date", type=str, required=True, help="YYYY-MM-DD")
    p_ins.add_argument("--ticker", type=str, default=None, help="Optional single ticker")

    # intraday-run
    p_run = sub.add_parser("intraday-run", help="Step 7: run the full intraday pipeline (steps 1-6) in order")
    p_run.add_argument("--date", type=str, required=True, help="YYYY-MM-DD")
    p_run.add_argument("--lookback", type=int, default=None, help="Override lookback days for ingest-5m")
    p_run.add_argument("--no-extended", action="store_true", help="Disable pre/after-hours where supported")
    p_run.add_argument("--ticker", type=str, default=None, help="Optional single ticker (e.g., TSLA)")

    # daily-features
    p_df = sub.add_parser("daily-features", help="Step 8: build daily features for training/EOD")
    p_df.add_argument("--date", type=str, required=True, help="YYYY-MM-DD")
    p_df.add_argument("--ticker", type=str, default=None, help="Optional single ticker (e.g., SPY)")

    # regime-labels
    p_rl = sub.add_parser("regime-labels", help="Step 9: apply rule-based regime labels from daily features")
    p_rl.add_argument("--date", type=str, required=True, help="YYYY-MM-DD")
    p_rl.add_argument("--ticker", type=str, default=None, help="Optional single ticker (e.g., SPY)")

    # train-spx-regime
    p_tspx = sub.add_parser("train-spx-regime", help="Step 10: train SPX daily regime classifier")
    p_tspx.add_argument("--date", type=str, required=True, help="YYYY-MM-DD")
    p_tspx.add_argument("--version", type=str, default="v1", help="Model version folder name (e.g., v1)")

    # train-m7-regime
    p_m7 = sub.add_parser("train-m7-regime", help="Step 11: train M7 daily regime models")
    p_m7.add_argument("--date", type=str, required=True, help="YYYY-MM-DD")
    p_m7.add_argument("--version", type=str, required=True, help="Model version (e.g., v1)")
    p_m7.add_argument("--ticker", type=str, default=None, help="Optional single ticker (e.g., NVDA)")

    # tomorrow-range
    p_tr = sub.add_parser("tomorrow-range", help="Step 12: EOD tomorrow range forecast")
    p_tr.add_argument("--date", type=str, required=True, help="EOD run day YYYY-MM-DD")
    p_tr.add_argument("--ticker", type=str, default=None, help="Optional single ticker")

    # eod-run
    p_eod = sub.add_parser("eod-run", help="Step 13: run EOD pipeline to write tomorrow plans")
    p_eod.add_argument("--date", type=str, required=True, help="EOD run day YYYY-MM-DD")
    p_eod.add_argument("--ticker", type=str, default=None, help="Optional single ticker")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    cfg = load_config(args.root)

    if args.cmd == "ingest-daily":
        start = args.start
        end = args.end
        if args.date:
            d0 = date.fromisoformat(args.date)
            start = d0.isoformat()
            end = (d0 + timedelta(days=1)).isoformat()

        tickers = _ticker_list_or_none(args.ticker)
        res = ingest_daily_all(cfg, start=start, end=end, tickers=tickers)
        for k, v in res.items():
            print(k, vars(v))
        return

    if args.cmd == "ingest-5m":
        include_ext = not args.no_extended
        lookback = args.lookback if args.lookback is not None else int(
            cfg.raw.get("ingestion", {}).get("intraday_5m", {}).get("default_lookback_days", 5)
        )
        tickers = _ticker_list_or_none(args.ticker)
        res = ingest_intraday_for_date(
            cfg,
            day=args.date,
            lookback_days=lookback,
            include_extended_hours=include_ext,
            tickers=tickers,
        )
        for k, v in res.items():
            print(k, vars(v))
        return

    if args.cmd == "ingest-all":
        res = ingest_all(
            cfg,
            daily_start=args.daily_start,
            daily_end=args.daily_end,
            intraday_day=args.date,
            intraday_lookback_days=args.lookback,
        )
        print("daily:", res.daily)
        print("intraday:", res.intraday)
        return

    if args.cmd == "session-state":
        # ✅ FIX: function expects `tickers=`, not `ticker=`
        tickers = _ticker_list_or_none(args.ticker)
        res = update_session_state_all(cfg, day=args.date, tickers=tickers)
        for k, v in res.items():
            print(k, v)
        return

    if args.cmd == "levels":
        # ✅ Prefer `tickers=` for consistency (most "all" orchestrators use tickers list)
        tickers = _ticker_list_or_none(args.ticker)
        res = compute_levels_all(cfg, day=args.date, tickers=tickers)
        for k, v in res.items():
            print(k, v)
        return

    if args.cmd == "intraday-features":
        t = _norm_ticker(args.ticker)
        if t:
            res = build_intraday_features_one(cfg, t, day=args.date)
            print(res)
        else:
            res = build_intraday_features_all(cfg, day=args.date)
            print(res)
        return

    if args.cmd == "today-range":
        t = _norm_ticker(args.ticker)
        if t:
            res = nowcast_today_range_one_ticker(cfg, ticker=t, day=args.date)
        else:
            res = nowcast_today_range_all(cfg, day=args.date)
        print(res)
        return

    if args.cmd == "intraday-insights":
        t = _norm_ticker(args.ticker)
        if t:
            res = intraday_insights_one(cfg, ticker=t, day=args.date)
        else:
            res = intraday_insights_all(cfg, day=args.date)
        print(res)
        return

    if args.cmd == "intraday-run":
        include_ext = not args.no_extended
        t = _norm_ticker(args.ticker)
        res = intraday_run(
            cfg,
            day=args.date,
            lookback_days=args.lookback,
            include_extended_hours=include_ext,
            ticker=t,
        )
        print({"day": res.day, "status": res.status, "error": res.error, "stages": res.stages})
        return

    if args.cmd == "daily-features":
        t = _norm_ticker(args.ticker)
        if t:
            res = daily_features_one(cfg, ticker=t, day=args.date)
        else:
            res = daily_features_all(cfg, day=args.date)
        print(res)
        return

    if args.cmd == "regime-labels":
        t = _norm_ticker(args.ticker)
        if t:
            res = regime_labels_one(cfg, ticker=t, day=args.date)
        else:
            res = regime_labels_all(cfg, day=args.date)
        print(res)
        return

    if args.cmd == "train-spx-regime":
        res = train_spx_regime_model(cfg, day=args.date, version=args.version)
        print(res)
        return

    if args.cmd == "train-m7-regime":
        t = _norm_ticker(args.ticker)
        res = train_m7_regime_models(cfg, day=args.date, version=args.version, ticker=t)
        print(res)
        return

    if args.cmd == "tomorrow-range":
        t = _norm_ticker(args.ticker)
        if t:
            res = tomorrow_range_one(cfg, ticker=t, day=args.date)
        else:
            res = tomorrow_range_all(cfg, day=args.date)
        print(res)
        return

    if args.cmd == "eod-run":
        t = _norm_ticker(args.ticker)
        res = eod_run_all(cfg, day=args.date, ticker=t)
        print(res)
        return


if __name__ == "__main__":
    main()
