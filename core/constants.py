from __future__ import annotations

REQUIRED_BAR_COLS = ["open", "high", "low", "close", "volume"]
CANONICAL_COLS = ["open", "high", "low", "close", "volume", "vwap", "session"]

SESSION_PRE = "PRE"
SESSION_RTH = "RTH"
SESSION_AFT = "AFT"
SESSION_CLOSED = "CLOSED"

# US equities typical sessions (ET)
PRE_START = (4, 0)     # 04:00
RTH_START = (9, 30)    # 09:30
RTH_END = (16, 0)      # 16:00
AFT_END = (20, 0)      # 20:00
