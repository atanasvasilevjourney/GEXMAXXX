from datetime import datetime, time
from zoneinfo import ZoneInfo

ET = ZoneInfo('America/New_York')

MARKET_OPEN  = time(9, 30)
MARKET_CLOSE = time(16, 0)


def market_hours(now: datetime | None = None) -> bool:
    """
    Return True if the US equity market is currently open.

    Checks: Monday–Friday, 09:30–16:00 ET.
    Does NOT account for market holidays (conservative: a holiday appears open,
    but OI will still be SETTLED → is_intraday_stale=True → no new arming).

    Args:
        now: datetime to evaluate (default: datetime.now(ET)).
             Must be timezone-aware if provided.

    Returns:
        bool
    """
    if now is None:
        now = datetime.now(tz=ET)
    now_et = now.astimezone(ET)
    if now_et.weekday() >= 5:   # Saturday=5, Sunday=6
        return False
    t = now_et.time()
    return MARKET_OPEN <= t < MARKET_CLOSE
