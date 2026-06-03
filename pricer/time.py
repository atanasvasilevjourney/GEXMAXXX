from datetime import datetime
from zoneinfo import ZoneInfo

ET = ZoneInfo('America/New_York')

# 60-second absolute floor — avoids QuantLib singularity at T=0
MIN_T_YEARS: float = 60 / (365.25 * 24 * 3600)

# Expiry clock by instrument type (ET)
_EXPIRY_TIMES: dict[str, tuple[int, int]] = {
    'equity':  (16,  0),   # 4:00 PM ET — equity options (SPX, QQQ, SPY, etc.)
    'futures': (16, 15),   # 4:15 PM ET — CME equity futures options (ES, NQ, MNQ)
}


def time_to_expiry(expiration_str: str, style: str = 'equity') -> float:
    """
    Compute time-to-expiry in years from now to expiry time ET on expiration_str.

    Replaces the old T.clip(lower=0.001) which mapped a 30-min 0DTE to
    T=0.001yr (8.76 hr), inflating gamma ~4x. This function uses exact
    minutes remaining with a 60-second floor.

    Args:
        expiration_str: 'YYYY-MM-DD'
        style:          'equity' (4:00 PM ET, default) or 'futures' (4:15 PM ET)

    Returns:
        T in years, >= MIN_T_YEARS (never negative, never zero)

    Raises:
        ValueError: if expiration_str is not 'YYYY-MM-DD'
    """
    expiry_hour, expiry_minute = _EXPIRY_TIMES.get(style, _EXPIRY_TIMES['equity'])
    now = datetime.now(tz=ET)
    try:
        exp = datetime.strptime(expiration_str, '%Y-%m-%d').replace(
            hour=expiry_hour, minute=expiry_minute, second=0, microsecond=0, tzinfo=ET
        )
    except ValueError:
        raise ValueError(
            f"time_to_expiry: invalid expiration_str {expiration_str!r}. "
            f"Expected format 'YYYY-MM-DD'."
        )
    secs = (exp - now).total_seconds()
    return max(secs, 60) / (365.25 * 24 * 3600)
