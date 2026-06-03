from datetime import datetime
from zoneinfo import ZoneInfo

ET = ZoneInfo('America/New_York')

# 60-second absolute floor — avoids QuantLib singularity at T=0
MIN_T_YEARS: float = 60 / (365.25 * 24 * 3600)


def time_to_expiry(expiration_str: str) -> float:
    """
    Compute time-to-expiry in years from now to 4:00 PM ET on expiration_str.

    Replaces the old T.clip(lower=0.001) which mapped a 30-min 0DTE to
    T=0.001yr (8.76 hr), inflating gamma ~4x. This function uses exact
    minutes remaining with a 60-second floor.

    Args:
        expiration_str: 'YYYY-MM-DD'

    Returns:
        T in years, >= MIN_T_YEARS (never negative, never zero)
    """
    now = datetime.now(tz=ET)
    exp = datetime.strptime(expiration_str, '%Y-%m-%d').replace(
        hour=16, minute=0, second=0, microsecond=0, tzinfo=ET
    )
    secs = (exp - now).total_seconds()
    return max(secs, 60) / (365.25 * 24 * 3600)
