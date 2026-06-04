from enum import Enum


class OISource(str, Enum):
    SETTLED   = "settled"    # prior-day EOD settlement — yfinance, CBOE delayed
    ESTIMATED = "estimated"  # intraday flow classification — requires paid data (future)
    LIVE      = "live"       # real-time feed — Tradier, prop account (future)


# Maps source_name strings (from data.py) to OISource classification
_SOURCE_MAP: dict[str, OISource] = {
    "yfinance": OISource.SETTLED,
    "cboe":     OISource.SETTLED,
    "tradier":  OISource.LIVE,
}


def classify_source(source_name: str) -> OISource:
    """
    Map a data source name to its OI quality classification.

    Unknown sources default to SETTLED (conservative — no false confidence).

    Args:
        source_name: "yfinance" | "cboe" | "tradier"

    Returns:
        OISource
    """
    return _SOURCE_MAP.get(source_name.lower(), OISource.SETTLED)
