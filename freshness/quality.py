from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone

from .sources import OISource, classify_source
from .market import market_hours


@dataclass
class SnapshotQuality:
    oi_source:         OISource   # quality of OI data in this snapshot
    fetched_at:        datetime   # UTC timestamp of the fetch
    market_open:       bool       # was US equity market open at fetch time?
    is_intraday_stale: bool       # = (oi_source == SETTLED) and market_open
    source_name:       str        # "yfinance" | "cboe" | "tradier"

    def to_dict(self) -> dict:
        """Serialise for API response embedding."""
        return {
            "oi_source":         self.oi_source.value,
            "is_intraday_stale": self.is_intraday_stale,
            "source_name":       self.source_name,
            "fetched_at":        self.fetched_at.isoformat(),
            "market_open":       self.market_open,
        }


def assess_snapshot(source_name: str,
                    fetched_at: datetime | None = None) -> SnapshotQuality:
    """
    Classify a data snapshot by OI source and derive the intraday staleness flag.

    Args:
        source_name: "yfinance" | "cboe" | "tradier"
        fetched_at:  UTC timestamp of the fetch (default: now)

    Returns:
        SnapshotQuality
    """
    if fetched_at is None:
        fetched_at = datetime.now(tz=timezone.utc)

    oi_source = classify_source(source_name)
    open_now  = market_hours(fetched_at)
    stale     = (oi_source == OISource.SETTLED) and open_now

    return SnapshotQuality(
        oi_source         = oi_source,
        fetched_at        = fetched_at,
        market_open       = open_now,
        is_intraday_stale = stale,
        source_name       = source_name,
    )
