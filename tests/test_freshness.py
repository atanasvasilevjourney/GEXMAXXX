import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from freshness import OISource, classify_source, market_hours, SnapshotQuality, assess_snapshot


# --- Helpers ---

def _inside_market() -> datetime:
    """Tuesday 2026-06-02 10:30 AM ET expressed in UTC."""
    return datetime(2026, 6, 2, 14, 30, 0, tzinfo=timezone.utc)

def _outside_market() -> datetime:
    """Saturday 2026-06-06 10:00 AM ET expressed in UTC."""
    return datetime(2026, 6, 6, 14, 0, 0, tzinfo=timezone.utc)


# --- OISource ---

def test_oi_source_values():
    assert OISource.SETTLED.value   == "settled"
    assert OISource.ESTIMATED.value == "estimated"
    assert OISource.LIVE.value      == "live"


# --- market_hours ---

def test_market_hours_open():
    # Tuesday 10:00 AM ET → open
    dt = datetime(2026, 6, 2, 10, 0, 0, tzinfo=ZoneInfo('America/New_York'))
    assert market_hours(dt) is True


def test_market_hours_closed_weekend():
    # Saturday → closed
    dt = datetime(2026, 6, 6, 10, 0, 0, tzinfo=ZoneInfo('America/New_York'))
    assert market_hours(dt) is False


def test_market_hours_closed_premarket():
    # Monday 8:00 AM ET → closed (before 09:30)
    dt = datetime(2026, 6, 1, 8, 0, 0, tzinfo=ZoneInfo('America/New_York'))
    assert market_hours(dt) is False


def test_market_hours_closed_afterhours():
    # Wednesday 5:00 PM ET → closed (after 16:00)
    dt = datetime(2026, 6, 3, 17, 0, 0, tzinfo=ZoneInfo('America/New_York'))
    assert market_hours(dt) is False


# --- market_hours boundary ---

def test_market_hours_open_at_930():
    # Exactly 09:30:00 ET → open (boundary inclusive)
    dt = datetime(2026, 6, 2, 9, 30, 0, tzinfo=ZoneInfo('America/New_York'))
    assert market_hours(dt) is True


def test_market_hours_closed_at_1600():
    # Exactly 16:00:00 ET → closed (boundary exclusive)
    dt = datetime(2026, 6, 2, 16, 0, 0, tzinfo=ZoneInfo('America/New_York'))
    assert market_hours(dt) is False


def test_market_hours_naive_datetime_raises():
    # Naive datetime should raise ValueError, not silently convert
    from datetime import datetime as dt_cls
    naive = dt_cls(2026, 6, 2, 10, 0, 0)  # no tzinfo
    with pytest.raises(ValueError, match="timezone-aware"):
        market_hours(naive)


# --- classify_source ---

def test_classify_source_yfinance():
    assert classify_source("yfinance") == OISource.SETTLED

def test_classify_source_cboe():
    assert classify_source("cboe") == OISource.SETTLED

def test_classify_source_tradier():
    assert classify_source("tradier") == OISource.LIVE

def test_classify_source_unknown_defaults_settled():
    assert classify_source("unknown_source") == OISource.SETTLED

def test_classify_source_case_insensitive():
    assert classify_source("YFinance") == OISource.SETTLED
    assert classify_source("CBOE") == OISource.SETTLED


# --- assess_snapshot ---

def test_assess_snapshot_yfinance_settled():
    q = assess_snapshot("yfinance", fetched_at=_outside_market())
    assert q.oi_source   == OISource.SETTLED
    assert q.source_name == "yfinance"


def test_assess_snapshot_cboe_settled():
    q = assess_snapshot("cboe", fetched_at=_outside_market())
    assert q.oi_source == OISource.SETTLED


def test_assess_snapshot_tradier_live():
    q = assess_snapshot("tradier", fetched_at=_outside_market())
    assert q.oi_source == OISource.LIVE


def test_is_intraday_stale_during_market():
    # SETTLED source during market hours → is_intraday_stale=True
    q = assess_snapshot("yfinance", fetched_at=_inside_market())
    assert q.is_intraday_stale is True
    assert q.market_open is True


def test_is_intraday_stale_outside_market():
    # SETTLED source outside market hours → is_intraday_stale=False
    q = assess_snapshot("yfinance", fetched_at=_outside_market())
    assert q.is_intraday_stale is False
    assert q.market_open is False
