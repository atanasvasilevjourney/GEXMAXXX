# GPR Freshness — Design Spec
**Date:** 2026-06-03
**Status:** Approved
**Sub-project:** 6 of 10 — Staleness Detection + OI Source Flags

## Overview

yfinance and CBOE delayed quotes always return **prior-day settled OI**. During market hours the gamma levels computed from this data reflect yesterday's positioning, not today's. The system currently serves this data without any flag, which the Part A audit classified as a blocker for live use.

This sub-project adds a `freshness/` package that classifies OI source (settled/estimated/live), detects whether the market is open, and derives an `is_intraday_stale` flag. The flag propagates to the API response and feeds into the signal layer via `MarketTick.in_event_window=True`, blocking new trade arming on stale gamma levels.

## Part A Audit Finding Resolved

| Finding | Severity | Fix |
|---|---|---|
| Staleness never surfaced; stale snapshot drives signal | Blocker | `SnapshotQuality.is_intraday_stale` in API; signal reads it |

## Approach

Option A — `freshness/` standalone package. Follows the established package pattern (`pricer/`, `regime/`, `level_projection/`). Clean separation between data fetching and quality assessment. Testable in isolation with no network calls.

---

## File Structure

```
D:/2026/GEX/
├── freshness/
│   ├── __init__.py      # exports: OISource, SnapshotQuality, assess_snapshot, market_hours
│   ├── sources.py       # OISource enum
│   ├── quality.py       # SnapshotQuality dataclass + assess_snapshot()
│   └── market.py        # market_hours() — is US equity market open?
└── tests/
    └── test_freshness.py  # 10 tests
```

**Modified files:**
- `data.py` — `fetch_chain()` returns `(df, spot, SnapshotQuality)`; `fetch_chain_cboe()` same
- `serve.py` — `_run_pipeline_sync()` unpacks three-tuple; `/api/gex` response includes `oi_source` and `is_intraday_stale`

**NOT modified:**
- `gex_calc.py` — standalone legacy script with its own `fetch_chain`; unaffected
- `signal_gpr/` — state machine unchanged; caller sets `in_event_window=True` based on quality flag

---

## Module Specifications

### `freshness/sources.py`

```python
from enum import Enum

class OISource(str, Enum):
    SETTLED   = "settled"    # prior-day EOD settlement — yfinance, CBOE delayed
    ESTIMATED = "estimated"  # intraday flow classification — requires paid data (future)
    LIVE      = "live"       # real-time feed — Tradier, prop account (future)

# Maps source_name strings returned by data.py to OISource
_SOURCE_MAP: dict[str, OISource] = {
    "yfinance": OISource.SETTLED,
    "cboe":     OISource.SETTLED,
    "tradier":  OISource.LIVE,
}

def classify_source(source_name: str) -> OISource:
    """
    Map a data source name to its OI quality classification.

    Unknown sources default to SETTLED (conservative).
    """
    return _SOURCE_MAP.get(source_name.lower(), OISource.SETTLED)
```

---

### `freshness/market.py`

```python
from datetime import datetime, time
from zoneinfo import ZoneInfo

ET = ZoneInfo('America/New_York')

MARKET_OPEN  = time(9, 30)
MARKET_CLOSE = time(16, 0)


def market_hours(now: datetime | None = None) -> bool:
    """
    Return True if the US equity market is currently open.

    Checks: Monday–Friday, 09:30–16:00 ET.
    Does NOT account for market holidays (conservative: a holiday will appear open,
    but OI will still be classified as SETTLED → is_intraday_stale=True → no signals).

    Args:
        now: datetime to check (default: datetime.now(ET)). Must be timezone-aware.

    Returns:
        bool
    """
    if now is None:
        now = datetime.now(tz=ET)
    now_et = now.astimezone(ET)
    if now_et.weekday() >= 5:          # Saturday=5, Sunday=6
        return False
    t = now_et.time()
    return MARKET_OPEN <= t < MARKET_CLOSE
```

---

### `freshness/quality.py`

```python
from __future__ import annotations
from dataclasses import dataclass, field
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
        """Serialise for API response."""
        return {
            "oi_source":         self.oi_source.value,
            "is_intraday_stale": self.is_intraday_stale,
            "source_name":       self.source_name,
            "fetched_at":        self.fetched_at.isoformat(),
            "market_open":       self.market_open,
        }


def assess_snapshot(source_name: str, fetched_at: datetime | None = None) -> SnapshotQuality:
    """
    Classify a data snapshot by OI source and derive the intraday staleness flag.

    Args:
        source_name: "yfinance" | "cboe" | "tradier" (from data.py)
        fetched_at:  UTC timestamp of the fetch (default: now)

    Returns:
        SnapshotQuality
    """
    if fetched_at is None:
        fetched_at = datetime.now(tz=timezone.utc)

    oi_source   = classify_source(source_name)
    open_now    = market_hours(fetched_at)
    stale       = (oi_source == OISource.SETTLED) and open_now

    return SnapshotQuality(
        oi_source         = oi_source,
        fetched_at        = fetched_at,
        market_open       = open_now,
        is_intraday_stale = stale,
        source_name       = source_name,
    )
```

---

### `freshness/__init__.py`

```python
from .sources import OISource, classify_source
from .quality import SnapshotQuality, assess_snapshot
from .market import market_hours

__all__ = [
    'OISource', 'classify_source',
    'SnapshotQuality', 'assess_snapshot',
    'market_hours',
]
```

---

### Integration: `data.py`

Both `fetch_chain` and `fetch_chain_cboe` gain a third return value:

```python
from freshness import assess_snapshot

def fetch_chain(ticker: str) -> tuple[pd.DataFrame, float, SnapshotQuality]:
    # ... existing fetch logic unchanged ...
    quality = assess_snapshot("yfinance")
    return df, float(spot), quality

def fetch_chain_cboe(ticker: str) -> tuple[pd.DataFrame, float, SnapshotQuality]:
    # ... existing fetch logic unchanged ...
    quality = assess_snapshot("cboe")
    return df, spot, quality
```

`fetch_chain_tradier` stub updated to document that it would return `assess_snapshot("tradier")`.

---

### Integration: `serve.py`

`_run_pipeline_sync` unpacks the three-tuple and attaches quality to the result:

```python
# Before (two-tuple):
df, spot = fetch_chain(ticker)
source = "yfinance"

# After (three-tuple):
df, spot, quality = fetch_chain(ticker)
```

And on the fallback path:
```python
df, spot, quality = fetch_chain_cboe(ticker)
```

The quality dict is merged into the `levels` result:
```python
levels.update(quality.to_dict())
```

`/api/gex` response automatically includes `oi_source`, `is_intraday_stale`, `source_name`, `fetched_at`, `market_open` for each ticker — no API endpoint changes required.

---

### Signal integration (caller convention)

The `GPRStateMachine._handle_in_trade` checks `in_event_window` as a **forced-exit trigger**. Setting it to `True` on every tick during market hours (where every yfinance refresh returns settled OI) would force-exit every IN_TRADE position immediately. This is wrong — a trade entered on good gamma levels should not be force-exited just because the next 5-minute refresh returned the same settled OI.

**Correct convention:** the caller sets `in_event_window` based on **both** staleness and current state:

```python
stale = levels.get("is_intraday_stale", False)
in_trade = (sm.state == SignalState.IN_TRADE)

# Only block arming on stale data; never force-exit a running trade for staleness alone
tick = MarketTick(
    price=price,
    atr=atr,
    in_event_window=stale and not in_trade,
)
sm.on_tick(tick, regime, fut_levels)
```

This way:
- **IDLE / ARMED:** `in_event_window=True` when stale → blocks `IDLE → ARMED` transition
- **IN_TRADE:** `in_event_window=False` regardless of staleness → trade runs to natural exit (stop, target, regime flip)

The state machine itself is unchanged. The caller holds the reference to `sm` and has access to `sm.state`.

---

## Tests (`tests/test_freshness.py`)

All 10 tests are synthetic — no network calls, no yfinance.

```python
# 1
def test_oi_source_settled_value():
    assert OISource.SETTLED.value == "settled"
    assert OISource.LIVE.value    == "live"
    assert OISource.ESTIMATED.value == "estimated"

# 2
def test_assess_snapshot_yfinance_settled():
    q = assess_snapshot("yfinance", fetched_at=_outside_market())
    assert q.oi_source == OISource.SETTLED
    assert q.source_name == "yfinance"

# 3
def test_assess_snapshot_cboe_settled():
    q = assess_snapshot("cboe", fetched_at=_outside_market())
    assert q.oi_source == OISource.SETTLED

# 4
def test_assess_snapshot_tradier_live():
    q = assess_snapshot("tradier", fetched_at=_outside_market())
    assert q.oi_source == OISource.LIVE

# 5
def test_market_hours_open():
    # Tuesday 10:00 AM ET
    dt = datetime(2026, 6, 2, 10, 0, 0, tzinfo=ZoneInfo('America/New_York'))
    assert market_hours(dt) is True

# 6
def test_market_hours_closed_weekend():
    # Saturday
    dt = datetime(2026, 6, 6, 10, 0, 0, tzinfo=ZoneInfo('America/New_York'))
    assert market_hours(dt) is False

# 7
def test_market_hours_closed_premarket():
    # Monday 8:00 AM ET
    dt = datetime(2026, 6, 1, 8, 0, 0, tzinfo=ZoneInfo('America/New_York'))
    assert market_hours(dt) is False

# 8
def test_market_hours_closed_afterhours():
    # Wednesday 5:00 PM ET
    dt = datetime(2026, 6, 3, 17, 0, 0, tzinfo=ZoneInfo('America/New_York'))
    assert market_hours(dt) is False

# 9
def test_is_intraday_stale_during_market():
    # SETTLED source during market hours → stale
    dt = _inside_market()
    q = assess_snapshot("yfinance", fetched_at=dt)
    assert q.is_intraday_stale is True
    assert q.market_open is True

# 10
def test_is_intraday_stale_outside_market():
    # SETTLED source outside market hours → NOT stale
    dt = _outside_market()
    q = assess_snapshot("yfinance", fetched_at=dt)
    assert q.is_intraday_stale is False
    assert q.market_open is False

# Helpers
def _inside_market() -> datetime:
    """Tuesday 10:30 AM ET in UTC."""
    return datetime(2026, 6, 2, 14, 30, 0, tzinfo=timezone.utc)  # 10:30 AM ET

def _outside_market() -> datetime:
    """Saturday 10:00 AM ET in UTC."""
    return datetime(2026, 6, 6, 14, 0, 0, tzinfo=timezone.utc)   # Saturday
```

---

## Acceptance Criteria

- All 10 tests pass
- `assess_snapshot("yfinance")` → `OISource.SETTLED`
- `assess_snapshot("tradier")` → `OISource.LIVE`
- `market_hours()` returns `False` on weekends and outside 09:30–16:00 ET
- `is_intraday_stale = True` iff `oi_source == SETTLED` AND market is open
- `fetch_chain()` and `fetch_chain_cboe()` return three-tuple `(df, spot, SnapshotQuality)`
- `/api/gex` response includes `oi_source`, `is_intraday_stale`, `market_open` per ticker
- No network calls in `freshness/`
- Existing 106 tests still pass

---

## Relationship to GPR Sub-projects

| Sub-project | Note |
|---|---|
| 5 — pricer (done) | Correctness gate; freshness is orthogonal |
| 6 — freshness (this) | OI quality flag; prerequisite for honest statistical validation |
| 7 — statistical validation | PBO/DSR/Monte Carlo; needs correct + honest data quality context |
| 8 — nautilus_trader | Event-driven backtest→live; will consume SnapshotQuality |
