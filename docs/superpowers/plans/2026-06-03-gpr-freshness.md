# GPR Freshness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `freshness/` package — OI source classification, market-hours detection, and intraday staleness flag — and wire it into `data.py` and `serve.py`.

**Architecture:** Three focused modules (`sources.py`, `market.py`, `quality.py`) in a new `freshness/` package. `data.py:fetch_chain` and `fetch_chain_cboe` return a third value `SnapshotQuality`. `serve.py` unpacks the three-tuple and merges quality fields into the API response.

**Tech Stack:** Python 3.13, zoneinfo (stdlib), dataclasses (stdlib), pytest

---

## File Map

| File | Action | Purpose |
|---|---|---|
| `freshness/__init__.py` | Create | Public API exports |
| `freshness/sources.py` | Create | OISource enum + classify_source() |
| `freshness/market.py` | Create | market_hours() — US equity open detection |
| `freshness/quality.py` | Create | SnapshotQuality dataclass + assess_snapshot() |
| `tests/test_freshness.py` | Create | 10 fixture-based tests |
| `data.py` | Modify | fetch_chain + fetch_chain_cboe return 3-tuple |
| `serve.py` | Modify | Unpack 3-tuple; merge quality.to_dict() into levels |

---

## Task 1: OISource + market_hours (5 tests)

**Files:**
- Create: `D:/2026/GEX/freshness/__init__.py` (stub)
- Create: `D:/2026/GEX/freshness/sources.py`
- Create: `D:/2026/GEX/freshness/market.py`
- Create: `D:/2026/GEX/tests/test_freshness.py`

- [ ] **Step 1: Write 5 failing tests**

Create `D:/2026/GEX/tests/test_freshness.py`:

```python
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from freshness import OISource, classify_source, market_hours, assess_snapshot, SnapshotQuality


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


# --- assess_snapshot (added in Task 2) ---
# DO NOT add assess_snapshot tests here yet.
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd D:/2026/GEX
python -m pytest tests/test_freshness.py::test_oi_source_values tests/test_freshness.py::test_market_hours_open tests/test_freshness.py::test_market_hours_closed_weekend tests/test_freshness.py::test_market_hours_closed_premarket tests/test_freshness.py::test_market_hours_closed_afterhours -v 2>&1 | head -10
```

Expected: `ModuleNotFoundError: No module named 'freshness'`

- [ ] **Step 3: Create `freshness/sources.py`**

```python
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
```

- [ ] **Step 4: Create `freshness/market.py`**

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
```

- [ ] **Step 5: Create stub `freshness/__init__.py`**

```python
from .sources import OISource, classify_source
from .market import market_hours

# assess_snapshot and SnapshotQuality added in Task 2
__all__ = ['OISource', 'classify_source', 'market_hours']
```

- [ ] **Step 6: Run 5 tests**

```bash
cd D:/2026/GEX
python -m pytest tests/test_freshness.py::test_oi_source_values tests/test_freshness.py::test_market_hours_open tests/test_freshness.py::test_market_hours_closed_weekend tests/test_freshness.py::test_market_hours_closed_premarket tests/test_freshness.py::test_market_hours_closed_afterhours -v
```

Expected: 5 passed

- [ ] **Step 7: Run full suite (no regressions)**

```bash
cd D:/2026/GEX
python -m pytest --ignore=tests/test_freshness.py -v 2>&1 | tail -3
```

Expected: 106 passed

- [ ] **Step 8: Commit**

```bash
cd D:/2026/GEX
git add freshness/__init__.py freshness/sources.py freshness/market.py tests/test_freshness.py
git commit -m "feat: add freshness package scaffold — OISource, classify_source, market_hours"
```

---

## Task 2: SnapshotQuality + assess_snapshot (5 tests)

**Files:**
- Create: `D:/2026/GEX/freshness/quality.py`
- Modify: `D:/2026/GEX/freshness/__init__.py` (add SnapshotQuality, assess_snapshot)
- Modify: `D:/2026/GEX/tests/test_freshness.py` (append 5 tests)

- [ ] **Step 1: Append 5 tests to `tests/test_freshness.py`**

Remove the "DO NOT add" comment and add these functions after the existing 5 tests:

```python
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
```

- [ ] **Step 2: Run to verify they fail**

```bash
cd D:/2026/GEX
python -m pytest tests/test_freshness.py -k "assess_snapshot or intraday_stale" -v 2>&1 | head -15
```

Expected: `ImportError: cannot import name 'assess_snapshot' from 'freshness'`

- [ ] **Step 3: Create `freshness/quality.py`**

```python
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
```

- [ ] **Step 4: Update `freshness/__init__.py`** (full version)

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

- [ ] **Step 5: Run all 10 freshness tests**

```bash
cd D:/2026/GEX
python -m pytest tests/test_freshness.py -v
```

Expected: 10 passed

- [ ] **Step 6: Run full suite**

```bash
cd D:/2026/GEX
python -m pytest -v 2>&1 | tail -3
```

Expected: 116 passed (106 + 10)

- [ ] **Step 7: Commit**

```bash
cd D:/2026/GEX
git add freshness/quality.py freshness/__init__.py tests/test_freshness.py
git commit -m "feat: add SnapshotQuality and assess_snapshot — intraday staleness detection"
```

---

## Task 3: data.py + serve.py integration

**Files:**
- Modify: `D:/2026/GEX/data.py`
- Modify: `D:/2026/GEX/serve.py`

No new tests — the existing 116 tests cover correctness. Integration verified by running the full suite.

- [ ] **Step 1: Modify `data.py`**

Add the import at the top (after existing imports):

```python
from freshness import assess_snapshot, SnapshotQuality
```

Replace the `return` statement in `fetch_chain` (currently `return df, float(spot)`) with:

```python
    quality = assess_snapshot("yfinance")
    return df, float(spot), quality
```

Replace the `return` statement in `fetch_chain_cboe` (currently `return df, spot`) with:

```python
    quality = assess_snapshot("cboe")
    return df, spot, quality
```

Update the `fetch_chain_tradier` stub docstring line to note it would return `assess_snapshot("tradier")`:

```python
def fetch_chain_tradier(ticker: str, api_key: str) -> tuple:
    """Tradier real-time options chain. Stub -- configure API key first.
    Returns: (df, spot, SnapshotQuality) where quality.oi_source == OISource.LIVE
    """
    raise NotImplementedError(
        "Tradier not configured. Sign up at tradier.com (free brokerage account), "
        "get API key, then implement this function in data.py."
    )
```

Full updated `data.py` (complete file for reference):

```python
import yfinance as yf
import pandas as pd
import requests
from datetime import datetime
from io import StringIO

from freshness import assess_snapshot, SnapshotQuality


def fetch_chain(ticker: str) -> tuple:
    """Fetch options chain from Yahoo Finance (primary source, 15-min delayed).

    Returns:
        (df, spot, SnapshotQuality) — OI is prior-day settled (OISource.SETTLED).
    """
    stock = yf.Ticker(ticker)
    spot = stock.fast_info.last_price
    if spot is None:
        spot = stock.info.get('regularMarketPrice')
    if spot is None:
        spot = stock.history(period='1d')['Close'].iloc[-1]

    expirations = stock.options[:4]  # nearest 4 expirations
    if not expirations:
        raise ValueError(f"No options expirations found for {ticker}")

    frames = []
    for exp in expirations:
        chain = stock.option_chain(exp)

        calls = chain.calls[['strike', 'openInterest', 'impliedVolatility']].copy()
        calls['type'] = 'call'

        puts = chain.puts[['strike', 'openInterest', 'impliedVolatility']].copy()
        puts['type'] = 'put'

        combined = pd.concat([calls, puts], ignore_index=True)
        combined['expiration'] = exp
        frames.append(combined)

    df = pd.concat(frames, ignore_index=True)
    quality = assess_snapshot("yfinance")
    return df, float(spot), quality


def fetch_chain_cboe(ticker: str) -> tuple:
    """Fetch options chain from CBOE delayed quote table (fallback, free, no account).

    Returns:
        (df, spot, SnapshotQuality) — OI is prior-day settled (OISource.SETTLED).
    """
    url = f"https://www.cboe.com/delayed_quotes/{ticker.lower()}/quote_table/"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()

    tables = pd.read_html(StringIO(resp.text))
    if len(tables) < 2:
        raise ValueError(f"CBOE page for {ticker} did not return expected tables")

    # CBOE renders calls (table 0) and puts (table 1)
    def parse_table(tbl, option_type):
        # CBOE column names vary; normalize common variants
        col_map = {}
        for col in tbl.columns:
            c = str(col).lower()
            if 'strike' in c:
                col_map[col] = 'strike'
            elif 'open' in c and 'int' in c:
                col_map[col] = 'openInterest'
            elif 'iv' in c or 'impl' in c:
                col_map[col] = 'impliedVolatility'
        tbl = tbl.rename(columns=col_map)
        needed = [c for c in ('strike', 'openInterest', 'impliedVolatility') if c in tbl.columns]
        tbl = tbl[needed].copy()
        tbl['type'] = option_type
        return tbl

    calls = parse_table(tables[0], 'call')
    puts  = parse_table(tables[1], 'put')

    # Use yfinance only for spot price
    stock = yf.Ticker(ticker)
    spot = float(stock.fast_info.last_price or stock.info.get('regularMarketPrice', 0))

    # IV from CBOE may be a percentage string like "25.3%" — normalize to decimal
    for df in (calls, puts):
        if 'impliedVolatility' in df.columns:
            df['impliedVolatility'] = (
                pd.to_numeric(
                    df['impliedVolatility'].astype(str).str.replace('%', '', regex=False),
                    errors='coerce'
                ) / 100
            ).fillna(0.20)
        if 'openInterest' in df.columns:
            df['openInterest'] = pd.to_numeric(df['openInterest'], errors='coerce').fillna(0).astype(int)

    # CBOE quote table is front-month only — use today as expiration placeholder
    exp = datetime.now().strftime('%Y-%m-%d')
    calls['expiration'] = exp
    puts['expiration']  = exp

    df = pd.concat([calls, puts], ignore_index=True)
    quality = assess_snapshot("cboe")
    return df, spot, quality


def fetch_chain_tradier(ticker: str, api_key: str) -> tuple:
    """Tradier real-time options chain. Stub -- configure API key first.
    Returns: (df, spot, SnapshotQuality) where quality.oi_source == OISource.LIVE
    """
    raise NotImplementedError(
        "Tradier not configured. Sign up at tradier.com (free brokerage account), "
        "get API key, then implement this function in data.py."
    )
```

- [ ] **Step 2: Verify data.py import works**

```bash
cd D:/2026/GEX
python -c "from data import fetch_chain, fetch_chain_cboe; print('data.py imports OK')"
```

Expected: `data.py imports OK`

- [ ] **Step 3: Modify `serve.py`**

The `_run_pipeline_sync` function currently unpacks two-tuples. Update it to unpack three-tuples and merge quality into `levels`.

Find this block (lines 48-58 approx):

```python
            try:
                df, spot = fetch_chain(ticker)
                source = "yfinance"
            except Exception as e_yf:
                print(f"[{ticker}] yfinance failed ({e_yf}), trying CBOE...")
                df, spot = fetch_chain_cboe(ticker)
                source = "cboe"
            df = calculate_all_greeks(df, spot)
            levels = get_all_levels(df, spot)
            levels = add_futures_conversion(levels, ticker, spot)
            levels["source"] = source
            levels["ticker"] = ticker
```

Replace with:

```python
            try:
                df, spot, quality = fetch_chain(ticker)
            except Exception as e_yf:
                print(f"[{ticker}] yfinance failed ({e_yf}), trying CBOE...")
                df, spot, quality = fetch_chain_cboe(ticker)
            df = calculate_all_greeks(df, spot)
            levels = get_all_levels(df, spot)
            levels = add_futures_conversion(levels, ticker, spot)
            levels.update(quality.to_dict())   # adds oi_source, is_intraday_stale, etc.
            levels["source"] = quality.source_name  # backward-compat alias
            levels["ticker"] = ticker
```

- [ ] **Step 4: Verify serve.py import works**

```bash
cd D:/2026/GEX
python -c "import serve; print('serve.py imports OK')"
```

Expected: `serve.py imports OK`

- [ ] **Step 5: Run full suite**

```bash
cd D:/2026/GEX
python -m pytest -v 2>&1 | tail -5
```

Expected: 116 passed, 0 failed

- [ ] **Step 6: Commit**

```bash
cd D:/2026/GEX
git add data.py serve.py
git commit -m "feat: wire SnapshotQuality into data.py and serve.py — OI freshness in API response"
```

---

## Task 4: Final Verification + Push

- [ ] **Step 1: Verify public API**

```bash
cd D:/2026/GEX
python -c "
from freshness import OISource, SnapshotQuality, assess_snapshot, market_hours, classify_source
q = assess_snapshot('yfinance')
print(f'source:    {q.source_name}')
print(f'oi_source: {q.oi_source}')
print(f'stale:     {q.is_intraday_stale}')
print(f'market:    {q.market_open}')
d = q.to_dict()
assert 'oi_source' in d
assert 'is_intraday_stale' in d
print('OK')
"
```

Expected: prints source/oi_source/stale/market fields and `OK`

- [ ] **Step 2: Run full test suite**

```bash
cd D:/2026/GEX
python -m pytest -v 2>&1 | tail -5
```

Expected: 116 passed, 0 failed

- [ ] **Step 3: Push**

```bash
cd D:/2026/GEX
git push
```

---

## Self-Review Checklist

**Spec coverage:**
- [x] OISource enum (SETTLED, ESTIMATED, LIVE) → Task 1, sources.py
- [x] classify_source("yfinance") → SETTLED → Task 2, test 6
- [x] classify_source("cboe") → SETTLED → Task 2, test 7
- [x] classify_source("tradier") → LIVE → Task 2, test 8
- [x] market_hours() open → Task 1, test 2
- [x] market_hours() closed weekend → Task 1, test 3
- [x] market_hours() closed pre-market → Task 1, test 4
- [x] market_hours() closed after-hours → Task 1, test 5
- [x] is_intraday_stale=True during market → Task 2, test 9
- [x] is_intraday_stale=False outside market → Task 2, test 10
- [x] fetch_chain returns 3-tuple → Task 3, data.py
- [x] fetch_chain_cboe returns 3-tuple → Task 3, data.py
- [x] serve.py unpacks 3-tuple → Task 3, serve.py
- [x] /api/gex includes oi_source + is_intraday_stale → Task 3, levels.update(quality.to_dict())
- [x] No network calls in freshness/ → all tests use synthetic datetimes
- [x] Existing 106 tests still pass → Task 3 Step 5, Task 4 Step 2
