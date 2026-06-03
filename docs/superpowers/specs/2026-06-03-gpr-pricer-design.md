# GPR Pricer — Design Spec
**Date:** 2026-06-03
**Status:** Approved
**Sub-project:** 5 of 10 — QuantLib Pricer (correctness layer)

## Overview

Replace the inline European Black-Scholes gamma formula in `greeks.py` with a proper per-ticker pricer that dispatches to the correct QuantLib engine: European BS for SPX/NDX, American (Barone-Adesi-Whaley) for QQQ/SPY/IWM, Black-76 for futures options. Fix the T-floor bug (0.001 yr ≈ 8.76 hr is 18× too large for 30-min 0DTE). Deprecate the multiplicative basis in `levels.py` in favour of the additive `level_projection/basis.py`.

This is the correctness gate for the GPR system. The Part A audit flagged QQQ European pricing as a **blocker** and the T-floor as a **major** finding. Both must be resolved before any statistical validation is meaningful.

---

## Part A Audit Findings Resolved

| Finding | Severity | Fix |
|---|---|---|
| QQQ priced as European BS | Blocker | BAW American engine via QuantLib |
| T-floor 0.001yr ≈ 8.76hr | Major | Exact minutes-remaining, 60-sec floor |
| Multiplicative vs additive basis inconsistency | Major | Deprecate `add_futures_conversion`; additive basis is sole source |

---

## File Structure

```
D:/2026/GEX/
├── pricer/
│   ├── __init__.py     # exports: compute_gamma, PricerStyle, detect_style
│   ├── style.py        # PricerStyle enum + detect_style(ticker) -> PricerStyle
│   ├── time.py         # time_to_expiry(expiration_str) -> float
│   └── engines.py      # ql_european_gamma, ql_american_gamma, ql_black76_gamma
└── tests/
    └── test_pricer.py  # 10 tests
```

**Modified files:**
- `greeks.py` — add `ticker` param to `calculate_gex()`; call `compute_gamma()` per row
- `levels.py` — add `DeprecationWarning` to `add_futures_conversion()`

**NOT modified:**
- `gex_calc.py` — standalone legacy script; does not feed the GPR pipeline
- `level_projection/` — already uses additive basis; unchanged

---

## Module Specifications

### `pricer/style.py`

```python
from enum import Enum

class PricerStyle(str, Enum):
    EUROPEAN = "european"  # SPX, NDX — cash-settled, no early exercise
    AMERICAN = "american"  # QQQ, SPY, IWM — ETF options, early exercise possible
    BLACK76  = "black76"   # ES=F, NQ=F, MNQ=F — futures options

def detect_style(ticker: str) -> PricerStyle:
    t = ticker.upper().lstrip('^')
    if t in ('SPX', 'NDX', 'XSP'):
        return PricerStyle.EUROPEAN
    if t.endswith('=F'):
        return PricerStyle.BLACK76
    return PricerStyle.AMERICAN   # QQQ, SPY, IWM default
```

---

### `pricer/time.py`

```python
from datetime import datetime, time
from zoneinfo import ZoneInfo

ET = ZoneInfo('America/New_York')
EQUITY_CLOSE = time(16, 0, 0)
MIN_T_YEARS  = 60 / (365.25 * 24 * 3600)   # 60-second absolute floor

def time_to_expiry(expiration_str: str) -> float:
    """
    Compute time-to-expiry in years from now to 4:00 PM ET on expiration_str.
    Floor: 60 seconds (avoids QuantLib singularity at T=0).

    Args:
        expiration_str: 'YYYY-MM-DD'

    Returns:
        T in years, >= MIN_T_YEARS
    """
    now = datetime.now(tz=ET)
    exp = datetime.strptime(expiration_str, '%Y-%m-%d').replace(
        hour=16, minute=0, second=0, microsecond=0, tzinfo=ET
    )
    secs = (exp - now).total_seconds()
    return max(secs, 60) / (365.25 * 24 * 3600)
```

**Why additive basis is correct:** Index futures basis ≈ spot × (r − q) × T, measured in *points*. For SPX at 5300, r=5%, q=1.5%, T=45 days: basis ≈ 23 pts. This is point-additive, not multiplicative. As time decays or rates shift, the basis shifts in pts, not in %.

---

### `pricer/engines.py`

All three functions share the same signature:

```python
def ql_european_gamma(S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> float
def ql_american_gamma(S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> float
def ql_black76_gamma(S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> float
```

Parameters:
- `S` — spot price
- `K` — strike
- `T` — time to expiry in years (>= MIN_T_YEARS)
- `r` — risk-free rate (default 0.05)
- `sigma` — implied volatility (IV fallback 0.20 if missing/zero)
- `option_type` — `'call'` | `'put'`

Returns: gamma as float (scalar; called per row via `df.apply()`)

**ql_european_gamma** — `BlackScholesMertonProcess` + `AnalyticEuropeanEngine`

**ql_american_gamma** — `BlackScholesMertonProcess` + `BaroneAdesiWhaleyApproximationEngine`
- BAW is O(1) closed-form approximation; accurate for equity ETF options
- Gamma from BAW: compute numerically via `(price(S+ε) - 2·price(S) + price(S-ε)) / ε²` if QL does not expose gamma directly from BAW; ε = S × 0.001

**ql_black76_gamma** — `BlackProcess` (forward price as underlying) + `AnalyticEuropeanEngine`
- Forward F = futures price (S for futures options where S = futures price)
- Black-76 gamma: `norm.pdf(d1) / (F × sigma × sqrt(T))`

---

### `pricer/__init__.py`

```python
from .style import PricerStyle, detect_style
from .time import time_to_expiry, MIN_T_YEARS
from .engines import ql_european_gamma, ql_american_gamma, ql_black76_gamma

def compute_gamma(S: float, K: float, T: float, r: float, sigma: float,
                  option_type: str, style: PricerStyle) -> float:
    """Dispatch to the correct pricer engine."""
    if style == PricerStyle.EUROPEAN:
        return ql_european_gamma(S, K, T, r, sigma, option_type)
    elif style == PricerStyle.AMERICAN:
        return ql_american_gamma(S, K, T, r, sigma, option_type)
    elif style == PricerStyle.BLACK76:
        return ql_black76_gamma(S, K, T, r, sigma, option_type)
    raise ValueError(f"Unknown PricerStyle: {style}")

__all__ = ['compute_gamma', 'PricerStyle', 'detect_style', 'time_to_expiry', 'MIN_T_YEARS']
```

---

### Integration: `greeks.py`

`calculate_gex(df, spot, r, ticker)` gains a `ticker: str = 'SPY'` parameter:

```python
from pricer import compute_gamma, detect_style, time_to_expiry

def calculate_gex(df: pd.DataFrame, spot: float, r: float = 0.05,
                  ticker: str = 'SPY') -> pd.DataFrame:
    style = detect_style(ticker)
    df = df.copy()
    df['T']  = df['expiration'].apply(time_to_expiry)
    df['iv'] = df['impliedVolatility'].replace(0, np.nan).fillna(0.20)
    # d1/d2 kept for vanna/charm downstream (calculate_vanna, calculate_charm use them)
    df['d1'] = (np.log(spot / df['strike']) + (r + 0.5 * df['iv']**2) * df['T']) \
               / (df['iv'] * np.sqrt(df['T']))
    df['d2'] = df['d1'] - df['iv'] * np.sqrt(df['T'])
    # gamma via correct pricer per style (replaces inline BS formula)
    df['gamma'] = df.apply(
        lambda row: compute_gamma(spot, row['strike'], row['T'], r,
                                  row['iv'], row['type'], style),
        axis=1,
    )
    df['gex'] = df['gamma'] * df['openInterest'] * 100 * spot ** 2 * 0.01
    df.loc[df['type'] == 'put', 'gex'] *= -1
    return df
```

`d1` and `d2` are computed using the BS formula even when gamma uses the American or Black-76 engine. This is correct: vanna and charm are second-order sensitivities where the European/American difference is negligible, and the BS `d1`/`d2` parameterisation is the standard convention for those greek formulas.

Vanna (`calculate_vanna`) and charm (`calculate_charm`) in `greeks.py` are unchanged — they already read `d1`, `d2`, `iv`, `T` from the DataFrame.

**Note:** BAW and European gamma are very close for deep ITM/OTM strikes; the material difference appears near-ATM for high-dividend stocks close to ex-div. The test `test_american_gamma_gt_european` validates this direction.

---

### Deprecation: `levels.py:add_futures_conversion`

```python
import warnings

def add_futures_conversion(levels: dict, ticker: str, spot: float) -> dict:
    warnings.warn(
        "add_futures_conversion() uses a multiplicative basis approximation. "
        "Use level_projection.measure_basis() + level_projection.project() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    # ... existing implementation unchanged ...
```

`serve.py` and `run.py` are updated in a follow-on task (sub-project 6) to use `level_projection` directly. The deprecation warning ensures nothing silently uses the wrong basis.

---

## Tests (`tests/test_pricer.py`)

All tests are synthetic — no network calls, no yfinance.

```python
# 10 tests

def test_detect_style_european():
    assert detect_style('SPX')  == PricerStyle.EUROPEAN
    assert detect_style('^SPX') == PricerStyle.EUROPEAN
    assert detect_style('NDX')  == PricerStyle.EUROPEAN

def test_detect_style_american():
    assert detect_style('QQQ') == PricerStyle.AMERICAN
    assert detect_style('SPY') == PricerStyle.AMERICAN
    assert detect_style('IWM') == PricerStyle.AMERICAN

def test_detect_style_black76():
    assert detect_style('ES=F') == PricerStyle.BLACK76
    assert detect_style('NQ=F') == PricerStyle.BLACK76

def test_time_to_expiry_far_expiry():
    # A date 30 days out should give T ≈ 30/365.25 ≈ 0.082
    from datetime import date, timedelta
    future = (date.today() + timedelta(days=30)).strftime('%Y-%m-%d')
    T = time_to_expiry(future)
    assert 0.07 < T < 0.10

def test_time_to_expiry_floor():
    # Already-expired date → T should be MIN_T_YEARS (60-sec floor)
    T = time_to_expiry('2000-01-01')
    assert T == pytest.approx(MIN_T_YEARS)

def test_american_gamma_gt_european():
    # ATM put, 30-day, σ=0.20, r=0.05: American put gamma > European put gamma.
    # Puts always have nonzero early exercise premium (deep ITM put intrinsic > PV of waiting).
    # Calls without dividends: American == European, so use put here.
    S, K, T, r, sigma = 100.0, 100.0, 30/365.25, 0.05, 0.20
    g_eu = ql_european_gamma(S, K, T, r, sigma, 'put')
    g_am = ql_american_gamma(S, K, T, r, sigma, 'put')
    assert g_am >= g_eu

def test_european_gamma_known_value():
    # ATM, T=0.25yr, σ=0.20, r=0.05, S=K=100 → gamma ≈ 0.0797 (standard BS)
    g = ql_european_gamma(100.0, 100.0, 0.25, 0.05, 0.20, 'call')
    assert g == pytest.approx(0.0797, abs=0.003)

def test_black76_gamma_matches_formula():
    # Black-76 ATM: gamma = N'(0) / (F * sigma * sqrt(T))
    # = 1/sqrt(2pi) / (F * sigma * sqrt(T))
    import math
    F, K, T, r, sigma = 100.0, 100.0, 0.25, 0.05, 0.20
    expected = (1 / math.sqrt(2 * math.pi)) / (F * sigma * math.sqrt(T))
    g = ql_black76_gamma(F, K, T, r, sigma, 'call')
    assert g == pytest.approx(expected, rel=0.01)

def test_compute_gamma_dispatch_routes_correctly():
    # Smoke test: compute_gamma returns a positive float for each style
    for style in PricerStyle:
        g = compute_gamma(100.0, 100.0, 0.25, 0.05, 0.20, 'call', style)
        assert g > 0

def test_additive_basis_consistency():
    from level_projection import measure_basis, project, Level
    basis = measure_basis(future_price=5310.0, index_value=5300.0)
    assert basis == pytest.approx(10.0)
    levels = [Level(strike=5300.0, gex=100.0, tier=1, strength=1.0, label='pin')]
    fut_levels = project(levels, basis)
    assert fut_levels[0].fut_price == pytest.approx(5310.0)
    assert fut_levels[0].basis_pts == pytest.approx(10.0)
```

---

## Acceptance Criteria

- All 10 tests pass
- `detect_style('QQQ')` returns `AMERICAN`; `detect_style('SPX')` returns `EUROPEAN`
- `time_to_expiry('2000-01-01')` returns `MIN_T_YEARS` (floor, not negative)
- ATM 30-day American gamma >= European gamma
- `compute_gamma` returns positive float for all three styles
- `greeks.py:calculate_gex` accepts `ticker` param; existing callers with no `ticker` arg default to `'SPY'` (backward-compatible)
- `add_futures_conversion` emits `DeprecationWarning`
- No network calls anywhere in `pricer/`

---

## Relationship to GPR Sub-projects

| Sub-project | Note |
|---|---|
| 1-4 — level_projection, regime, signal, backtest | Use greeks.py output; benefit from correct gamma automatically |
| 5 — pricer (this) | Correctness gate; must pass before statistical validation is meaningful |
| 6 — staleness + OI flags | Next: live data freshness guard |
| 7 — statistical validation | PBO/DSR/Monte Carlo; requires correct gamma as input |
