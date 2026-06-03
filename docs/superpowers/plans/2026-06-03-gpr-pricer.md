# GPR Pricer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `pricer/` package — correct QuantLib-backed gamma dispatch (European/American/Black-76) with exact T computation — and integrate it into `greeks.py`.

**Architecture:** Four focused modules (`style.py`, `time.py`, `engines.py`, `__init__.py`) in a new `pricer/` package. `greeks.py:calculate_gex` gains a `ticker` param and delegates gamma computation to `pricer.compute_gamma()`. `levels.py:add_futures_conversion` gets a `DeprecationWarning`. Existing tests remain green throughout.

**Tech Stack:** Python 3.13, QuantLib (pip install QuantLib), pytest, zoneinfo (stdlib)

---

## File Map

| File | Action | Purpose |
|---|---|---|
| `pricer/__init__.py` | Create | Public API: compute_gamma, PricerStyle, detect_style, time_to_expiry, MIN_T_YEARS |
| `pricer/style.py` | Create | PricerStyle enum + detect_style(ticker) |
| `pricer/time.py` | Create | time_to_expiry(expiration_str) with 60-sec floor |
| `pricer/engines.py` | Create | ql_european_gamma, ql_american_gamma, ql_black76_gamma |
| `greeks.py` | Modify | Add ticker param; call compute_gamma; keep d1/d2 for vanna/charm |
| `levels.py` | Modify | Add DeprecationWarning to add_futures_conversion |
| `tests/test_pricer.py` | Create | 10 fixture-based tests |

---

## Pre-flight: Install QuantLib

```bash
pip install QuantLib
python -c "import QuantLib as ql; print(ql.__version__)"
```

Expected: prints a version string like `1.34` (no error).

---

## Task 1: PricerStyle + time_to_expiry (4 tests)

**Files:**
- Create: `D:/2026/GEX/pricer/__init__.py` (stub)
- Create: `D:/2026/GEX/pricer/style.py`
- Create: `D:/2026/GEX/pricer/time.py`
- Create: `D:/2026/GEX/tests/test_pricer.py`

- [ ] **Step 1: Write 4 failing tests**

Create `D:/2026/GEX/tests/test_pricer.py`:

```python
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import pytest
from datetime import date, timedelta

from pricer import PricerStyle, detect_style, time_to_expiry, MIN_T_YEARS
from pricer.engines import ql_european_gamma, ql_american_gamma, ql_black76_gamma
from pricer import compute_gamma


# --- Style detection ---

def test_detect_style_european():
    assert detect_style('SPX')  == PricerStyle.EUROPEAN
    assert detect_style('^SPX') == PricerStyle.EUROPEAN
    assert detect_style('NDX')  == PricerStyle.EUROPEAN
    assert detect_style('^NDX') == PricerStyle.EUROPEAN


def test_detect_style_american():
    assert detect_style('QQQ') == PricerStyle.AMERICAN
    assert detect_style('SPY') == PricerStyle.AMERICAN
    assert detect_style('IWM') == PricerStyle.AMERICAN


def test_detect_style_black76():
    assert detect_style('ES=F') == PricerStyle.BLACK76
    assert detect_style('NQ=F') == PricerStyle.BLACK76
    assert detect_style('MNQ=F') == PricerStyle.BLACK76


# --- Time to expiry ---

def test_time_to_expiry_far_expiry():
    # 30 days out → T ≈ 30/365.25 ≈ 0.082
    future = (date.today() + timedelta(days=30)).strftime('%Y-%m-%d')
    T = time_to_expiry(future)
    assert 0.07 < T < 0.10


def test_time_to_expiry_floor():
    # Already-expired date → T must be MIN_T_YEARS (60-sec floor, not negative)
    T = time_to_expiry('2000-01-01')
    assert T == pytest.approx(MIN_T_YEARS)


# --- Engine tests (added in Task 2) ---
# Placeholders so the file is valid Python even before Task 2 runs.
# DO NOT add engine tests here; append them in Task 2 Step 1.
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd D:/2026/GEX
python -m pytest tests/test_pricer.py::test_detect_style_european tests/test_pricer.py::test_detect_style_american tests/test_pricer.py::test_detect_style_black76 tests/test_pricer.py::test_time_to_expiry_far_expiry tests/test_pricer.py::test_time_to_expiry_floor -v
```

Expected: `ModuleNotFoundError: No module named 'pricer'`

- [ ] **Step 3: Create `pricer/style.py`**

```python
from enum import Enum


class PricerStyle(str, Enum):
    EUROPEAN = "european"  # SPX, NDX — cash-settled index options, no early exercise
    AMERICAN = "american"  # QQQ, SPY, IWM — ETF options, early exercise possible
    BLACK76  = "black76"   # ES=F, NQ=F, MNQ=F — futures options


def detect_style(ticker: str) -> PricerStyle:
    """
    Detect the correct pricing model for an options chain by ticker.

    Args:
        ticker: e.g. 'SPX', '^SPX', 'QQQ', 'ES=F'

    Returns:
        PricerStyle.EUROPEAN, AMERICAN, or BLACK76
    """
    t = ticker.upper().lstrip('^')
    if t in ('SPX', 'NDX', 'XSP'):
        return PricerStyle.EUROPEAN
    if t.endswith('=F'):
        return PricerStyle.BLACK76
    return PricerStyle.AMERICAN  # QQQ, SPY, IWM and any unknown ETF
```

- [ ] **Step 4: Create `pricer/time.py`**

```python
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
```

- [ ] **Step 5: Create stub `pricer/__init__.py`**

```python
from .style import PricerStyle, detect_style
from .time import time_to_expiry, MIN_T_YEARS

# compute_gamma and engines added in Task 2
__all__ = ['PricerStyle', 'detect_style', 'time_to_expiry', 'MIN_T_YEARS']
```

- [ ] **Step 6: Run 4 style + time tests**

```bash
cd D:/2026/GEX
python -m pytest tests/test_pricer.py::test_detect_style_european tests/test_pricer.py::test_detect_style_american tests/test_pricer.py::test_detect_style_black76 tests/test_pricer.py::test_time_to_expiry_far_expiry tests/test_pricer.py::test_time_to_expiry_floor -v
```

Expected: 5 passed

- [ ] **Step 7: Run full suite to check no regressions**

```bash
cd D:/2026/GEX
python -m pytest -v 2>&1 | tail -5
```

Expected: 95 tests pass (existing), 5 new pass = 100 total.

- [ ] **Step 8: Commit**

```bash
cd D:/2026/GEX
git add pricer/__init__.py pricer/style.py pricer/time.py tests/test_pricer.py
git commit -m "feat: add pricer package scaffold — PricerStyle, detect_style, time_to_expiry"
```

---

## Task 2: Engines + compute_gamma dispatch (5 tests)

**Files:**
- Create: `D:/2026/GEX/pricer/engines.py`
- Modify: `D:/2026/GEX/pricer/__init__.py` (add compute_gamma)
- Modify: `D:/2026/GEX/tests/test_pricer.py` (append 5 engine tests)

- [ ] **Step 1: Append 5 engine tests to `tests/test_pricer.py`**

Add these functions after the existing tests (remove the placeholder comment from Task 1):

```python
# --- Engine tests ---

def test_european_gamma_known_value():
    # ATM, T=0.25yr, σ=0.20, r=0.05, S=K=100
    # d1 = (ln(1) + (0.05+0.02)*0.25) / (0.20*0.5) = 0.175
    # gamma = N'(0.175) / (100*0.20*0.5) ≈ 0.0393
    g = ql_european_gamma(100.0, 100.0, 0.25, 0.05, 0.20, 'call')
    assert g == pytest.approx(0.0393, abs=0.002)


def test_american_gamma_gte_european_put():
    # ATM put, 30-day, σ=0.20, r=0.05: American put >= European put.
    # Puts always have nonzero early exercise premium (deep ITM put intrinsic
    # exceeds PV of waiting). For calls without dividends, American = European,
    # so this test uses put to guarantee the inequality.
    S, K, T, r, sigma = 100.0, 100.0, 30 / 365.25, 0.05, 0.20
    g_eu = ql_european_gamma(S, K, T, r, sigma, 'put')
    g_am = ql_american_gamma(S, K, T, r, sigma, 'put')
    assert g_am >= g_eu


def test_black76_gamma_matches_formula():
    # Black-76 ATM gamma: N'(d1) / (F * sigma * sqrt(T))
    # For ATM: d1 = 0.5 * sigma * sqrt(T)
    F, K, T, r, sigma = 100.0, 100.0, 0.25, 0.05, 0.20
    d1 = 0.5 * sigma * math.sqrt(T)
    expected = math.exp(-0.5 * d1 ** 2) / math.sqrt(2 * math.pi) / (F * sigma * math.sqrt(T))
    g = ql_black76_gamma(F, K, T, r, sigma, 'call')
    assert g == pytest.approx(expected, rel=0.01)


def test_compute_gamma_dispatch_returns_positive():
    # Smoke test: compute_gamma returns a positive float for each style
    for style in PricerStyle:
        g = compute_gamma(100.0, 100.0, 0.25, 0.05, 0.20, 'call', style)
        assert g > 0, f"Expected positive gamma for style={style}"


def test_compute_gamma_unknown_style_raises():
    with pytest.raises((ValueError, AttributeError)):
        compute_gamma(100.0, 100.0, 0.25, 0.05, 0.20, 'call', 'invalid_style')
```

- [ ] **Step 2: Run engine tests to verify they fail**

```bash
cd D:/2026/GEX
python -m pytest tests/test_pricer.py -k "engine or european or american or black76 or dispatch or unknown" -v
```

Expected: `ImportError: cannot import name 'ql_european_gamma' from 'pricer.engines'`

- [ ] **Step 3: Create `pricer/engines.py`**

```python
from __future__ import annotations
import math
import QuantLib as ql
from .time import MIN_T_YEARS


def _ql_date_from_T(T: float) -> ql.Date:
    """Convert T in years to a QuantLib Date offset from today."""
    days = max(int(T * 365.25), 1)
    return ql.Date.todaysDate() + days


def _bsm_process(S: float, r: float, sigma: float) -> ql.BlackScholesMertonProcess:
    """Build a Black-Scholes-Merton process with zero dividend yield."""
    spot   = ql.QuoteHandle(ql.SimpleQuote(S))
    r_ts   = ql.YieldTermStructureHandle(
        ql.FlatForward(0, ql.NullCalendar(),
                       ql.QuoteHandle(ql.SimpleQuote(r)),
                       ql.Actual365Fixed()))
    div_ts = ql.YieldTermStructureHandle(
        ql.FlatForward(0, ql.NullCalendar(),
                       ql.QuoteHandle(ql.SimpleQuote(0.0)),
                       ql.Actual365Fixed()))
    vol_ts = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(0, ql.NullCalendar(),
                            ql.QuoteHandle(ql.SimpleQuote(sigma)),
                            ql.Actual365Fixed()))
    return ql.BlackScholesMertonProcess(spot, div_ts, r_ts, vol_ts)


def ql_european_gamma(S: float, K: float, T: float, r: float,
                      sigma: float, option_type: str) -> float:
    """
    European option gamma via QuantLib AnalyticEuropeanEngine.

    Use for SPX, NDX — cash-settled index options with no early exercise.
    """
    ql.Settings.instance().evaluationDate = ql.Date.todaysDate()
    opt_type = ql.Option.Call if option_type == 'call' else ql.Option.Put
    payoff   = ql.PlainVanillaPayoff(opt_type, K)
    exercise = ql.EuropeanExercise(_ql_date_from_T(T))
    process  = _bsm_process(S, r, sigma)
    option   = ql.VanillaOption(payoff, exercise)
    option.setPricingEngine(ql.AnalyticEuropeanEngine(process))
    return option.gamma()


def ql_american_gamma(S: float, K: float, T: float, r: float,
                      sigma: float, option_type: str) -> float:
    """
    American option gamma via Barone-Adesi-Whaley approximation.

    Use for QQQ, SPY, IWM — ETF options where early exercise is possible.
    Gamma computed via central-difference numerical differentiation of BAW price:
        gamma ≈ (P(S+ε) + P(S-ε) - 2·P(S)) / ε²  where ε = S × 0.001
    """
    ql.Settings.instance().evaluationDate = ql.Date.todaysDate()
    opt_type = ql.Option.Call if option_type == 'call' else ql.Option.Put
    maturity = _ql_date_from_T(T)
    today    = ql.Date.todaysDate()

    def _price(s: float) -> float:
        process = _bsm_process(s, r, sigma)
        option  = ql.VanillaOption(
            ql.PlainVanillaPayoff(opt_type, K),
            ql.AmericanExercise(today, maturity),
        )
        option.setPricingEngine(ql.BaroneAdesiWhaleyApproximationEngine(process))
        return option.NPV()

    eps = S * 0.001
    return (_price(S + eps) + _price(S - eps) - 2.0 * _price(S)) / (eps ** 2)


def ql_black76_gamma(S: float, K: float, T: float, r: float,
                     sigma: float, option_type: str) -> float:
    """
    Black-76 gamma for futures options (closed-form).

    S is the futures price (forward). r is unused but kept for API consistency.
    Formula: N'(d1) / (F * sigma * sqrt(T))
    where d1 = (ln(F/K) + 0.5 * sigma^2 * T) / (sigma * sqrt(T))
    """
    d1      = (math.log(S / K) + 0.5 * sigma ** 2 * T) / (sigma * math.sqrt(T))
    npdf_d1 = math.exp(-0.5 * d1 ** 2) / math.sqrt(2 * math.pi)
    return npdf_d1 / (S * sigma * math.sqrt(T))
```

- [ ] **Step 4: Update `pricer/__init__.py`** (full version)

```python
from .style import PricerStyle, detect_style
from .time import time_to_expiry, MIN_T_YEARS
from .engines import ql_european_gamma, ql_american_gamma, ql_black76_gamma


def compute_gamma(S: float, K: float, T: float, r: float, sigma: float,
                  option_type: str, style: PricerStyle) -> float:
    """
    Dispatch to the correct pricer engine by PricerStyle.

    Args:
        S:           spot price (or futures price for BLACK76)
        K:           strike
        T:           time to expiry in years (use time_to_expiry() to compute)
        r:           risk-free rate
        sigma:       implied volatility
        option_type: 'call' | 'put'
        style:       PricerStyle.EUROPEAN | AMERICAN | BLACK76

    Returns:
        gamma (float, always positive)

    Raises:
        ValueError: if style is unrecognised
    """
    if style == PricerStyle.EUROPEAN:
        return ql_european_gamma(S, K, T, r, sigma, option_type)
    if style == PricerStyle.AMERICAN:
        return ql_american_gamma(S, K, T, r, sigma, option_type)
    if style == PricerStyle.BLACK76:
        return ql_black76_gamma(S, K, T, r, sigma, option_type)
    raise ValueError(f"Unknown PricerStyle: {style!r}")


__all__ = [
    'compute_gamma', 'PricerStyle', 'detect_style',
    'time_to_expiry', 'MIN_T_YEARS',
    'ql_european_gamma', 'ql_american_gamma', 'ql_black76_gamma',
]
```

- [ ] **Step 5: Run all 10 pricer tests**

```bash
cd D:/2026/GEX
python -m pytest tests/test_pricer.py -v
```

Expected: 10 passed

- [ ] **Step 6: Run full suite to check no regressions**

```bash
cd D:/2026/GEX
python -m pytest -v 2>&1 | tail -5
```

Expected: 105 total (95 + 10), all pass.

- [ ] **Step 7: Commit**

```bash
cd D:/2026/GEX
git add pricer/engines.py pricer/__init__.py tests/test_pricer.py
git commit -m "feat: add pricer engines — QuantLib European/American/Black76 gamma dispatch"
```

---

## Task 3: greeks.py integration + levels.py deprecation (1 test)

**Files:**
- Modify: `D:/2026/GEX/greeks.py`
- Modify: `D:/2026/GEX/levels.py`
- Modify: `D:/2026/GEX/tests/test_pricer.py` (append test 10)

- [ ] **Step 1: Append test 10 to `tests/test_pricer.py`**

```python
def test_additive_basis_consistency():
    """level_projection additive basis round-trips correctly; no multiplicative fallback."""
    from level_projection import measure_basis, project
    from level_projection.models import Level
    basis = measure_basis(future_price=5310.0, index_value=5300.0)
    assert basis == pytest.approx(10.0)
    levels = [Level(strike=5300.0, gex=100.0, tier=1, strength=1.0, label='pin')]
    fut_levels = project(levels, basis)
    assert fut_levels[0].fut_price == pytest.approx(5310.0)
    assert fut_levels[0].basis_pts == pytest.approx(10.0)
```

Run to confirm it passes (basis.py already exists):

```bash
cd D:/2026/GEX
python -m pytest tests/test_pricer.py::test_additive_basis_consistency -v
```

Expected: PASS (level_projection already correct).

- [ ] **Step 2: Run test_greeks.py to get baseline**

```bash
cd D:/2026/GEX
python -m pytest tests/test_greeks.py -v
```

Expected: 9 passed. Note: `test_calculate_gex_stores_d1_d2` verifies d1/d2 columns exist — this test must still pass after our change.

- [ ] **Step 3: Modify `greeks.py`**

Replace the entire `calculate_gex` function. The other three functions (`calculate_vanna`, `calculate_charm`, `calculate_all_greeks`) are unchanged.

New `greeks.py` (full file):

```python
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime

from pricer import compute_gamma, detect_style, time_to_expiry


def calculate_gex(df: pd.DataFrame, spot: float, r: float = 0.05,
                  ticker: str = 'SPY') -> pd.DataFrame:
    """
    Compute GEX (gamma exposure) for each option in the chain.

    Uses the correct pricer per ticker:
      - European BS  for SPX, NDX (cash-settled, no early exercise)
      - American BAW for QQQ, SPY, IWM (ETF options, early exercise possible)
      - Black-76     for ES=F, NQ=F (futures options)

    Args:
        df:     options chain DataFrame with columns:
                  strike, type ('call'|'put'), openInterest,
                  impliedVolatility, expiration ('YYYY-MM-DD')
        spot:   current underlying price
        r:      risk-free rate (default 0.05)
        ticker: underlying ticker — used to select pricer style (default 'SPY')

    Returns:
        df with added columns: T, iv, d1, d2, gamma, gex
    """
    style = detect_style(ticker)
    df    = df.copy()

    # Exact T in years (60-second floor; replaces old clip(lower=0.001))
    df['T']  = df['expiration'].apply(time_to_expiry)
    df['iv'] = df['impliedVolatility'].replace(0, np.nan).fillna(0.20)

    # d1/d2 retained for vanna/charm downstream (calculate_vanna, calculate_charm)
    df['d1'] = (
        np.log(spot / df['strike']) + (r + 0.5 * df['iv'] ** 2) * df['T']
    ) / (df['iv'] * np.sqrt(df['T']))
    df['d2'] = df['d1'] - df['iv'] * np.sqrt(df['T'])

    # Gamma via correct pricer per style
    df['gamma'] = df.apply(
        lambda row: compute_gamma(
            spot, row['strike'], row['T'], r, row['iv'], row['type'], style
        ),
        axis=1,
    )

    # GEX: gamma * OI * 100 (multiplier) * spot^2 * 0.01 (per-1%-move convention)
    df['gex'] = df['gamma'] * df['openInterest'] * 100 * spot ** 2 * 0.01
    df.loc[df['type'] == 'put', 'gex'] *= -1
    return df


def calculate_vanna(df: pd.DataFrame, spot: float) -> pd.DataFrame:
    # Vanna = d(delta)/d(iv) = -norm.pdf(d1) * d2 / iv
    # Measures how dealer delta changes when IV moves (vol-driven hedging flows)
    df['vanna'] = -norm.pdf(df['d1']) * df['d2'] / df['iv']
    df['vex'] = df['vanna'] * df['openInterest'] * 100 * spot * 0.01
    df.loc[df['type'] == 'put', 'vex'] *= -1
    return df


def calculate_charm(df: pd.DataFrame, spot: float, r: float = 0.05) -> pd.DataFrame:
    # Charm = d(delta)/d(t) = time-decay rate of delta (theta of delta)
    # Measures how dealer delta changes as time passes (time-driven hedging flows)
    df['charm'] = -norm.pdf(df['d1']) * (
        2 * r * df['T'] - df['d2'] * df['iv'] * np.sqrt(df['T'])
    ) / (2 * df['T'] * df['iv'] * np.sqrt(df['T']))
    df['chex'] = df['charm'] * df['openInterest'] * 100
    df.loc[df['type'] == 'put', 'chex'] *= -1
    return df


def calculate_all_greeks(df: pd.DataFrame, spot: float, r: float = 0.05) -> pd.DataFrame:
    df = calculate_gex(df, spot, r)
    df = calculate_vanna(df, spot)
    df = calculate_charm(df, spot, r)
    return df
```

- [ ] **Step 4: Run test_greeks.py to verify integration**

```bash
cd D:/2026/GEX
python -m pytest tests/test_greeks.py -v
```

Expected: 9 passed. All existing tests pass — d1/d2 columns present, puts negative, etc.

Note: `calculate_all_greeks` calls `calculate_gex` without `ticker`, so it defaults to `'SPY'` (AMERICAN) — correct for the test fixture.

- [ ] **Step 5: Add DeprecationWarning to `levels.py:add_futures_conversion`**

In `D:/2026/GEX/levels.py`, add `import warnings` at the top (after the existing imports), then replace the function signature line:

Current (line 90):
```python
def add_futures_conversion(levels: dict, ticker: str, spot: float) -> dict:
```

New (full function with warning prepended):
```python
def add_futures_conversion(levels: dict, ticker: str, spot: float) -> dict:
    import warnings
    warnings.warn(
        "add_futures_conversion() uses a multiplicative basis approximation. "
        "Use level_projection.measure_basis() + level_projection.project() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
```

Only add the `import warnings` and the `warnings.warn(...)` block at the top of the function body. Do NOT change any other code in the function.

- [ ] **Step 6: Run all 10 pricer tests + full suite**

```bash
cd D:/2026/GEX
python -m pytest tests/test_pricer.py -v
```

Expected: 10 passed

```bash
cd D:/2026/GEX
python -m pytest -v 2>&1 | tail -5
```

Expected: 104 total (95 + 10 pricer - 1 overlap from test_additive_basis_consistency which was already passing), all pass. Actual count = 105 if all tests are distinct.

- [ ] **Step 7: Commit**

```bash
cd D:/2026/GEX
git add greeks.py levels.py tests/test_pricer.py
git commit -m "feat: integrate QuantLib pricer into greeks.py; deprecate multiplicative basis"
```

---

## Task 4: Final Verification + Push

- [ ] **Step 1: Verify public API**

```bash
cd D:/2026/GEX
python -c "
from pricer import compute_gamma, PricerStyle, detect_style, time_to_expiry, MIN_T_YEARS
g = compute_gamma(5300.0, 5300.0, 0.25, 0.05, 0.20, 'call', PricerStyle.AMERICAN)
print(f'QQQ ATM gamma: {g:.6f}')
print(f'detect_style QQQ: {detect_style(\"QQQ\")}')
print(f'detect_style SPX: {detect_style(\"SPX\")}')
print(f'detect_style ES=F: {detect_style(\"ES=F\")}')
print('OK')
"
```

Expected output:
```
QQQ ATM gamma: 0.00XXXX
detect_style QQQ: american
detect_style SPX: european
detect_style ES=F: black76
OK
```

- [ ] **Step 2: Run full test suite**

```bash
cd D:/2026/GEX
python -m pytest -v 2>&1 | tail -10
```

Expected: ≥105 passed, 0 failed.

- [ ] **Step 3: Push**

```bash
cd D:/2026/GEX
git push
```

---

## Self-Review Checklist

**Spec coverage:**
- [x] PricerStyle enum (EUROPEAN, AMERICAN, BLACK76) → Task 1, style.py
- [x] detect_style('SPX') → EUROPEAN → Task 1, test 1
- [x] detect_style('QQQ') → AMERICAN → Task 1, test 2
- [x] detect_style('ES=F') → BLACK76 → Task 1, test 3
- [x] time_to_expiry far expiry → Task 1, test 4
- [x] time_to_expiry floor (60-sec) → Task 1, test 5
- [x] ql_european_gamma known value → Task 2, test 6
- [x] American put gamma >= European put gamma → Task 2, test 7
- [x] Black-76 gamma matches closed-form → Task 2, test 8
- [x] compute_gamma dispatch positive for all styles → Task 2, test 9
- [x] additive basis consistency → Task 3, test 10
- [x] greeks.py ticker param + compute_gamma integration → Task 3
- [x] d1/d2 retained for vanna/charm → Task 3 (greeks.py code shows both)
- [x] add_futures_conversion DeprecationWarning → Task 3
- [x] No network calls in pricer/ → all engines use QuantLib only
- [x] Backward-compatible (ticker defaults to 'SPY') → existing callers unaffected
