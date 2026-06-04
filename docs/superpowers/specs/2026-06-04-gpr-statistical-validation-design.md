# GPR Statistical Validation — Design Spec
**Date:** 2026-06-04
**Status:** Approved
**Sub-project:** 7 of 10 — Statistical Validation (Monte Carlo + Kupiec VaR)

## Overview

The existing backtest harness (`backtest/stats.py`) gates live execution via `signal_only: bool` — it passes only when the positive-regime IS edge is real and OOS PnL is positive. This is a necessary but not sufficient condition. It does not test whether the observed Sharpe is better than random trade ordering, nor whether the loss distribution is well-calibrated.

This sub-project adds `backtest/validation.py` — two statistical tests that run on `list[TradeRecord]` and produce a `ValidationResult`. Both tests must pass for `signal_only = False`.

## Part A Audit Finding Resolved

| Finding | Severity | Fix |
|---|---|---|
| No statistical validation of edge — IS/OOS split alone is insufficient against overfitting | Blocker | Monte Carlo permutation p-value + Kupiec POF gate in `signal_only` |

## Decisions

| Question | Decision |
|---|---|
| PBO (CSCV-based)? | Skipped — too few trades per ticker for meaningful CSCV |
| DSR (Deflated Sharpe Ratio)? | Skipped — no strategy trial count to condition on |
| Multiple-testing correction? | Monte Carlo permutation test serves this role |
| VaR confidence level | 95% |
| Architecture | Single module `backtest/validation.py` inside existing package |
| Integration point | `validate()` called inside `compute_stats()`; result stored on `BacktestResult` |

---

## File Structure

```
D:/2026/GEX/
├── backtest/
│   ├── validation.py     # NEW — ValidationResult + monte_carlo_pvalue + kupiec_pof + validate
│   ├── models.py         # MODIFY — add validation: ValidationResult field to BacktestResult
│   ├── stats.py          # MODIFY — call validate(), fold into signal_only
│   └── __init__.py       # MODIFY — export ValidationResult, validate
└── tests/
    └── test_validation.py  # NEW — 7 synthetic tests
```

---

## Module Specification

### `backtest/validation.py`

Note: `validation.py` defines its own private `_sharpe()` (identical formula to `stats.py`). The two modules are intentionally independent — `validation.py` has no imports from `stats.py`, keeping it testable in isolation. scipy is already a project dependency (`greeks.py` uses `scipy.stats.norm`), so `_chi2_sf` will always use the scipy path at runtime; the `math.erfc` fallback is a belt-and-suspenders guard.

```python
from __future__ import annotations
import math
import random
import statistics
from dataclasses import dataclass
from .models import TradeRecord


@dataclass
class ValidationResult:
    mc_pvalue:      float   # fraction of shuffles with Sharpe >= observed; lower is better
    mc_passed:      bool    # True if mc_pvalue < 0.05
    kupiec_pvalue:  float   # chi-squared p-value; higher = better-calibrated VaR
    kupiec_passed:  bool    # True if kupiec_pvalue >= 0.05 (or n < 10 → auto-pass)
    n_permutations: int     # number of MC shuffles run
    passed:         bool    # True if mc_passed AND kupiec_passed


def _sharpe(pnls: list[float]) -> float:
    """Annualised-style Sharpe (mean/stdev). Returns 0.0 if < 2 trades."""
    if len(pnls) < 2:
        return 0.0
    mean = statistics.mean(pnls)
    std  = max(statistics.stdev(pnls), 1e-9)
    return mean / std


def monte_carlo_pvalue(
    pnls: list[float],
    n_permutations: int = 1000,
    seed: int = 42,
) -> float:
    """
    Permutation test: what fraction of random orderings of the same PnLs
    achieve a Sharpe >= the observed Sharpe?

    A low p-value means the observed Sharpe is better than chance —
    the strategy's timing adds value beyond the PnL magnitudes alone.

    Args:
        pnls:           per-trade PnL list (from TradeRecord.pnl_pts)
        n_permutations: number of shuffles (default 1000)
        seed:           RNG seed for reproducibility

    Returns:
        float in [0, 1] — p-value; < 0.05 passes
    """
    if len(pnls) < 2:
        return 1.0   # inconclusive → conservative fail
    observed = _sharpe(pnls)
    rng = random.Random(seed)
    buf = list(pnls)
    beats = 0
    for _ in range(n_permutations):
        rng.shuffle(buf)
        if _sharpe(buf) >= observed:
            beats += 1
    return beats / n_permutations


def kupiec_pof(pnls: list[float], confidence: float = 0.95) -> float:
    """
    Kupiec Proportion of Failures test at the stated confidence level.

    Tests whether the observed exceedance rate (trades below VaR) is
    consistent with the expected rate (1 - confidence).

    Returns a p-value from a chi-squared(1) likelihood-ratio test.
    p >= 0.05 means we cannot reject the VaR model → passes.

    Returns 1.0 (auto-pass) if n < 10 (inconclusive with too few trades).

    Args:
        pnls:       per-trade PnL list
        confidence: VaR confidence level (default 0.95)

    Returns:
        float in [0, 1] — p-value; >= 0.05 passes
    """
    n = len(pnls)
    if n < 10:
        return 1.0   # too few trades — inconclusive, auto-pass

    alpha = 1.0 - confidence                   # expected exceedance rate = 0.05
    sorted_pnls = sorted(pnls)
    var_threshold = sorted_pnls[max(0, int(n * alpha) - 1)]  # 5th-percentile PnL

    x = sum(1 for p in pnls if p < var_threshold)  # observed exceedances
    p_hat = x / n                                   # observed rate

    # Kupiec LR statistic: -2 * log(L0/L1)
    # L0 = alpha^x * (1-alpha)^(n-x)  [null: exceedance rate = alpha]
    # L1 = p_hat^x * (1-p_hat)^(n-x)  [alt: exceedance rate = p_hat]
    if x == 0:
        # Zero exceedances: LR based on null only
        lr = -2.0 * (n * math.log(1.0 - alpha))
    elif x == n:
        lr = -2.0 * (n * math.log(alpha))
    else:
        lr = -2.0 * (
            x * math.log(alpha / p_hat) +
            (n - x) * math.log((1.0 - alpha) / (1.0 - p_hat))
        )

    # Chi-squared(1) survival function (1 - CDF)
    # Using the regularised incomplete gamma: p = 1 - P(0.5, lr/2)
    p_value = _chi2_sf(lr, df=1)
    return p_value


def _chi2_sf(x: float, df: int) -> float:
    """
    Survival function of chi-squared distribution (1 - CDF).
    Uses scipy.stats if available; falls back to a pure-Python approximation.
    df=1 only.
    """
    try:
        from scipy.stats import chi2
        return float(chi2.sf(x, df))
    except ImportError:
        # Pure-Python regularised incomplete gamma for df=1 (chi2(1) = Gamma(0.5, 0.5))
        # P(chi2(1) > x) = erfc(sqrt(x/2))
        return math.erfc(math.sqrt(x / 2.0))


def validate(trades: list[TradeRecord], n_permutations: int = 1000) -> ValidationResult:
    """
    Run both statistical tests on a list of completed trades.

    Args:
        trades:         output of backtest.replay()
        n_permutations: MC shuffle count (default 1000)

    Returns:
        ValidationResult
    """
    pnls = [t.pnl_pts for t in trades]

    mc_p   = monte_carlo_pvalue(pnls, n_permutations=n_permutations)
    kup_p  = kupiec_pof(pnls)

    mc_passed  = mc_p  < 0.05
    kup_passed = kup_p >= 0.05

    return ValidationResult(
        mc_pvalue      = mc_p,
        mc_passed      = mc_passed,
        kupiec_pvalue  = kup_p,
        kupiec_passed  = kup_passed,
        n_permutations = n_permutations,
        passed         = mc_passed and kup_passed,
    )
```

---

### `backtest/models.py` (modified)

Add `validation` field to `BacktestResult`. It is `None` when `compute_stats` is called with `validate=False` (for compatibility with existing tests that don't need the full stats).

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .validation import ValidationResult


@dataclass
class TradeRecord:
    entry_bar:      int
    exit_bar:       int
    direction:      str
    entry_price:    float
    exit_price:     float
    pnl_pts:        float
    regime_at_entry: str
    level_tier:     int
    level_strength: float


@dataclass
class BacktestResult:
    total_trades:              int
    positive_regime_trades:    int
    negative_regime_trades:    int
    positive_regime_pnl:       float
    negative_regime_pnl:       float
    positive_regime_sharpe:    float
    negative_regime_sharpe:    float
    oos_positive_pnl:          float
    regime_split_significant:  bool
    signal_only:               bool
    validation:                 'ValidationResult | None' = field(default=None)
```

---

### `backtest/stats.py` (modified)

`compute_stats` gains a `run_validation: bool = True` parameter. When `True` (default), it calls `validate(trades)` and the result tightens the `signal_only` gate.

```python
# New gate logic:
from .validation import validate, ValidationResult

def compute_stats(trades, oos_start_pct=0.70, run_validation=True):
    ...
    vr = validate(trades) if run_validation else None
    validation_passed = vr.passed if vr is not None else True

    signal_only = not (regime_split and oos_pos_pnl > 0 and validation_passed)

    return BacktestResult(
        ...,
        signal_only = signal_only,
        validation  = vr,
    )
```

Existing tests pass `run_validation=False` or work unchanged because the new `validation` field has a default of `None`.

---

### `backtest/__init__.py` (modified)

```python
from .models import TradeRecord, BacktestResult
from .stats import compute_stats
from .report import format_report
from .replay import replay
from .validation import ValidationResult, validate

__all__ = [
    'TradeRecord', 'BacktestResult',
    'compute_stats', 'format_report', 'replay',
    'ValidationResult', 'validate',
]
```

---

## Tests (`tests/test_validation.py`)

All synthetic — no replay, no network calls.

```python
# Helpers
def _pos_pnls(n=50):
    """n consistently winning trades."""
    return [10.0] * n

def _random_pnls(n=50, seed=0):
    """alternating +1 / -1."""
    return [1.0 if i % 2 == 0 else -1.0 for i in range(n)]

def _calibrated_pnls(n=100):
    """100 trades; exactly 5 are below the 5th percentile (by construction)."""
    # 95 trades at +10, 5 trades at -100
    return [10.0] * 95 + [-100.0] * 5


# 1
def test_mc_pvalue_all_winners():
    # All identical PnLs → every shuffle has same Sharpe → 0/1000 beat observed
    pnls = _pos_pnls(50)
    p = monte_carlo_pvalue(pnls, n_permutations=1000, seed=42)
    assert p == pytest.approx(0.0)


# 2
def test_mc_pvalue_random_not_significant():
    # Alternating +1/-1 → p-value near 0.5 (not < 0.05)
    pnls = _random_pnls(50)
    p = monte_carlo_pvalue(pnls, n_permutations=1000, seed=42)
    assert p > 0.05


# 3
def test_kupiec_well_calibrated():
    # 5% exceedance rate matches 95% confidence → p >= 0.05
    pnls = _calibrated_pnls(100)
    p = kupiec_pof(pnls, confidence=0.95)
    assert p >= 0.05


# 4
def test_kupiec_too_few_trades_auto_pass():
    p = kupiec_pof([1.0, -2.0, 3.0], confidence=0.95)
    assert p == pytest.approx(1.0)


# 5
def test_validate_consistent_winners_passes():
    trades = [TradeRecord(entry_bar=i, exit_bar=i+1, direction='short',
                          entry_price=100.0, exit_price=90.0, pnl_pts=10.0,
                          regime_at_entry='positive', level_tier=1, level_strength=0.9)
              for i in range(0, 100, 2)]
    vr = validate(trades)
    assert vr.mc_passed   is True
    assert vr.passed      is True


# 6
def test_validate_random_pnls_fails():
    trades = [TradeRecord(entry_bar=i, exit_bar=i+1, direction='short',
                          entry_price=100.0, exit_price=99.0 if i%2==0 else 101.0,
                          pnl_pts=1.0 if i%2==0 else -1.0,
                          regime_at_entry='positive', level_tier=1, level_strength=0.9)
              for i in range(50)]
    vr = validate(trades)
    assert vr.mc_passed is False
    assert vr.passed    is False


# 7
def test_compute_stats_validation_failure_forces_signal_only():
    # Random trades → regime split fails (pos_pnl=0) AND validation fails (MC p > 0.05)
    # Both reasons produce signal_only=True; test verifies validation is wired in
    trades = [TradeRecord(entry_bar=i, exit_bar=i+1, direction='short',
                          entry_price=100.0, exit_price=99.0 if i%2==0 else 101.0,
                          pnl_pts=1.0 if i%2==0 else -1.0,
                          regime_at_entry='positive', level_tier=1, level_strength=0.9)
              for i in range(50)]
    result = compute_stats(trades, run_validation=True)
    assert result.signal_only is True
    assert result.validation is not None
    assert result.validation.passed is False
```

---

## Acceptance Criteria

- All 7 new tests pass
- `monte_carlo_pvalue([10]*50)` → `0.0`
- `monte_carlo_pvalue` with random alternating PnLs → `> 0.05`
- `kupiec_pof` with < 10 trades → `1.0` (auto-pass)
- `kupiec_pof` with well-calibrated 5% exceedances → `>= 0.05`
- `validate(consistent_winners).passed` → `True`
- `validate(random_pnls).passed` → `False`
- `BacktestResult.validation` is `None` when `run_validation=False`
- All existing 124 tests still pass

---

## Relationship to GPR Sub-projects

| Sub-project | Note |
|---|---|
| 4 — backtest harness (done) | `TradeRecord`, `replay()`, `compute_stats()` — consumed here |
| 5 — pricer (done) | Correctness prerequisite |
| 6 — freshness (done) | Data quality prerequisite |
| 7 — statistical validation (this) | PBO/DSR replaced by MC permutation + Kupiec |
| 8 — nautilus_trader | Will consume `signal_only` verdict after validation |
