# GPR Level Projection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `level_projection/` package — significant-strike selection + index→futures basis projection — as the foundation of the GPR strategy.

**Architecture:** Three focused modules (`models.py`, `selector.py`, `basis.py`) inside a new `level_projection/` package in the existing `D:/2026/GEX` repo. Consumes `find_levels()` output dict from existing `levels.py`. No changes to engine internals.

**Tech Stack:** Python 3.13, pandas, dataclasses, pytest

---

## File Map

| File | Action | Purpose |
|---|---|---|
| `level_projection/__init__.py` | Create | Public API exports |
| `level_projection/models.py` | Create | `Level` + `FutLevel` dataclasses |
| `level_projection/selector.py` | Create | `select_levels()` |
| `level_projection/basis.py` | Create | `measure_basis()` + `project()` |
| `tests/test_level_projection.py` | Create | All 12 fixture-based tests |

---

## Shared Test Fixture

All four tasks use this fixture. It is defined once at the top of `tests/test_level_projection.py` and reused in every test function.

```python
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pandas as pd
from level_projection import Level, FutLevel, select_levels, measure_basis, project

# max |gex| = 900 (at 5300)
# Tier-1 anchors: call_wall=5350, put_wall=5200, zero_gamma=5280, pin=5300
# Tier-2 candidate above threshold: 5320 (|gex|=400, strength=0.444 >= 0.30)
# Below threshold: 5100 (|gex|=150, strength=0.167), 5250 (|gex|=200, strength=0.222)
BY_STRIKE = pd.DataFrame([
    {'strike': 5100.0, 'gex': -150.0, 'vex': 0.0, 'chex': 0.0},
    {'strike': 5200.0, 'gex': -500.0, 'vex': 0.0, 'chex': 0.0},
    {'strike': 5250.0, 'gex': -200.0, 'vex': 0.0, 'chex': 0.0},
    {'strike': 5280.0, 'gex':  100.0, 'vex': 0.0, 'chex': 0.0},
    {'strike': 5300.0, 'gex':  900.0, 'vex': 0.0, 'chex': 0.0},
    {'strike': 5320.0, 'gex':  400.0, 'vex': 0.0, 'chex': 0.0},
    {'strike': 5350.0, 'gex':  700.0, 'vex': 0.0, 'chex': 0.0},
])

SNAPSHOT = {
    'spot': 5300.0,
    'call_wall': 5350.0,
    'put_wall':  5200.0,
    'gamma_flip': 5280.0,
    'hvl': 5200.0,
    'total_gex': 2250.0,
    'by_strike': BY_STRIKE,
}
```

---

## Task 1: Package Scaffold + Models

**Files:**
- Create: `D:/2026/GEX/level_projection/__init__.py`
- Create: `D:/2026/GEX/level_projection/models.py`
- Create: `D:/2026/GEX/tests/test_level_projection.py` (models tests only for now)

- [ ] **Step 1: Write failing model tests**

Create `tests/test_level_projection.py` with the shared fixture and two model tests:

```python
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pandas as pd
from level_projection import Level, FutLevel, select_levels, measure_basis, project

BY_STRIKE = pd.DataFrame([
    {'strike': 5100.0, 'gex': -150.0, 'vex': 0.0, 'chex': 0.0},
    {'strike': 5200.0, 'gex': -500.0, 'vex': 0.0, 'chex': 0.0},
    {'strike': 5250.0, 'gex': -200.0, 'vex': 0.0, 'chex': 0.0},
    {'strike': 5280.0, 'gex':  100.0, 'vex': 0.0, 'chex': 0.0},
    {'strike': 5300.0, 'gex':  900.0, 'vex': 0.0, 'chex': 0.0},
    {'strike': 5320.0, 'gex':  400.0, 'vex': 0.0, 'chex': 0.0},
    {'strike': 5350.0, 'gex':  700.0, 'vex': 0.0, 'chex': 0.0},
])

SNAPSHOT = {
    'spot': 5300.0,
    'call_wall': 5350.0,
    'put_wall':  5200.0,
    'gamma_flip': 5280.0,
    'hvl': 5200.0,
    'total_gex': 2250.0,
    'by_strike': BY_STRIKE,
}


def test_level_dataclass_fields():
    lv = Level(strike=5300.0, gex=900.0, tier=1, strength=1.0, label='pin')
    assert lv.strike == 5300.0
    assert lv.gex == 900.0
    assert lv.tier == 1
    assert lv.strength == 1.0
    assert lv.label == 'pin'


def test_futlevel_dataclass_fields():
    fl = FutLevel(fut_price=5320.0, tier=1, strength=1.0, label='pin',
                  source_strike=5300.0, basis_pts=20.0)
    assert fl.fut_price == 5320.0
    assert fl.source_strike == 5300.0
    assert fl.basis_pts == 20.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd D:/2026/GEX
python -m pytest tests/test_level_projection.py::test_level_dataclass_fields tests/test_level_projection.py::test_futlevel_dataclass_fields -v
```

Expected: `ModuleNotFoundError: No module named 'level_projection'`

- [ ] **Step 3: Create `level_projection/models.py`**

```python
from dataclasses import dataclass


@dataclass
class Level:
    strike: float
    gex: float
    tier: int
    strength: float
    label: str


@dataclass
class FutLevel:
    fut_price: float
    tier: int
    strength: float
    label: str
    source_strike: float
    basis_pts: float
```

- [ ] **Step 4: Create `level_projection/__init__.py`** (stub — will be completed in Task 4)

```python
from .models import Level, FutLevel

__all__ = ['Level', 'FutLevel']
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd D:/2026/GEX
python -m pytest tests/test_level_projection.py::test_level_dataclass_fields tests/test_level_projection.py::test_futlevel_dataclass_fields -v
```

Expected:
```
test_level_dataclass_fields PASSED
test_futlevel_dataclass_fields PASSED
2 passed
```

- [ ] **Step 6: Commit**

```bash
cd D:/2026/GEX
git add level_projection/__init__.py level_projection/models.py tests/test_level_projection.py
git commit -m "feat: add level_projection package scaffold and Level/FutLevel models"
```

---

## Task 2: `select_levels()` — Significant Strike Selection

**Files:**
- Create: `D:/2026/GEX/level_projection/selector.py`
- Modify: `D:/2026/GEX/tests/test_level_projection.py` (add 7 selector tests)

- [ ] **Step 1: Add selector tests to `tests/test_level_projection.py`**

Append these tests to the existing file:

```python
def test_select_levels_always_includes_tier1():
    """call_wall, put_wall, pin, zero_gamma always present regardless of threshold."""
    levels = select_levels(SNAPSHOT, pct_threshold=0.99)  # very high threshold
    labels = {lv.label for lv in levels}
    assert 'call_wall' in labels
    assert 'put_wall' in labels
    assert 'pin' in labels
    assert 'zero_gamma' in labels


def test_select_levels_below_threshold_excluded():
    """Strike 5100 (strength=0.167) and 5250 (strength=0.222) excluded at default threshold."""
    levels = select_levels(SNAPSHOT, pct_threshold=0.30)
    strikes = {lv.strike for lv in levels}
    assert 5100.0 not in strikes
    assert 5250.0 not in strikes


def test_select_levels_above_threshold_included():
    """Strike 5320 (strength=0.444) included as Tier-2 cluster at default threshold."""
    levels = select_levels(SNAPSHOT, pct_threshold=0.30)
    strikes = {lv.strike for lv in levels}
    assert 5320.0 in strikes


def test_select_levels_tier_assignment():
    """Tier-1 = walls/pin/zero_gamma; clusters = Tier-2."""
    levels = select_levels(SNAPSHOT, pct_threshold=0.30)
    by_label = {lv.label: lv for lv in levels}
    assert by_label['call_wall'].tier == 1
    assert by_label['put_wall'].tier == 1
    assert by_label['pin'].tier == 1
    assert by_label['zero_gamma'].tier == 1
    assert by_label['cluster'].tier == 2


def test_select_levels_strength_normalised():
    """All strength values in [0, 1]; the max strength across levels == 1.0."""
    levels = select_levels(SNAPSHOT, pct_threshold=0.30)
    assert all(0.0 <= lv.strength <= 1.0 for lv in levels)
    assert max(lv.strength for lv in levels) == pytest.approx(1.0)


def test_select_levels_empty_snapshot():
    """Empty by_strike returns empty list without error."""
    empty_snap = {**SNAPSHOT, 'by_strike': pd.DataFrame()}
    result = select_levels(empty_snap, pct_threshold=0.30)
    assert result == []


def test_select_levels_no_call_wall():
    """call_wall=None handled gracefully; no call_wall Level in output."""
    snap = {**SNAPSHOT, 'call_wall': None}
    levels = select_levels(snap, pct_threshold=0.30)
    labels = [lv.label for lv in levels]
    assert 'call_wall' not in labels
    assert 'put_wall' in labels  # other Tier-1 still present
```

- [ ] **Step 2: Run selector tests to verify they fail**

```bash
cd D:/2026/GEX
python -m pytest tests/test_level_projection.py -k "select_levels" -v
```

Expected: `ImportError: cannot import name 'select_levels' from 'level_projection'`

- [ ] **Step 3: Create `level_projection/selector.py`**

```python
from __future__ import annotations
import pandas as pd
from .models import Level


def select_levels(snapshot: dict, pct_threshold: float = 0.30) -> list[Level]:
    """
    Select significant gamma strikes from a find_levels() snapshot.

    Always includes (Tier-1): call_wall, put_wall, pin_strike, zero_gamma.
    Adds (Tier-2): any strike where |gex| >= pct_threshold * max|gex|.

    Args:
        snapshot: dict returned by levels.find_levels()
        pct_threshold: minimum |gex| fraction of max to qualify as Tier-2

    Returns:
        List of Level objects sorted by strike ascending.
    """
    by_strike: pd.DataFrame = snapshot.get('by_strike', pd.DataFrame())
    if by_strike.empty:
        return []

    max_abs_gex = by_strike['gex'].abs().max()
    if max_abs_gex == 0:
        return []

    # Tier-1 anchors: always included
    call_wall  = snapshot.get('call_wall')
    put_wall   = snapshot.get('put_wall')
    gamma_flip = snapshot.get('gamma_flip')
    pin_strike = float(by_strike.loc[by_strike['gex'].abs().idxmax(), 'strike'])

    tier1: dict[float, str] = {}
    if call_wall is not None:
        tier1[float(call_wall)] = 'call_wall'
    if put_wall is not None:
        tier1[float(put_wall)] = 'put_wall'
    if gamma_flip is not None:
        tier1[float(gamma_flip)] = 'zero_gamma'
    if pin_strike not in tier1:
        tier1[pin_strike] = 'pin'

    def _gex_at(strike: float) -> float:
        row = by_strike[by_strike['strike'] == strike]
        return float(row.iloc[0]['gex']) if not row.empty else 0.0

    levels: list[Level] = []

    for strike, label in tier1.items():
        gex = _gex_at(strike)
        strength = abs(gex) / max_abs_gex
        levels.append(Level(strike=strike, gex=gex, tier=1, strength=strength, label=label))

    tier1_strikes = set(tier1.keys())

    for _, row in by_strike.iterrows():
        strike = float(row['strike'])
        if strike in tier1_strikes:
            continue
        gex = float(row['gex'])
        strength = abs(gex) / max_abs_gex
        if strength >= pct_threshold:
            levels.append(Level(strike=strike, gex=gex, tier=2, strength=strength, label='cluster'))

    levels.sort(key=lambda lv: lv.strike)
    return levels
```

- [ ] **Step 4: Add `select_levels` to `level_projection/__init__.py`**

```python
from .models import Level, FutLevel
from .selector import select_levels

__all__ = ['Level', 'FutLevel', 'select_levels']
```

- [ ] **Step 5: Run selector tests to verify they pass**

```bash
cd D:/2026/GEX
python -m pytest tests/test_level_projection.py -k "select_levels or level_dataclass or futlevel_dataclass" -v
```

Expected:
```
test_level_dataclass_fields PASSED
test_futlevel_dataclass_fields PASSED
test_select_levels_always_includes_tier1 PASSED
test_select_levels_below_threshold_excluded PASSED
test_select_levels_above_threshold_included PASSED
test_select_levels_tier_assignment PASSED
test_select_levels_strength_normalised PASSED
test_select_levels_empty_snapshot PASSED
test_select_levels_no_call_wall PASSED
9 passed
```

- [ ] **Step 6: Commit**

```bash
cd D:/2026/GEX
git add level_projection/selector.py level_projection/__init__.py tests/test_level_projection.py
git commit -m "feat: add select_levels() — significant-strike selection with tier/strength"
```

---

## Task 3: `measure_basis()` + `project()`

**Files:**
- Create: `D:/2026/GEX/level_projection/basis.py`
- Modify: `D:/2026/GEX/tests/test_level_projection.py` (add 5 basis tests)

- [ ] **Step 1: Add basis tests to `tests/test_level_projection.py`**

Append to the existing file:

```python
def test_measure_basis():
    assert measure_basis(5320.0, 5300.0) == pytest.approx(20.0)


def test_measure_basis_negative():
    assert measure_basis(5280.0, 5300.0) == pytest.approx(-20.0)


def test_project_shifts_by_basis():
    """Every FutLevel.fut_price == source_strike + basis_pts."""
    levels = select_levels(SNAPSHOT, pct_threshold=0.30)
    basis = 25.0
    fut_levels = project(levels, basis)
    for fl in fut_levels:
        assert fl.fut_price == pytest.approx(fl.source_strike + basis)


def test_project_preserves_tier_strength_label():
    """tier, strength, label unchanged after projection."""
    levels = select_levels(SNAPSHOT, pct_threshold=0.30)
    fut_levels = project(levels, basis_pts=20.0)
    for orig, proj in zip(
        sorted(levels, key=lambda l: l.strike),
        sorted(fut_levels, key=lambda f: f.source_strike),
    ):
        assert proj.tier == orig.tier
        assert proj.strength == pytest.approx(orig.strength)
        assert proj.label == orig.label


def test_project_stores_source_and_basis():
    """source_strike and basis_pts stored correctly on each FutLevel."""
    levels = select_levels(SNAPSHOT, pct_threshold=0.30)
    basis = 42.5
    fut_levels = project(levels, basis_pts=basis)
    orig_strikes = {lv.strike for lv in levels}
    for fl in fut_levels:
        assert fl.source_strike in orig_strikes
        assert fl.basis_pts == pytest.approx(basis)
```

- [ ] **Step 2: Run basis tests to verify they fail**

```bash
cd D:/2026/GEX
python -m pytest tests/test_level_projection.py -k "basis or project" -v
```

Expected: `ImportError: cannot import name 'measure_basis' from 'level_projection'`

- [ ] **Step 3: Create `level_projection/basis.py`**

```python
from __future__ import annotations
from .models import Level, FutLevel


def measure_basis(future_price: float, index_value: float) -> float:
    """
    Additive basis in index points.

    basis_pts = future_price - index_value

    Works for ES/SPX and NQ/NDX (both track 1:1 in points).
    No hardcoded multiplier — basis is always measured, never assumed.
    """
    return future_price - index_value


def project(levels: list[Level], basis_pts: float) -> list[FutLevel]:
    """
    Project each index-space Level to a futures price.

    fut_price = strike + basis_pts

    Args:
        levels:    list of Level objects (from select_levels)
        basis_pts: measured basis from measure_basis()

    Returns:
        list of FutLevel objects in the same order as input levels.
    """
    return [
        FutLevel(
            fut_price=level.strike + basis_pts,
            tier=level.tier,
            strength=level.strength,
            label=level.label,
            source_strike=level.strike,
            basis_pts=basis_pts,
        )
        for level in levels
    ]
```

- [ ] **Step 4: Add `measure_basis` and `project` to `level_projection/__init__.py`**

```python
from .models import Level, FutLevel
from .selector import select_levels
from .basis import measure_basis, project

__all__ = ['Level', 'FutLevel', 'select_levels', 'measure_basis', 'project']
```

- [ ] **Step 5: Run all tests to verify they pass**

```bash
cd D:/2026/GEX
python -m pytest tests/test_level_projection.py -v
```

Expected:
```
test_level_dataclass_fields PASSED
test_futlevel_dataclass_fields PASSED
test_select_levels_always_includes_tier1 PASSED
test_select_levels_below_threshold_excluded PASSED
test_select_levels_above_threshold_included PASSED
test_select_levels_tier_assignment PASSED
test_select_levels_strength_normalised PASSED
test_select_levels_empty_snapshot PASSED
test_select_levels_no_call_wall PASSED
test_measure_basis PASSED
test_measure_basis_negative PASSED
test_project_shifts_by_basis PASSED
test_project_preserves_tier_strength_label PASSED
test_project_stores_source_and_basis PASSED
14 passed
```

- [ ] **Step 6: Run the full test suite to confirm no regressions**

```bash
cd D:/2026/GEX
python -m pytest -v
```

Expected: all existing tests still pass + 14 new tests pass.

- [ ] **Step 7: Commit**

```bash
cd D:/2026/GEX
git add level_projection/basis.py level_projection/__init__.py tests/test_level_projection.py
git commit -m "feat: add measure_basis() and project() — index-to-futures basis projection"
```

---

## Task 4: Final Verification + Push

- [ ] **Step 1: Verify public API imports cleanly**

```bash
cd D:/2026/GEX
python -c "from level_projection import select_levels, measure_basis, project, Level, FutLevel; print('OK')"
```

Expected: `OK`

- [ ] **Step 2: Verify no hardcoded multiplier in codebase**

```bash
cd D:/2026/GEX
grep -r "multiplier\|10\.01\|40\.\|41\." level_projection/
```

Expected: no output (no hardcoded multipliers)

- [ ] **Step 3: Run full test suite one final time**

```bash
cd D:/2026/GEX
python -m pytest -v
```

Expected: all tests pass (existing + 14 new)

- [ ] **Step 4: Push**

```bash
cd D:/2026/GEX
git push
```

---

## Self-Review Checklist

**Spec coverage:**
- [x] `Level` + `FutLevel` dataclasses → Task 1
- [x] `select_levels()` always includes Tier-1 → Task 2, test 1
- [x] Below-threshold strikes excluded → Task 2, test 2
- [x] Above-threshold clusters included → Task 2, test 3
- [x] Tier-1 / Tier-2 assignment → Task 2, test 4
- [x] `strength` normalised 0–1 → Task 2, test 5
- [x] Empty snapshot edge case → Task 2, test 6
- [x] `call_wall=None` edge case → Task 2, test 7
- [x] `measure_basis()` additive, no multiplier → Task 3, tests 1-2
- [x] `project()` shifts by basis → Task 3, test 3
- [x] `project()` preserves tier/strength/label → Task 3, test 4
- [x] `source_strike` + `basis_pts` stored → Task 3, test 5
- [x] No hardcoded multiplier → Task 4, step 2 grep
- [x] Clean public import → Task 4, step 1
