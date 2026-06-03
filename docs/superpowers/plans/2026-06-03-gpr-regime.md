# GPR Regime Classification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `regime/` package — gamma regime classification (positive/negative) with distance-scaled conviction — as the gate layer of the GPR strategy.

**Architecture:** Two focused modules (`models.py`, `classifier.py`) inside a new `regime/` package. Consumes `find_levels()` output dict. No changes to engine internals.

**Tech Stack:** Python 3.13, dataclasses, pytest

---

## File Map

| File | Action | Purpose |
|---|---|---|
| `regime/__init__.py` | Create | Public API exports |
| `regime/models.py` | Create | `Regime` dataclass |
| `regime/classifier.py` | Create | `classify()` |
| `tests/test_regime.py` | Create | 9 fixture-based tests |

---

## Shared Test Fixtures

Defined at the top of `tests/test_regime.py`, reused across all tests:

```python
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from regime import Regime, classify

# spot 50 pts above flip → positive, conviction > 0
POSITIVE_SNAP = {
    'spot': 5300.0,
    'gamma_flip': 5250.0,
    'total_gex': 1_500_000_000.0,
}

# spot 50 pts below flip → negative, conviction > 0
NEGATIVE_SNAP = {
    'spot': 5200.0,
    'gamma_flip': 5250.0,
    'total_gex': -800_000_000.0,
}

# spot exactly at flip → conviction == 0
AT_FLIP_SNAP = {
    'spot': 5250.0,
    'gamma_flip': 5250.0,
    'total_gex': 100_000.0,
}

# spot 300 pts above flip (300/5300 = 5.66% > 5%) → conviction capped at 1.0
FAR_SNAP = {
    'spot': 5300.0,
    'gamma_flip': 5000.0,
    'total_gex': 2_000_000_000.0,
}
```

---

## Task 1: Package Scaffold + Regime Model

**Files:**
- Create: `D:/2026/GEX/regime/__init__.py`
- Create: `D:/2026/GEX/regime/models.py`
- Create: `D:/2026/GEX/tests/test_regime.py` (model test only for now)

- [ ] **Step 1: Write failing model test**

Create `D:/2026/GEX/tests/test_regime.py`:

```python
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from regime import Regime, classify

POSITIVE_SNAP = {
    'spot': 5300.0,
    'gamma_flip': 5250.0,
    'total_gex': 1_500_000_000.0,
}

NEGATIVE_SNAP = {
    'spot': 5200.0,
    'gamma_flip': 5250.0,
    'total_gex': -800_000_000.0,
}

AT_FLIP_SNAP = {
    'spot': 5250.0,
    'gamma_flip': 5250.0,
    'total_gex': 100_000.0,
}

FAR_SNAP = {
    'spot': 5300.0,
    'gamma_flip': 5000.0,
    'total_gex': 2_000_000_000.0,
}


def test_regime_dataclass_fields():
    r = Regime(state='positive', distance_to_zero_gamma=50.0, conviction=0.5)
    assert r.state == 'positive'
    assert r.distance_to_zero_gamma == 50.0
    assert r.conviction == 0.5
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd D:/2026/GEX
python -m pytest tests/test_regime.py::test_regime_dataclass_fields -v
```

Expected: `ModuleNotFoundError: No module named 'regime'`

- [ ] **Step 3: Create `regime/models.py`**

```python
from dataclasses import dataclass


@dataclass
class Regime:
    state: str                     # "positive" | "negative"
    distance_to_zero_gamma: float  # signed points: + when spot above flip, - when below
    conviction: float              # 0.0–1.0; scales monotonically with |distance|
```

- [ ] **Step 4: Create `regime/__init__.py`** (stub)

```python
from .models import Regime

__all__ = ['Regime']
```

- [ ] **Step 5: Run test to verify it passes**

```bash
cd D:/2026/GEX
python -m pytest tests/test_regime.py::test_regime_dataclass_fields -v
```

Expected: `1 passed`

- [ ] **Step 6: Commit**

```bash
cd D:/2026/GEX
git add regime/__init__.py regime/models.py tests/test_regime.py
git commit -m "feat: add regime package scaffold and Regime model"
```

---

## Task 2: `classify()` — Regime Classification

**Files:**
- Create: `D:/2026/GEX/regime/classifier.py`
- Modify: `D:/2026/GEX/regime/__init__.py` (add classify)
- Modify: `D:/2026/GEX/tests/test_regime.py` (add 8 tests, fix import)

- [ ] **Step 1: Add 8 classify tests to `tests/test_regime.py`**

Update the import line:
```python
from regime import Regime, classify
```

Append these tests:

```python
def test_classify_positive_when_net_gex_positive():
    r = classify(POSITIVE_SNAP)
    assert r.state == 'positive'


def test_classify_negative_when_net_gex_negative():
    r = classify(NEGATIVE_SNAP)
    assert r.state == 'negative'


def test_classify_negative_when_net_gex_zero():
    snap = {**POSITIVE_SNAP, 'total_gex': 0.0}
    r = classify(snap)
    assert r.state == 'negative'


def test_classify_distance_positive_when_spot_above_flip():
    r = classify(POSITIVE_SNAP)
    # spot=5300, gamma_flip=5250 → distance = +50
    assert r.distance_to_zero_gamma == pytest.approx(50.0)


def test_classify_distance_negative_when_spot_below_flip():
    r = classify(NEGATIVE_SNAP)
    # spot=5200, gamma_flip=5250 → distance = -50
    assert r.distance_to_zero_gamma == pytest.approx(-50.0)


def test_classify_distance_zero_at_flip():
    r = classify(AT_FLIP_SNAP)
    assert r.distance_to_zero_gamma == pytest.approx(0.0)


def test_classify_conviction_zero_at_flip():
    r = classify(AT_FLIP_SNAP)
    assert r.conviction == pytest.approx(0.0)


def test_classify_conviction_increases_with_distance():
    # FAR_SNAP: 300 pts away. POSITIVE_SNAP: 50 pts away. FAR should have higher conviction.
    r_near = classify(POSITIVE_SNAP)
    r_far  = classify(FAR_SNAP)
    assert r_far.conviction > r_near.conviction


def test_classify_conviction_capped_at_1():
    # FAR_SNAP: 300/5300 = 5.66% > default conviction_scale=0.05 → capped at 1.0
    r = classify(FAR_SNAP)
    assert r.conviction == pytest.approx(1.0)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd D:/2026/GEX
python -m pytest tests/test_regime.py -k "classify" -v
```

Expected: `ImportError: cannot import name 'classify' from 'regime'`

- [ ] **Step 3: Create `regime/classifier.py`**

```python
from __future__ import annotations
from .models import Regime


def classify(snapshot: dict, conviction_scale: float = 0.05) -> Regime:
    """
    Classify the current gamma regime from a find_levels() snapshot.

    Args:
        snapshot:         dict from find_levels() with 'total_gex', 'spot', 'gamma_flip'
        conviction_scale: fraction of spot that equals full conviction distance (default 5%)

    Returns:
        Regime with state ('positive'|'negative'), distance_to_zero_gamma, conviction (0–1)
    """
    spot       = float(snapshot['spot'])
    gamma_flip = float(snapshot['gamma_flip'])
    total_gex  = float(snapshot['total_gex'])

    if spot == 0.0:
        raise ValueError("spot cannot be zero")

    state = 'positive' if total_gex > 0 else 'negative'

    distance_to_zero_gamma = spot - gamma_flip

    full_conviction_distance = spot * conviction_scale
    conviction = min(1.0, abs(distance_to_zero_gamma) / full_conviction_distance)

    return Regime(
        state=state,
        distance_to_zero_gamma=distance_to_zero_gamma,
        conviction=conviction,
    )
```

- [ ] **Step 4: Update `regime/__init__.py`**

```python
from .models import Regime
from .classifier import classify

__all__ = ['Regime', 'classify']
```

- [ ] **Step 5: Run all regime tests**

```bash
cd D:/2026/GEX
python -m pytest tests/test_regime.py -v
```

Expected:
```
test_regime_dataclass_fields PASSED
test_classify_positive_when_net_gex_positive PASSED
test_classify_negative_when_net_gex_negative PASSED
test_classify_negative_when_net_gex_zero PASSED
test_classify_distance_positive_when_spot_above_flip PASSED
test_classify_distance_negative_when_spot_below_flip PASSED
test_classify_distance_zero_at_flip PASSED
test_classify_conviction_zero_at_flip PASSED
test_classify_conviction_increases_with_distance PASSED
test_classify_conviction_capped_at_1 PASSED
10 passed
```

Wait — that's 10 tests (1 model + 9 classify). All 10 must pass.

- [ ] **Step 6: Run full suite to confirm no regressions**

```bash
cd D:/2026/GEX
python -m pytest -v
```

Expected: all existing 60 tests + 10 new = 70 total, all pass.

- [ ] **Step 7: Commit**

```bash
cd D:/2026/GEX
git add regime/classifier.py regime/__init__.py tests/test_regime.py
git commit -m "feat: add classify() — gamma regime classification with conviction score"
```

---

## Task 3: Final Verification + Push

- [ ] **Step 1: Verify public API imports cleanly**

```bash
cd D:/2026/GEX
python -c "from regime import Regime, classify; print('OK')"
```

Expected: `OK`

- [ ] **Step 2: Run full test suite**

```bash
cd D:/2026/GEX
python -m pytest -v
```

Expected: 70 tests pass.

- [ ] **Step 3: Push**

```bash
cd D:/2026/GEX
git push
```

---

## Self-Review Checklist

**Spec coverage:**
- [x] `Regime` dataclass with `state`, `distance_to_zero_gamma`, `conviction` → Task 1
- [x] `state = "positive"` when `total_gex > 0` → Task 2, test 1
- [x] `state = "negative"` when `total_gex < 0` → Task 2, test 2
- [x] `state = "negative"` when `total_gex == 0` → Task 2, test 3
- [x] `distance_to_zero_gamma = spot - gamma_flip` (signed) → Task 2, tests 4-6
- [x] `conviction = 0` when at flip → Task 2, test 7
- [x] `conviction` monotonically increases with distance → Task 2, test 8
- [x] `conviction` capped at 1.0 → Task 2, test 9
- [x] Clean public import → Task 3, step 1
