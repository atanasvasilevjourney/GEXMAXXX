# GPR Regime Classification — Design Spec
**Date:** 2026-06-03
**Status:** Approved
**Sub-project:** 2 of 6 — Gamma-Pin Reversion (GPR) strategy

## Overview

Add a `regime/` package to `D:/2026/GEX`. It consumes the output of `find_levels()` and classifies the current gamma regime as positive or negative, with a conviction score that scales with distance from the zero-gamma flip point.

Regime is the load-bearing gate of the GPR strategy:
- **Positive** → fades ARMED (mean-reversion at gamma levels)
- **Negative** → fades DISABLED (levels become breakout triggers or no-trade)

---

## File Structure

```
D:/2026/GEX/
├── regime/
│   ├── __init__.py         # exports: Regime, classify
│   ├── models.py           # Regime dataclass
│   └── classifier.py       # classify()
└── tests/
    └── test_regime.py
```

No changes to existing engine files.

---

## Module Specifications

### `models.py`

```python
from dataclasses import dataclass


@dataclass
class Regime:
    state: str                     # "positive" | "negative"
    distance_to_zero_gamma: float  # signed points: + when spot above flip, - when below
    conviction: float              # 0.0–1.0; scales monotonically with |distance|
```

`state` is a string sentinel — downstream modules switch on `regime.state == "positive"`.

---

### `classifier.py`

**`classify(snapshot: dict, conviction_scale: float = 0.05) -> Regime`**

`snapshot` is the dict from `find_levels()`:
```python
{
    'spot': float,
    'gamma_flip': float,   # zero-gamma strike (closest cumsum sign change to spot)
    'total_gex': float,    # net dealer gamma (sum of all strikes)
    ...
}
```

**Algorithm:**

1. `state = "positive"` if `total_gex > 0`, else `"negative"`
2. `distance_to_zero_gamma = spot - gamma_flip`
   - Positive: spot is above the flip (fades supported)
   - Negative: spot is below the flip (fades risky)
3. `conviction = min(1.0, abs(distance_to_zero_gamma) / (spot * conviction_scale))`
   - At zero_gamma (distance = 0): conviction = 0.0
   - At `conviction_scale * spot` distance away: conviction = 1.0 (capped)
   - Default `conviction_scale = 0.05` → full conviction at 5% of spot distance

**Edge cases:**
- `spot == gamma_flip` → `distance_to_zero_gamma = 0.0`, `conviction = 0.0`
- `spot == 0` → raise `ValueError("spot cannot be zero")`
- `total_gex == 0` → defaults to `"negative"` (no-trade if no gamma signal)

---

### `__init__.py`

```python
from .models import Regime
from .classifier import classify

__all__ = ['Regime', 'classify']
```

---

## Tests (`tests/test_regime.py`)

All tests use synthetic snapshots — no network calls.

**Fixtures:**
```python
POSITIVE_SNAP = {
    'spot': 5300.0,
    'gamma_flip': 5250.0,   # spot 50 pts above flip
    'total_gex': 1_500_000_000.0,
}

NEGATIVE_SNAP = {
    'spot': 5200.0,
    'gamma_flip': 5250.0,   # spot 50 pts below flip
    'total_gex': -800_000_000.0,
}

AT_FLIP_SNAP = {
    'spot': 5250.0,
    'gamma_flip': 5250.0,   # spot exactly at flip
    'total_gex': 100_000.0, # technically positive but at zero gamma
}
```

**Tests (9):**

1. `test_classify_positive_when_net_gex_positive` — `POSITIVE_SNAP` → `state == "positive"`
2. `test_classify_negative_when_net_gex_negative` — `NEGATIVE_SNAP` → `state == "negative"`
3. `test_classify_negative_when_net_gex_zero` — `total_gex=0` → `state == "negative"`
4. `test_classify_distance_positive_when_spot_above_flip` — `POSITIVE_SNAP` → `distance_to_zero_gamma > 0`
5. `test_classify_distance_negative_when_spot_below_flip` — `NEGATIVE_SNAP` → `distance_to_zero_gamma < 0`
6. `test_classify_distance_zero_at_flip` — `AT_FLIP_SNAP` → `distance_to_zero_gamma == 0.0`
7. `test_classify_conviction_zero_at_flip` — `AT_FLIP_SNAP` → `conviction == 0.0`
8. `test_classify_conviction_increases_with_distance` — two snapshots with different distances; farther → higher conviction
9. `test_classify_conviction_capped_at_1` — snapshot far from flip (> 5% of spot) → `conviction == 1.0`

---

## Acceptance Criteria

- All 9 tests pass
- `state` is always exactly `"positive"` or `"negative"` (never None or other)
- `conviction` is always in `[0.0, 1.0]`
- `conviction` scales monotonically with `|distance_to_zero_gamma|`
- `classify` imports cleanly: `from regime import classify, Regime`

---

## Relationship to GPR Sub-projects

| Sub-project | Uses regime output |
|---|---|
| 1 — level_projection | No |
| 2 — regime (this) | Produces Regime |
| 3 — signal | Gates fades on `regime.state == "positive"` |
| 4 — backtest | Conditions P&L by regime |
