# GPR Level Projection — Design Spec
**Date:** 2026-06-03
**Status:** Approved
**Sub-project:** 1 of 6 — Gamma-Pin Reversion (GPR) strategy

## Overview

Add a `level_projection/` package to the existing `D:/2026/GEX` project. It consumes the output of the existing `find_levels()` function and produces:
1. A filtered, tiered list of **significant** gamma strikes (Level objects)
2. Those strikes projected onto futures prices via a **continuously measured basis** (no hardcoded multiplier)

This is the foundation of the GPR strategy. All downstream modules (`regime`, `signal`, `backtest`) consume `[FutLevel]` produced here.

---

## File Structure

```
D:/2026/GEX/
├── level_projection/
│   ├── __init__.py         # exports: select_levels, measure_basis, project, Level, FutLevel
│   ├── models.py           # Level + FutLevel dataclasses
│   ├── selector.py         # select_levels()
│   └── basis.py            # measure_basis() + project()
└── tests/
    └── test_level_projection.py
```

No changes to existing engine files (`levels.py`, `greeks.py`, `data.py`).

---

## Module Specifications

### `models.py`

```python
from dataclasses import dataclass

@dataclass
class Level:
    strike: float       # index strike price
    gex: float          # net gamma notional at this strike (signed)
    tier: int           # 1 = primary (walls/pin/zero_gamma), 2 = secondary cluster
    strength: float     # |gex| / max(|gex|) across chain — normalised 0–1
    label: str          # "call_wall" | "put_wall" | "pin" | "zero_gamma" | "cluster"

@dataclass
class FutLevel:
    fut_price: float        # projected futures price
    tier: int
    strength: float
    label: str
    source_strike: float    # the index strike it was projected from
    basis_pts: float        # basis applied at projection time
```

---

### `selector.py`

**`select_levels(snapshot: dict, pct_threshold: float = 0.30) -> list[Level]`**

`snapshot` is the dict returned by `find_levels()`:
```python
{
    'spot': float,
    'call_wall': float | None,
    'put_wall': float | None,
    'gamma_flip': float,        # zero-gamma level
    'hvl': float,
    'by_strike': pd.DataFrame,  # columns: strike, gex, vex, chex
    'total_gex': float,
    ...
}
```

**Algorithm:**

1. Compute `max_abs_gex = by_strike['gex'].abs().max()`
2. Compute `pin_strike` = strike with max `|gex|` (absolute) on the full chain:
   ```python
   pin_strike = float(by_strike.loc[by_strike['gex'].abs().idxmax(), 'strike'])
   ```
3. Build **Tier-1 set** — always included regardless of threshold:
   - `call_wall` (label: `"call_wall"`) if not None
   - `put_wall` (label: `"put_wall"`) if not None
   - `pin_strike` (label: `"pin"`)
   - `gamma_flip` (label: `"zero_gamma"`)
4. Build **Tier-2 set** — strikes where `|gex| >= pct_threshold × max_abs_gex` and not already in Tier-1:
   - label: `"cluster"`
5. For every included strike, look up its `gex` in `by_strike`, compute `strength = |gex| / max_abs_gex`
6. Return combined list sorted by strike ascending

**Edge cases:**
- If `by_strike` is empty or `max_abs_gex == 0`: return `[]`
- If `call_wall` or `put_wall` is `None`: skip them (0DTE chains may have no put_wall)
- Duplicate strikes (e.g. pin_strike == call_wall): deduplicate, keep Tier-1 label

---

### `basis.py`

**`measure_basis(future_price: float, index_value: float) -> float`**

```python
def measure_basis(future_price: float, index_value: float) -> float:
    """
    Additive basis in index points.
    basis_pts = future_price - index_value
    Works for ES/SPX and NQ/NDX (both track 1:1 in points).
    No multiplier — never hardcode one.
    """
    return future_price - index_value
```

**`project(levels: list[Level], basis_pts: float) -> list[FutLevel]`**

```python
def project(levels: list[Level], basis_pts: float) -> list[FutLevel]:
    """
    Project each index-space Level to a futures price.
    fut_price = strike + basis_pts
    """
```

Returns a `FutLevel` for every input `Level`, preserving tier/strength/label. `source_strike` and `basis_pts` stored for auditability.

---

### `__init__.py`

```python
from .models import Level, FutLevel
from .selector import select_levels
from .basis import measure_basis, project

__all__ = ['Level', 'FutLevel', 'select_levels', 'measure_basis', 'project']
```

---

## Tests (`tests/test_level_projection.py`)

All tests use fixtures — no network calls.

**Fixture snapshot** (synthetic `find_levels()` output):
```python
SPOT = 5300.0
BY_STRIKE = pd.DataFrame([
    {'strike': 5100.0, 'gex': -150.0, 'vex': 0.0, 'chex': 0.0},  # below threshold (150/900=0.17 < 0.30)
    {'strike': 5200.0, 'gex': -500.0, 'vex': 0.0, 'chex': 0.0},  # put_wall (most negative below spot)
    {'strike': 5250.0, 'gex': -200.0, 'vex': 0.0, 'chex': 0.0},  # cluster (200/900=0.22 < 0.30, excluded)
    {'strike': 5280.0, 'gex':  100.0, 'vex': 0.0, 'chex': 0.0},  # zero_gamma
    {'strike': 5300.0, 'gex':  900.0, 'vex': 0.0, 'chex': 0.0},  # pin (max |gex|=900, distinct from walls)
    {'strike': 5350.0, 'gex':  700.0, 'vex': 0.0, 'chex': 0.0},  # call_wall (700/900=0.78 >= 0.30, Tier-2 if not wall)
])
# pin_strike=5300 (|gex|=900), call_wall=5350, put_wall=5200, zero_gamma=5280
# All four are distinct strikes — no dedup needed in this fixture
SNAPSHOT = {
    'spot': SPOT,
    'call_wall': 5350.0,
    'put_wall': 5200.0,
    'gamma_flip': 5280.0,
    'hvl': 5200.0,
    'total_gex': 1750.0,
    'by_strike': BY_STRIKE,
}
```

**Tests:**

1. `test_select_levels_always_includes_tier1` — call_wall, put_wall, pin, zero_gamma always present
2. `test_select_levels_below_threshold_excluded` — strike at 5100 (strength 0.19) excluded at default 0.30
3. `test_select_levels_above_threshold_included` — strike at 5350 (strength 0.875) included
4. `test_select_levels_tier_assignment` — call_wall/put_wall/pin/zero_gamma = tier 1; clusters = tier 2
5. `test_select_levels_strength_normalised` — all strength values in [0, 1]; max strength == 1.0
6. `test_select_levels_empty_snapshot` — `by_strike` empty → returns `[]`
7. `test_select_levels_no_call_wall` — `call_wall=None` → still returns without error, no call_wall Level
8. `test_measure_basis` — `measure_basis(5320.0, 5300.0)` == `20.0`
9. `test_measure_basis_negative` — `measure_basis(5280.0, 5300.0)` == `-20.0`
10. `test_project_shifts_by_basis` — each `FutLevel.fut_price == source_strike + basis_pts`
11. `test_project_preserves_tier_strength_label` — tier, strength, label unchanged after projection
12. `test_project_stores_source_and_basis` — `source_strike` and `basis_pts` set correctly on each FutLevel

---

## Acceptance Criteria

- All 12 tests pass
- No hardcoded multiplier anywhere in `level_projection/`
- Equal-weight / low-gamma strikes below `pct_threshold` are excluded
- Tier-1 strikes always present regardless of threshold
- `strength` is normalised 0–1 and correlates monotonically with `|gex|`
- Module imports cleanly: `from level_projection import select_levels, measure_basis, project`

---

## Relationship to GPR Sub-projects

| Sub-project | Depends on |
|---|---|
| 1 — level_projection (this) | `find_levels()` output only |
| 2 — regime | `find_levels()` output (net_gex, gamma_flip) |
| 3 — signal | `[FutLevel]` from level_projection + Regime from regime |
| 4 — backtest | signal state machine in replay mode |
| 5 — execution | backtest validation gate passed |
| 6 — serve | all of the above |
