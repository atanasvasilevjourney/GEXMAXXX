# GPR Signal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `signal/` package — GPR state machine (IDLE/ARMED/IN_TRADE/STANDBY) with forced invalidations and pluggable trigger — in paper/replay mode only.

**Architecture:** Four focused modules (`models.py`, `trigger.py`, `risk.py`, `state_machine.py`) in a new `signal/` package. Consumes `Regime` from `regime/` and `[FutLevel]` from `level_projection/`. Returns `Action` enum on each tick.

**Tech Stack:** Python 3.13, dataclasses, enum, pytest

---

## File Map

| File | Action | Purpose |
|---|---|---|
| `signal/__init__.py` | Create | Public API exports |
| `signal/models.py` | Create | SignalState, Action (enums), MarketTick, TradePosition |
| `signal/trigger.py` | Create | ProxyConfirm pluggable trigger |
| `signal/risk.py` | Create | compute_stop(), select_target(), stop_hit() |
| `signal/state_machine.py` | Create | GPRStateMachine.on_tick() |
| `tests/test_signal.py` | Create | 15 tests |

---

## Shared Test Fixtures

At the top of `tests/test_signal.py`:

```python
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from signal_gpr import (
    GPRStateMachine, MarketTick, TradePosition,
    SignalState, Action, ProxyConfirm
)
from level_projection import FutLevel
from regime import Regime

# Positive regime: fades armed
POS_REGIME = Regime(state='positive', distance_to_zero_gamma=50.0, conviction=0.8)

# Negative regime: fades disabled
NEG_REGIME = Regime(state='negative', distance_to_zero_gamma=-50.0, conviction=0.8)

# Tier-1 level at 5350 — resistance above current price
CALL_LEVEL = FutLevel(fut_price=5350.0, tier=1, strength=0.9, label='call_wall',
                      source_strike=5330.0, basis_pts=20.0)

# Tier-1 level at 5250 — support below current price
PUT_LEVEL  = FutLevel(fut_price=5250.0, tier=1, strength=0.8, label='put_wall',
                      source_strike=5230.0, basis_pts=20.0)

# Tier-2 cluster at 5400 (target for a short from 5350)
CLUSTER_LEVEL = FutLevel(fut_price=5400.0, tier=2, strength=0.5, label='cluster',
                          source_strike=5380.0, basis_pts=20.0)

ALL_LEVELS = [PUT_LEVEL, CALL_LEVEL, CLUSTER_LEVEL]

# Standard tick: price near call_wall (5 pts below), positive regime, no confirmation
NEAR_TICK      = MarketTick(price=5345.0, atr=10.0)
# Confirmation tick: rejection bar at level
CONFIRM_TICK   = MarketTick(price=5348.0, atr=10.0, has_rejection_bar=True)
# Far tick: price far from levels (100 pts away)
FAR_TICK       = MarketTick(price=5200.0, atr=10.0)
# Event window tick
EVENT_TICK     = MarketTick(price=5348.0, atr=10.0, has_rejection_bar=True, in_event_window=True)
```

**Important naming note:** The package directory is named `signal/` but Python has a built-in `signal` module. Import via the alias `signal_gpr` — rename the package to `signal_gpr/` to avoid the stdlib conflict.

---

## Task 1: Models + Trigger + Risk

**Files:**
- Create: `D:/2026/GEX/signal_gpr/__init__.py`
- Create: `D:/2026/GEX/signal_gpr/models.py`
- Create: `D:/2026/GEX/signal_gpr/trigger.py`
- Create: `D:/2026/GEX/signal_gpr/risk.py`
- Create: `D:/2026/GEX/tests/test_signal.py` (risk tests only — 3 tests)

- [ ] **Step 1: Write failing risk tests**

Create `D:/2026/GEX/tests/test_signal.py`:

```python
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from signal_gpr import (
    GPRStateMachine, MarketTick, TradePosition,
    SignalState, Action, ProxyConfirm
)
from signal_gpr.risk import compute_stop, select_target, stop_hit
from level_projection import FutLevel
from regime import Regime

POS_REGIME = Regime(state='positive', distance_to_zero_gamma=50.0, conviction=0.8)
NEG_REGIME = Regime(state='negative', distance_to_zero_gamma=-50.0, conviction=0.8)

CALL_LEVEL = FutLevel(fut_price=5350.0, tier=1, strength=0.9, label='call_wall',
                      source_strike=5330.0, basis_pts=20.0)
PUT_LEVEL  = FutLevel(fut_price=5250.0, tier=1, strength=0.8, label='put_wall',
                      source_strike=5230.0, basis_pts=20.0)
CLUSTER_LEVEL = FutLevel(fut_price=5400.0, tier=2, strength=0.5, label='cluster',
                          source_strike=5380.0, basis_pts=20.0)
ALL_LEVELS = [PUT_LEVEL, CALL_LEVEL, CLUSTER_LEVEL]

NEAR_TICK    = MarketTick(price=5345.0, atr=10.0)
CONFIRM_TICK = MarketTick(price=5348.0, atr=10.0, has_rejection_bar=True)
FAR_TICK     = MarketTick(price=5200.0, atr=10.0)
EVENT_TICK   = MarketTick(price=5348.0, atr=10.0, has_rejection_bar=True, in_event_window=True)


def test_compute_stop_long():
    # buffer = max(0.75 * 10, 12) = max(7.5, 12) = 12
    # stop = 5300 - 12 = 5288
    stop = compute_stop(entry=5300.0, direction='long', atr=10.0, k_atr=0.75, min_ticks=12.0)
    assert stop == pytest.approx(5288.0)


def test_compute_stop_short():
    # buffer = max(0.75 * 10, 12) = 12
    # stop = 5300 + 12 = 5312
    stop = compute_stop(entry=5300.0, direction='short', atr=10.0, k_atr=0.75, min_ticks=12.0)
    assert stop == pytest.approx(5312.0)


def test_select_target_nearest_tier1():
    # Short from 5348: nearest Tier-1 BELOW 5348 is PUT_LEVEL at 5250
    target = select_target(entry=5348.0, direction='short', fut_levels=ALL_LEVELS)
    assert target == pytest.approx(5250.0)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd D:/2026/GEX
python -m pytest tests/test_signal.py -k "compute_stop or select_target" -v
```

Expected: `ModuleNotFoundError: No module named 'signal_gpr'`

- [ ] **Step 3: Create `signal_gpr/models.py`**

```python
from enum import Enum
from dataclasses import dataclass


class SignalState(str, Enum):
    IDLE     = "IDLE"
    ARMED    = "ARMED"
    IN_TRADE = "IN_TRADE"
    STANDBY  = "STANDBY"


class Action(str, Enum):
    NONE        = "NONE"
    ENTER_LONG  = "ENTER_LONG"
    ENTER_SHORT = "ENTER_SHORT"
    EXIT        = "EXIT"
    HOLD        = "HOLD"


@dataclass
class MarketTick:
    price: float
    atr: float
    has_rejection_bar: bool = False
    in_event_window: bool = False


@dataclass
class TradePosition:
    direction: str
    entry_price: float
    stop: float
    target: float | None
    level_strength: float
```

- [ ] **Step 4: Create `signal_gpr/trigger.py`**

```python
from .models import MarketTick


class ProxyConfirm:
    """
    Automation proxy for order-flow confirmation (paper/replay mode).
    Fires on rejection-reclaim bar: price wicks through level, closes back inside.
    Signalled by MarketTick.has_rejection_bar in replay mode.
    """
    def confirmed(self, tick: MarketTick) -> bool:
        return tick.has_rejection_bar
```

- [ ] **Step 5: Create `signal_gpr/risk.py`**

```python
from __future__ import annotations
from level_projection import FutLevel


def compute_stop(entry: float, direction: str, atr: float,
                 k_atr: float = 0.75, min_ticks: float = 12.0) -> float:
    """ATR-scaled stop with floor. buffer = max(k_atr * atr, min_ticks)."""
    buffer = max(k_atr * atr, min_ticks)
    return entry - buffer if direction == 'long' else entry + buffer


def select_target(entry: float, direction: str, fut_levels: list[FutLevel]) -> float | None:
    """Nearest Tier-1 FutLevel in trade direction."""
    if direction == 'long':
        candidates = [lv for lv in fut_levels if lv.tier == 1 and lv.fut_price > entry]
        return min(candidates, key=lambda lv: lv.fut_price).fut_price if candidates else None
    else:
        candidates = [lv for lv in fut_levels if lv.tier == 1 and lv.fut_price < entry]
        return max(candidates, key=lambda lv: lv.fut_price).fut_price if candidates else None


def stop_hit(tick: MarketTick, position: TradePosition) -> bool:
    """Returns True if current price has crossed the position stop."""
    if position.direction == 'long':
        return tick.price <= position.stop
    return tick.price >= position.stop


# avoid circular import — import MarketTick and TradePosition locally
from .models import MarketTick, TradePosition
```

- [ ] **Step 6: Create stub `signal_gpr/__init__.py`**

```python
from .models import SignalState, Action, MarketTick, TradePosition
from .trigger import ProxyConfirm

__all__ = ['SignalState', 'Action', 'MarketTick', 'TradePosition', 'ProxyConfirm']
```

- [ ] **Step 7: Run risk tests to verify they pass**

```bash
cd D:/2026/GEX
python -m pytest tests/test_signal.py -k "compute_stop or select_target" -v
```

Expected:
```
test_compute_stop_long PASSED
test_compute_stop_short PASSED
test_select_target_nearest_tier1 PASSED
3 passed
```

- [ ] **Step 8: Commit**

```bash
cd D:/2026/GEX
git add signal_gpr/__init__.py signal_gpr/models.py signal_gpr/trigger.py signal_gpr/risk.py tests/test_signal.py
git commit -m "feat: add signal_gpr models, trigger, and risk functions"
```

---

## Task 2: GPRStateMachine

**Files:**
- Create: `D:/2026/GEX/signal_gpr/state_machine.py`
- Modify: `D:/2026/GEX/signal_gpr/__init__.py` (add GPRStateMachine)
- Modify: `D:/2026/GEX/tests/test_signal.py` (add 12 state machine tests)

- [ ] **Step 1: Create `signal_gpr/state_machine.py`**

```python
from __future__ import annotations
from level_projection import FutLevel
from regime import Regime
from .models import SignalState, Action, MarketTick, TradePosition
from .trigger import ProxyConfirm
from .risk import compute_stop, select_target, stop_hit


def _nearest_tier1(price: float, fut_levels: list[FutLevel], arm_distance_pts: float) -> FutLevel | None:
    """Return nearest Tier-1 level within arm_distance_pts of price, or None."""
    candidates = [lv for lv in fut_levels if lv.tier == 1
                  and abs(lv.fut_price - price) <= arm_distance_pts]
    if not candidates:
        return None
    return min(candidates, key=lambda lv: abs(lv.fut_price - price))


class GPRStateMachine:
    """
    GPR state machine: IDLE → ARMED → IN_TRADE → EXIT cycle.
    Paper/replay mode only — returns Action, never places orders.

    Hard invariants:
    - Fades never arm in negative gamma.
    - IN_TRADE forced exits (stop / regime-flip / event-window) are checked FIRST.
    - Entry fires only on trigger confirmation, not on touch.
    """

    def __init__(self, arm_distance_pts: float = 5.0,
                 trigger: ProxyConfirm | None = None,
                 k_atr: float = 0.75,
                 min_ticks: float = 12.0):
        self.state: SignalState = SignalState.IDLE
        self.position: TradePosition | None = None
        self.armed_level: FutLevel | None = None
        self.arm_distance_pts = arm_distance_pts
        self.trigger = trigger or ProxyConfirm()
        self.k_atr = k_atr
        self.min_ticks = min_ticks

    def on_tick(self, tick: MarketTick, regime: Regime,
                fut_levels: list[FutLevel]) -> Action:

        if self.state == SignalState.IDLE:
            return self._handle_idle(tick, regime, fut_levels)

        if self.state == SignalState.ARMED:
            return self._handle_armed(tick, regime, fut_levels)

        if self.state == SignalState.IN_TRADE:
            return self._handle_in_trade(tick, regime)

        if self.state == SignalState.STANDBY:
            return self._handle_standby(tick, fut_levels)

        return Action.NONE

    def _handle_idle(self, tick: MarketTick, regime: Regime,
                     fut_levels: list[FutLevel]) -> Action:
        if regime.state != 'positive':
            return Action.NONE
        nearest = _nearest_tier1(tick.price, fut_levels, self.arm_distance_pts)
        if nearest is not None:
            self.state = SignalState.ARMED
            self.armed_level = nearest
        return Action.NONE

    def _handle_armed(self, tick: MarketTick, regime: Regime,
                      fut_levels: list[FutLevel]) -> Action:
        # Regime flip → standby
        if regime.state != 'positive':
            self.state = SignalState.STANDBY
            self.armed_level = None
            return Action.NONE

        # Price left zone → back to idle
        nearest = _nearest_tier1(tick.price, fut_levels, self.arm_distance_pts)
        if nearest is None:
            self.state = SignalState.IDLE
            self.armed_level = None
            return Action.NONE

        # Update armed level to nearest
        self.armed_level = nearest

        # No entry during event window
        if tick.in_event_window:
            return Action.NONE

        # Entry on confirmation only
        if not self.trigger.confirmed(tick):
            return Action.NONE

        # Determine fade direction
        if self.armed_level.fut_price > tick.price:
            direction = 'short'
        elif self.armed_level.fut_price < tick.price:
            direction = 'long'
        else:
            return Action.NONE  # price exactly at level — no trade

        stop   = compute_stop(tick.price, direction, tick.atr, self.k_atr, self.min_ticks)
        target = select_target(tick.price, direction, fut_levels)

        self.position = TradePosition(
            direction=direction,
            entry_price=tick.price,
            stop=stop,
            target=target,
            level_strength=self.armed_level.strength,
        )
        self.state = SignalState.IN_TRADE
        return Action.ENTER_LONG if direction == 'long' else Action.ENTER_SHORT

    def _handle_in_trade(self, tick: MarketTick, regime: Regime) -> Action:
        # FORCED INVALIDATIONS — checked first, non-overridable
        if stop_hit(tick, self.position):
            return self._exit()
        if regime.state != 'positive':
            return self._exit()
        if tick.in_event_window:
            return self._exit()
        return Action.HOLD

    def _handle_standby(self, tick: MarketTick,
                         fut_levels: list[FutLevel]) -> Action:
        nearest = _nearest_tier1(tick.price, fut_levels, self.arm_distance_pts)
        if nearest is None:
            self.state = SignalState.IDLE
            self.armed_level = None
        return Action.NONE

    def _exit(self) -> Action:
        self.state = SignalState.IDLE
        self.position = None
        self.armed_level = None
        return Action.EXIT
```

- [ ] **Step 2: Update `signal_gpr/__init__.py`**

```python
from .models import SignalState, Action, MarketTick, TradePosition
from .trigger import ProxyConfirm
from .state_machine import GPRStateMachine

__all__ = ['SignalState', 'Action', 'MarketTick', 'TradePosition',
           'ProxyConfirm', 'GPRStateMachine']
```

- [ ] **Step 3: Append 12 state machine tests to `tests/test_signal.py`**

```python
# ── State machine transition tests ──────────────────────────────────────────

def test_idle_stays_idle_in_negative_regime():
    sm = GPRStateMachine(arm_distance_pts=10.0)
    action = sm.on_tick(NEAR_TICK, NEG_REGIME, ALL_LEVELS)
    assert sm.state == SignalState.IDLE
    assert action == Action.NONE


def test_idle_to_armed_in_positive_regime():
    sm = GPRStateMachine(arm_distance_pts=10.0)
    sm.on_tick(NEAR_TICK, POS_REGIME, ALL_LEVELS)
    assert sm.state == SignalState.ARMED


def test_armed_to_standby_on_regime_flip():
    sm = GPRStateMachine(arm_distance_pts=10.0)
    sm.on_tick(NEAR_TICK, POS_REGIME, ALL_LEVELS)   # → ARMED
    assert sm.state == SignalState.ARMED
    sm.on_tick(NEAR_TICK, NEG_REGIME, ALL_LEVELS)   # → STANDBY
    assert sm.state == SignalState.STANDBY


def test_armed_to_idle_when_price_leaves():
    sm = GPRStateMachine(arm_distance_pts=10.0)
    sm.on_tick(NEAR_TICK, POS_REGIME, ALL_LEVELS)   # → ARMED (near call_wall 5350)
    sm.on_tick(FAR_TICK, POS_REGIME, ALL_LEVELS)    # price=5200, far from all levels → IDLE
    assert sm.state == SignalState.IDLE


def test_armed_to_in_trade_on_confirmation():
    sm = GPRStateMachine(arm_distance_pts=10.0)
    sm.on_tick(NEAR_TICK, POS_REGIME, ALL_LEVELS)     # → ARMED
    action = sm.on_tick(CONFIRM_TICK, POS_REGIME, ALL_LEVELS)  # → IN_TRADE
    assert sm.state == SignalState.IN_TRADE
    assert action in (Action.ENTER_LONG, Action.ENTER_SHORT)


def test_armed_no_entry_without_confirmation():
    sm = GPRStateMachine(arm_distance_pts=10.0)
    sm.on_tick(NEAR_TICK, POS_REGIME, ALL_LEVELS)   # → ARMED
    action = sm.on_tick(NEAR_TICK, POS_REGIME, ALL_LEVELS)  # no rejection bar
    assert sm.state == SignalState.ARMED
    assert action == Action.NONE


def test_armed_no_entry_in_event_window():
    sm = GPRStateMachine(arm_distance_pts=10.0)
    sm.on_tick(NEAR_TICK, POS_REGIME, ALL_LEVELS)    # → ARMED
    action = sm.on_tick(EVENT_TICK, POS_REGIME, ALL_LEVELS)  # confirm=True but event window
    assert sm.state == SignalState.ARMED
    assert action == Action.NONE


def test_in_trade_exits_on_stop_hit():
    sm = GPRStateMachine(arm_distance_pts=10.0)
    sm.on_tick(NEAR_TICK, POS_REGIME, ALL_LEVELS)
    sm.on_tick(CONFIRM_TICK, POS_REGIME, ALL_LEVELS)   # → IN_TRADE (short from ~5348)
    assert sm.state == SignalState.IN_TRADE
    stop_price = sm.position.stop  # short stop is above entry
    # Tick above the stop triggers stop_hit for a short
    stop_tick = MarketTick(price=stop_price + 1.0, atr=10.0)
    action = sm.on_tick(stop_tick, POS_REGIME, ALL_LEVELS)
    assert action == Action.EXIT
    assert sm.state == SignalState.IDLE
    assert sm.position is None


def test_in_trade_exits_on_regime_flip():
    sm = GPRStateMachine(arm_distance_pts=10.0)
    sm.on_tick(NEAR_TICK, POS_REGIME, ALL_LEVELS)
    sm.on_tick(CONFIRM_TICK, POS_REGIME, ALL_LEVELS)  # → IN_TRADE
    action = sm.on_tick(NEAR_TICK, NEG_REGIME, ALL_LEVELS)  # regime flip
    assert action == Action.EXIT
    assert sm.state == SignalState.IDLE


# ── Hard invariant tests ─────────────────────────────────────────────────────

def test_invariant_no_arm_in_negative_regime():
    """Negative regime must never produce ARMED state across many ticks."""
    sm = GPRStateMachine(arm_distance_pts=50.0)  # very wide arm window
    for _ in range(100):
        sm.on_tick(NEAR_TICK, NEG_REGIME, ALL_LEVELS)
        assert sm.state != SignalState.ARMED, "ARMED in negative regime — invariant violated"


def test_invariant_regime_flip_always_exits():
    """Any in-trade position must exit immediately on regime flip."""
    sm = GPRStateMachine(arm_distance_pts=10.0)
    sm.on_tick(NEAR_TICK, POS_REGIME, ALL_LEVELS)
    sm.on_tick(CONFIRM_TICK, POS_REGIME, ALL_LEVELS)
    assert sm.state == SignalState.IN_TRADE
    action = sm.on_tick(NEAR_TICK, NEG_REGIME, ALL_LEVELS)
    assert action == Action.EXIT, "Regime flip did not force EXIT — invariant violated"
    assert sm.state == SignalState.IDLE


def test_invariant_no_entry_on_touch_without_confirm():
    """Touching a level without rejection bar must never produce an entry action."""
    sm = GPRStateMachine(arm_distance_pts=10.0)
    for _ in range(50):
        action = sm.on_tick(NEAR_TICK, POS_REGIME, ALL_LEVELS)  # no has_rejection_bar
        assert action not in (Action.ENTER_LONG, Action.ENTER_SHORT), \
            "Entry fired without confirmation — invariant violated"
```

- [ ] **Step 4: Run tests to verify they fail first**

```bash
cd D:/2026/GEX
python -m pytest tests/test_signal.py -k "idle or armed or in_trade or invariant" -v
```

Expected: `ImportError: cannot import name 'GPRStateMachine' from 'signal_gpr'`

- [ ] **Step 5: Run all signal tests**

```bash
cd D:/2026/GEX
python -m pytest tests/test_signal.py -v
```

Expected: 15 passed

- [ ] **Step 6: Run full suite**

```bash
cd D:/2026/GEX
python -m pytest -v
```

Expected: 85 total, all pass (70 + 15 new)

- [ ] **Step 7: Commit**

```bash
cd D:/2026/GEX
git add signal_gpr/state_machine.py signal_gpr/__init__.py tests/test_signal.py
git commit -m "feat: add GPRStateMachine — state machine with forced invalidations"
```

---

## Task 3: Final Verification + Push

- [ ] **Step 1: Verify public API**

```bash
cd D:/2026/GEX
python -c "from signal_gpr import GPRStateMachine, MarketTick, Action, SignalState, ProxyConfirm; print('OK')"
```

Expected: `OK`

- [ ] **Step 2: Run full test suite**

```bash
cd D:/2026/GEX
python -m pytest -v 2>&1 | tail -5
```

Expected: 85 passed

- [ ] **Step 3: Push**

```bash
cd D:/2026/GEX
git push
```

---

## Self-Review Checklist

**Spec coverage:**
- [x] SignalState enum: IDLE, ARMED, IN_TRADE, STANDBY → Task 1, models.py
- [x] Action enum: NONE, ENTER_LONG, ENTER_SHORT, EXIT, HOLD → Task 1, models.py
- [x] MarketTick: price, atr, has_rejection_bar, in_event_window → Task 1
- [x] TradePosition: direction, entry_price, stop, target, level_strength → Task 1
- [x] ProxyConfirm.confirmed(tick) → Task 1, trigger.py
- [x] compute_stop() ATR-scaled with floor → Task 1, risk.py + tests 13-14
- [x] select_target() nearest Tier-1 in direction → Task 1, risk.py + test 15
- [x] IDLE → ARMED in positive regime only → Task 2, tests 1-2
- [x] ARMED → STANDBY on regime flip → Task 2, test 3
- [x] ARMED → IDLE when price leaves zone → Task 2, test 4
- [x] ARMED → IN_TRADE on confirmation only → Task 2, tests 5-7
- [x] IN_TRADE exits on stop hit (forced) → Task 2, test 8
- [x] IN_TRADE exits on regime flip (forced) → Task 2, test 9
- [x] Invariant: no ARMED in negative regime → Task 2, test 10
- [x] Invariant: regime flip always exits → Task 2, test 11
- [x] Invariant: no entry on touch → Task 2, test 12
