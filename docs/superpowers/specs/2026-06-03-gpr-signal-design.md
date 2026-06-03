# GPR Signal — Design Spec
**Date:** 2026-06-03
**Status:** Approved
**Sub-project:** 3 of 6 — Gamma-Pin Reversion (GPR) strategy

## Overview

Add a `signal/` package to `D:/2026/GEX`. It implements the GPR state machine (Part A.4) in **paper/replay mode only** — no live execution. Consumes `Regime` (from `regime/`) and `[FutLevel]` (from `level_projection/`) plus a `MarketTick` on each bar. Returns an `Action`.

Three hard invariants (must be impossible to violate):
1. Fades never arm in negative gamma
2. A regime flip against an open position ALWAYS forces exit
3. Entry fires only on trigger confirmation, not on touch

---

## File Structure

```
D:/2026/GEX/
├── signal/
│   ├── __init__.py       # exports: GPRStateMachine, MarketTick, TradePosition, SignalState, Action, ProxyConfirm
│   ├── models.py         # SignalState, Action (enums), MarketTick, TradePosition (dataclasses)
│   ├── trigger.py        # ProxyConfirm — pluggable confirmation interface
│   ├── risk.py           # compute_stop(), select_target()
│   └── state_machine.py  # GPRStateMachine.on_tick()
└── tests/
    └── test_signal.py
```

---

## Module Specifications

### `models.py`

```python
from enum import Enum
from dataclasses import dataclass, field


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
    has_rejection_bar: bool = False   # proxy trigger: bar rejected level, closed back inside
    in_event_window: bool = False     # True during macro event blackout (Fed/CPI/etc.)


@dataclass
class TradePosition:
    direction: str        # "long" | "short"
    entry_price: float
    stop: float
    target: float         # next significant strike in trade direction
    level_strength: float # 0–1 from FutLevel.strength
```

---

### `trigger.py`

**Pluggable trigger interface** (paper/replay proxy — no footprint feed needed):

```python
class ProxyConfirm:
    """
    Automation proxy for order-flow confirmation.
    Fires on rejection-reclaim bar: price wicks through the level then closes back inside.
    In paper/replay mode, this is signalled by MarketTick.has_rejection_bar.
    """
    def confirmed(self, tick: MarketTick) -> bool:
        return tick.has_rejection_bar
```

Design is open for extension: a `FootprintConfirm` class with the same `confirmed(tick)` interface can be swapped in without changing the state machine.

---

### `risk.py`

**`compute_stop(entry, direction, atr, k_atr=0.75, min_ticks=12.0) -> float`**
```
buffer = max(k_atr * atr, min_ticks)
long:  stop = entry - buffer
short: stop = entry + buffer
```

**`select_target(entry, direction, fut_levels) -> float | None`**
- For `"long"`: find the nearest Tier-1 `FutLevel` with `fut_price > entry`
- For `"short"`: find the nearest Tier-1 `FutLevel` with `fut_price < entry`
- Returns `None` if no suitable level found

**`stop_hit(tick, position) -> bool`**
```
long:  tick.price <= position.stop
short: tick.price >= position.stop
```

---

### `state_machine.py`

**`GPRStateMachine`**

```python
class GPRStateMachine:
    def __init__(self, arm_distance_pts: float = 5.0, trigger: ProxyConfirm | None = None,
                 k_atr: float = 0.75, min_ticks: float = 12.0):
        self.state: SignalState = SignalState.IDLE
        self.position: TradePosition | None = None
        self.armed_level: FutLevel | None = None
        self.arm_distance_pts = arm_distance_pts
        self.trigger = trigger or ProxyConfirm()
        self.k_atr = k_atr
        self.min_ticks = min_ticks

    def on_tick(self, tick: MarketTick, regime: Regime, fut_levels: list[FutLevel]) -> Action:
        ...
```

**State transitions:**

```
IDLE
  └─ regime.positive AND nearest Tier-1 within arm_distance_pts → ARMED (store armed_level)

ARMED
  ├─ regime.negative → STANDBY (armed_level cleared)
  ├─ trigger.confirmed AND NOT in_event_window → IN_TRADE (open position, return ENTER_LONG/SHORT)
  ├─ nearest Tier-1 no longer within arm_distance_pts → IDLE
  └─ else → NONE

IN_TRADE  [forced invalidations — non-overridable, checked FIRST before any other logic]
  ├─ stop_hit(tick, position) → EXIT → IDLE
  ├─ regime.negative (flip against position) → EXIT → IDLE
  ├─ tick.in_event_window → EXIT → IDLE
  └─ else → HOLD

STANDBY
  └─ nearest Tier-1 no longer within arm_distance_pts → IDLE
  └─ else → NONE (no entries, no exits)
```

**Fade direction** (at armed_level):
- `armed_level.fut_price > tick.price` → fade = `"short"` (selling resistance above)
- `armed_level.fut_price < tick.price` → fade = `"long"` (buying support below)
- If equal → no trade

**Opening a position (ARMED → IN_TRADE):**
```python
direction = "short" if armed_level.fut_price > tick.price else "long"
stop   = compute_stop(tick.price, direction, tick.atr, self.k_atr, self.min_ticks)
target = select_target(tick.price, direction, fut_levels)
self.position = TradePosition(
    direction=direction,
    entry_price=tick.price,
    stop=stop,
    target=target,
    level_strength=armed_level.strength,
)
```

**Exiting (IN_TRADE → IDLE):**
```python
self.state = SignalState.IDLE
self.position = None
self.armed_level = None
return Action.EXIT
```

---

## Tests (`tests/test_signal.py`)

All tests use synthetic ticks and fixtures — no network calls.

**15 tests total:**

**State machine transitions (9):**
1. `test_idle_stays_idle_in_negative_regime` — negative regime → stays IDLE
2. `test_idle_to_armed_in_positive_regime` — positive regime, price near Tier-1 → ARMED
3. `test_armed_to_standby_on_regime_flip` — flip to negative while ARMED → STANDBY
4. `test_armed_to_idle_when_price_leaves` — price moves away → IDLE
5. `test_armed_to_in_trade_on_confirmation` — positive regime, rejection bar → IN_TRADE, returns ENTER action
6. `test_armed_no_entry_without_confirmation` — positive regime, no rejection bar → stays ARMED
7. `test_armed_no_entry_in_event_window` — confirmation True but in_event_window → stays ARMED (not IN_TRADE)
8. `test_in_trade_exits_on_stop_hit` — stop crossed → EXIT, state=IDLE
9. `test_in_trade_exits_on_regime_flip` — regime flips negative while in trade → EXIT (forced)

**Hard invariants (3):**
10. `test_invariant_no_arm_in_negative_regime` — negative regime never produces ARMED state across 100 ticks near levels
11. `test_invariant_regime_flip_always_exits` — ANY in-trade position exits immediately on regime flip, no exceptions
12. `test_invariant_no_entry_on_touch_without_confirm` — touching a level with no rejection bar never triggers entry

**Risk functions (3):**
13. `test_compute_stop_long` — long stop = entry - max(k_atr * atr, min_ticks)
14. `test_compute_stop_short` — short stop = entry + max(k_atr * atr, min_ticks)
15. `test_select_target_nearest_tier1` — returns nearest Tier-1 in direction

---

## Acceptance Criteria

- All 15 tests pass
- Invariant tests 10-12 pass with zero exceptions across all replayed sequences
- `state_machine.on_tick` returns `Action` enum value, never raises
- In-trade forced exit fires BEFORE any other logic in IN_TRADE handling (checked first)
- No hardcoded tick/ATR values in state machine (all from `MarketTick`)

---

## Relationship to GPR Sub-projects

| Sub-project | Uses signal output |
|---|---|
| 1 — level_projection | Provides FutLevel inputs |
| 2 — regime | Provides Regime inputs |
| 3 — signal (this) | Produces Action |
| 4 — backtest | Replays signal in walk-forward harness |
| 5 — execution | Converts Action to broker orders (post-validation gate) |
