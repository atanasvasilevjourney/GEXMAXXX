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
