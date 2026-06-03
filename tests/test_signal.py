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
