import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from signal_gpr import (
    # GPRStateMachine added in Task 2
    MarketTick, TradePosition,
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
