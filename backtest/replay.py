from __future__ import annotations
from typing import Callable
from signal_gpr import GPRStateMachine, Action, MarketTick
from level_projection import FutLevel
from regime import Regime
from .models import TradeRecord


def replay(
    bars: list[MarketTick],
    snapshots_fn: Callable[[int], tuple[Regime, list[FutLevel]]],
    arm_distance_pts: float = 5.0,
    k_atr: float = 0.75,
    min_ticks: float = 12.0,
) -> list[TradeRecord]:
    """
    Replay a sequence of MarketTicks through GPRStateMachine.

    Args:
        bars:             list of MarketTick (one per bar)
        snapshots_fn:     bar_index → (Regime, list[FutLevel])
        arm_distance_pts: passed to GPRStateMachine
        k_atr, min_ticks: stop parameters

    Returns:
        list of TradeRecord (one per completed trade)
    """
    sm = GPRStateMachine(arm_distance_pts=arm_distance_pts, k_atr=k_atr, min_ticks=min_ticks)
    trades: list[TradeRecord] = []

    # State tracked between bars for open position
    open_trade: dict | None = None

    for i, tick in enumerate(bars):
        regime, fut_levels = snapshots_fn(i)
        action = sm.on_tick(tick, regime, fut_levels)

        if action in (Action.ENTER_LONG, Action.ENTER_SHORT):
            open_trade = {
                'entry_bar':      i,
                'entry_price':    tick.price,
                'direction':      sm.position.direction,
                'regime_at_entry': regime.state,
                'level_tier':     sm.armed_level.tier if sm.armed_level else 1,
                'level_strength': sm.position.level_strength,
            }

        elif action == Action.EXIT and open_trade is not None:
            direction   = open_trade['direction']
            entry_price = open_trade['entry_price']
            exit_price  = tick.price
            pnl = (exit_price - entry_price) if direction == 'long' else (entry_price - exit_price)
            trades.append(TradeRecord(
                entry_bar=open_trade['entry_bar'],
                exit_bar=i,
                direction=direction,
                entry_price=entry_price,
                exit_price=exit_price,
                pnl_pts=pnl,
                regime_at_entry=open_trade['regime_at_entry'],
                level_tier=open_trade['level_tier'],
                level_strength=open_trade['level_strength'],
            ))
            open_trade = None

    # Close any open trade at end of bars
    if open_trade is not None and bars:
        last_tick   = bars[-1]
        direction   = open_trade['direction']
        entry_price = open_trade['entry_price']
        exit_price  = last_tick.price
        pnl = (exit_price - entry_price) if direction == 'long' else (entry_price - exit_price)
        trades.append(TradeRecord(
            entry_bar=open_trade['entry_bar'],
            exit_bar=len(bars) - 1,
            direction=direction,
            entry_price=entry_price,
            exit_price=exit_price,
            pnl_pts=pnl,
            regime_at_entry=open_trade['regime_at_entry'],
            level_tier=open_trade['level_tier'],
            level_strength=open_trade['level_strength'],
        ))

    return trades
