import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from backtest import replay, compute_stats, format_report, TradeRecord, BacktestResult
from signal_gpr import MarketTick
from level_projection import FutLevel
from regime import Regime

POS_REGIME = Regime(state='positive', distance_to_zero_gamma=50.0, conviction=0.8)
NEG_REGIME = Regime(state='negative', distance_to_zero_gamma=-50.0, conviction=0.8)

CALL_LEVEL = FutLevel(fut_price=5350.0, tier=1, strength=0.9, label='call_wall',
                      source_strike=5330.0, basis_pts=20.0)
PUT_LEVEL  = FutLevel(fut_price=5250.0, tier=1, strength=0.8, label='put_wall',
                      source_strike=5230.0, basis_pts=20.0)
ALL_LEVELS = [PUT_LEVEL, CALL_LEVEL]

BARS = [
    MarketTick(price=5345.0, atr=10.0),
    MarketTick(price=5348.0, atr=10.0, has_rejection_bar=True),
    MarketTick(price=5345.0, atr=10.0),
    MarketTick(price=5342.0, atr=10.0),
    MarketTick(price=5338.0, atr=10.0),
    MarketTick(price=5333.0, atr=10.0),
    MarketTick(price=5328.0, atr=10.0),
    MarketTick(price=5323.0, atr=10.0),
    MarketTick(price=5320.0, atr=10.0),
    MarketTick(price=5318.0, atr=10.0),
]

def SNAPSHOTS_FN(i: int):
    if i <= 7:
        return POS_REGIME, ALL_LEVELS
    return NEG_REGIME, ALL_LEVELS


def test_trade_record_fields():
    tr = TradeRecord(entry_bar=1, exit_bar=8, direction='short',
                     entry_price=5348.0, exit_price=5320.0, pnl_pts=28.0,
                     regime_at_entry='positive', level_tier=1, level_strength=0.9)
    assert tr.entry_bar == 1
    assert tr.exit_bar == 8
    assert tr.direction == 'short'
    assert tr.pnl_pts == 28.0
    assert tr.regime_at_entry == 'positive'


def test_backtest_result_fields():
    result = BacktestResult(
        total_trades=1, positive_regime_trades=1, negative_regime_trades=0,
        positive_regime_pnl=28.0, negative_regime_pnl=0.0,
        positive_regime_sharpe=0.0, negative_regime_sharpe=0.0,
        oos_positive_pnl=0.0, regime_split_significant=True, signal_only=False,
    )
    assert result.total_trades == 1
    assert result.signal_only == False
    assert result.regime_split_significant == True


def test_compute_stats_separates_regime_pnl():
    trades = [
        TradeRecord(entry_bar=0, exit_bar=1, direction='short',
                    entry_price=100.0, exit_price=90.0, pnl_pts=10.0,
                    regime_at_entry='positive', level_tier=1, level_strength=0.9),
        TradeRecord(entry_bar=2, exit_bar=3, direction='short',
                    entry_price=100.0, exit_price=105.0, pnl_pts=-5.0,
                    regime_at_entry='negative', level_tier=1, level_strength=0.8),
    ]
    # Use oos_start_pct=1.0 so all trades are in-sample (cutoff will be >= max_bar)
    result = compute_stats(trades, oos_start_pct=1.0)
    assert result.positive_regime_pnl == pytest.approx(10.0)
    assert result.negative_regime_pnl == pytest.approx(-5.0)
    assert result.positive_regime_trades == 1
    assert result.negative_regime_trades == 1


def test_signal_only_true_when_no_positive_edge():
    # All trades in positive regime but losing → signal_only=True
    trades = [
        TradeRecord(entry_bar=0, exit_bar=1, direction='short',
                    entry_price=100.0, exit_price=105.0, pnl_pts=-5.0,
                    regime_at_entry='positive', level_tier=1, level_strength=0.9),
    ]
    result = compute_stats(trades)
    assert result.signal_only == True


def test_signal_only_false_when_edge_validated():
    # Multiple winning positive-regime trades (IS + OOS profitable)
    trades = [
        TradeRecord(entry_bar=i, exit_bar=i+1, direction='short',
                    entry_price=100.0, exit_price=90.0, pnl_pts=10.0,
                    regime_at_entry='positive', level_tier=1, level_strength=0.9)
        for i in range(0, 20, 2)
    ]
    result = compute_stats(trades)
    assert result.signal_only == False
    assert result.positive_regime_pnl > 0


def test_format_report_contains_signal_only_line():
    result = BacktestResult(
        total_trades=5, positive_regime_trades=4, negative_regime_trades=1,
        positive_regime_pnl=40.0, negative_regime_pnl=-5.0,
        positive_regime_sharpe=1.5, negative_regime_sharpe=-0.5,
        oos_positive_pnl=12.0, regime_split_significant=True, signal_only=False,
    )
    report = format_report(result)
    assert 'SIGNAL_ONLY' in report
    assert 'EDGE VALIDATED' in report
