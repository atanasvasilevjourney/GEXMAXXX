import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pandas as pd
from level_projection import Level, FutLevel, select_levels
# measure_basis, project added in Task 3

BY_STRIKE = pd.DataFrame([
    {'strike': 5100.0, 'gex': -150.0, 'vex': 0.0, 'chex': 0.0},
    {'strike': 5200.0, 'gex': -500.0, 'vex': 0.0, 'chex': 0.0},
    {'strike': 5250.0, 'gex': -200.0, 'vex': 0.0, 'chex': 0.0},
    {'strike': 5280.0, 'gex':  100.0, 'vex': 0.0, 'chex': 0.0},
    {'strike': 5300.0, 'gex':  900.0, 'vex': 0.0, 'chex': 0.0},
    {'strike': 5320.0, 'gex':  400.0, 'vex': 0.0, 'chex': 0.0},
    {'strike': 5350.0, 'gex':  700.0, 'vex': 0.0, 'chex': 0.0},
])

SNAPSHOT = {
    'spot': 5300.0,
    'call_wall': 5350.0,
    'put_wall':  5200.0,
    'gamma_flip': 5280.0,
    'hvl': 5200.0,
    'total_gex': 2250.0,
    'by_strike': BY_STRIKE,
}


def test_level_dataclass_fields():
    lv = Level(strike=5300.0, gex=900.0, tier=1, strength=1.0, label='pin')
    assert lv.strike == 5300.0
    assert lv.gex == 900.0
    assert lv.tier == 1
    assert lv.strength == 1.0
    assert lv.label == 'pin'


def test_futlevel_dataclass_fields():
    fl = FutLevel(fut_price=5320.0, tier=1, strength=1.0, label='pin',
                  source_strike=5300.0, basis_pts=20.0)
    assert fl.fut_price == 5320.0
    assert fl.source_strike == 5300.0
    assert fl.basis_pts == 20.0


def test_select_levels_always_includes_tier1():
    """call_wall, put_wall, pin, zero_gamma always present regardless of threshold."""
    levels = select_levels(SNAPSHOT, pct_threshold=0.99)  # very high threshold
    labels = {lv.label for lv in levels}
    assert 'call_wall' in labels
    assert 'put_wall' in labels
    assert 'pin' in labels
    assert 'zero_gamma' in labels


def test_select_levels_below_threshold_excluded():
    """Strike 5100 (strength=0.167) and 5250 (strength=0.222) excluded at default threshold."""
    levels = select_levels(SNAPSHOT, pct_threshold=0.30)
    strikes = {lv.strike for lv in levels}
    assert 5100.0 not in strikes
    assert 5250.0 not in strikes


def test_select_levels_above_threshold_included():
    """Strike 5320 (strength=0.444) included as Tier-2 cluster at default threshold."""
    levels = select_levels(SNAPSHOT, pct_threshold=0.30)
    strikes = {lv.strike for lv in levels}
    assert 5320.0 in strikes


def test_select_levels_tier_assignment():
    """Tier-1 = walls/pin/zero_gamma; clusters = Tier-2."""
    levels = select_levels(SNAPSHOT, pct_threshold=0.30)
    by_label = {lv.label: lv for lv in levels}
    assert by_label['call_wall'].tier == 1
    assert by_label['put_wall'].tier == 1
    assert by_label['pin'].tier == 1
    assert by_label['zero_gamma'].tier == 1
    assert by_label['cluster'].tier == 2


def test_select_levels_strength_normalised():
    """All strength values in [0, 1]; the max strength across levels == 1.0."""
    levels = select_levels(SNAPSHOT, pct_threshold=0.30)
    assert all(0.0 <= lv.strength <= 1.0 for lv in levels)
    assert max(lv.strength for lv in levels) == pytest.approx(1.0)


def test_select_levels_empty_snapshot():
    """Empty by_strike returns empty list without error."""
    empty_snap = {**SNAPSHOT, 'by_strike': pd.DataFrame()}
    result = select_levels(empty_snap, pct_threshold=0.30)
    assert result == []


def test_select_levels_no_call_wall():
    """call_wall=None handled gracefully; no call_wall Level in output."""
    snap = {**SNAPSHOT, 'call_wall': None}
    levels = select_levels(snap, pct_threshold=0.30)
    labels = [lv.label for lv in levels]
    assert 'call_wall' not in labels
    assert 'put_wall' in labels  # other Tier-1 still present
