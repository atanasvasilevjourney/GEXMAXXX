import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pandas as pd
from level_projection import Level, FutLevel
# select_levels, measure_basis, project added in Tasks 2-3

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
