import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from regime import Regime

POSITIVE_SNAP = {
    'spot': 5300.0,
    'gamma_flip': 5250.0,
    'total_gex': 1_500_000_000.0,
}

NEGATIVE_SNAP = {
    'spot': 5200.0,
    'gamma_flip': 5250.0,
    'total_gex': -800_000_000.0,
}

AT_FLIP_SNAP = {
    'spot': 5250.0,
    'gamma_flip': 5250.0,
    'total_gex': 100_000.0,
}

FAR_SNAP = {
    'spot': 5300.0,
    'gamma_flip': 5000.0,
    'total_gex': 2_000_000_000.0,
}


def test_regime_dataclass_fields():
    r = Regime(state='positive', distance_to_zero_gamma=50.0, conviction=0.5)
    assert r.state == 'positive'
    assert r.distance_to_zero_gamma == 50.0
    assert r.conviction == 0.5
