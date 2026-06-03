import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from regime import Regime, classify

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


def test_classify_positive_when_net_gex_positive():
    r = classify(POSITIVE_SNAP)
    assert r.state == 'positive'


def test_classify_negative_when_net_gex_negative():
    r = classify(NEGATIVE_SNAP)
    assert r.state == 'negative'


def test_classify_negative_when_net_gex_zero():
    snap = {**POSITIVE_SNAP, 'total_gex': 0.0}
    r = classify(snap)
    assert r.state == 'negative'


def test_classify_distance_positive_when_spot_above_flip():
    r = classify(POSITIVE_SNAP)
    # spot=5300, gamma_flip=5250 -> distance = +50
    assert r.distance_to_zero_gamma == pytest.approx(50.0)


def test_classify_distance_negative_when_spot_below_flip():
    r = classify(NEGATIVE_SNAP)
    # spot=5200, gamma_flip=5250 -> distance = -50
    assert r.distance_to_zero_gamma == pytest.approx(-50.0)


def test_classify_distance_zero_at_flip():
    r = classify(AT_FLIP_SNAP)
    assert r.distance_to_zero_gamma == pytest.approx(0.0)


def test_classify_conviction_zero_at_flip():
    r = classify(AT_FLIP_SNAP)
    assert r.conviction == pytest.approx(0.0)


def test_classify_conviction_increases_with_distance():
    # FAR_SNAP: 300 pts away. POSITIVE_SNAP: 50 pts away. FAR should have higher conviction.
    r_near = classify(POSITIVE_SNAP)
    r_far  = classify(FAR_SNAP)
    assert r_far.conviction > r_near.conviction


def test_classify_conviction_capped_at_1():
    # FAR_SNAP: distance=300, spot=5300, scale=0.05 -> full_dist=265 -> 300/265 > 1 -> capped at 1.0
    r = classify(FAR_SNAP)
    assert r.conviction == pytest.approx(1.0)
