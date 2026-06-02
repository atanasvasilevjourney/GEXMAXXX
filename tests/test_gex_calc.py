import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pandas as pd
import numpy as np
from gex_calc import calculate_gex, find_levels, print_levels


def make_chain():
    """Minimal synthetic options chain for testing. Spot = 540.0"""
    return pd.DataFrame([
        {'strike': 545.0, 'type': 'call', 'openInterest': 10000, 'impliedVolatility': 0.20, 'expiration': '2026-06-20'},
        {'strike': 550.0, 'type': 'call', 'openInterest': 5000,  'impliedVolatility': 0.18, 'expiration': '2026-06-20'},
        {'strike': 535.0, 'type': 'put',  'openInterest': 8000,  'impliedVolatility': 0.22, 'expiration': '2026-06-20'},
        {'strike': 530.0, 'type': 'put',  'openInterest': 3000,  'impliedVolatility': 0.25, 'expiration': '2026-06-20'},
    ])

SPOT = 540.0


def test_calculate_gex_calls_positive():
    df = make_chain()
    result = calculate_gex(df, SPOT)
    calls = result[result['type'] == 'call']
    assert (calls['gex'] > 0).all(), "Call GEX must be positive (dealers short calls = positive exposure)"


def test_calculate_gex_puts_negative():
    df = make_chain()
    result = calculate_gex(df, SPOT)
    puts = result[result['type'] == 'put']
    assert (puts['gex'] < 0).all(), "Put GEX must be negative (dealers long puts = negative exposure)"


def test_calculate_gex_iv_fallback():
    df = make_chain()
    df.loc[0, 'impliedVolatility'] = 0  # force IV=0 fallback
    result = calculate_gex(df, SPOT)
    assert not np.isnan(result.loc[0, 'gamma']), "Gamma must not be NaN when IV=0 (fallback to 0.20)"
    assert result.loc[0, 'gamma'] > 0


def test_calculate_gex_nan_iv_fallback():
    df = make_chain()
    df.loc[1, 'impliedVolatility'] = float('nan')
    result = calculate_gex(df, SPOT)
    assert not np.isnan(result.loc[1, 'gamma'])


def test_calculate_gex_adds_columns():
    df = make_chain()
    result = calculate_gex(df, SPOT)
    assert 'gamma' in result.columns
    assert 'gex' in result.columns


def test_find_levels_call_wall_above_spot():
    df = calculate_gex(make_chain(), SPOT)
    levels = find_levels(df, SPOT)
    assert levels['call_wall'] > SPOT, f"Call wall {levels['call_wall']} must be above spot {SPOT}"


def test_find_levels_put_wall_below_spot():
    df = calculate_gex(make_chain(), SPOT)
    levels = find_levels(df, SPOT)
    assert levels['put_wall'] < SPOT, f"Put wall {levels['put_wall']} must be below spot {SPOT}"


def test_find_levels_returns_all_keys():
    df = calculate_gex(make_chain(), SPOT)
    levels = find_levels(df, SPOT)
    required_keys = ('spot', 'call_wall', 'put_wall', 'gamma_flip', 'hvl', 'total_gex', 'regime')
    for key in required_keys:
        assert key in levels, f"Missing key: {key}"


def test_find_levels_regime_positive():
    """With more call OI than put OI, total GEX should be positive."""
    df = pd.DataFrame([
        {'strike': 545.0, 'type': 'call', 'openInterest': 50000, 'impliedVolatility': 0.20, 'expiration': '2026-06-20'},
        {'strike': 530.0, 'type': 'put',  'openInterest': 1000,  'impliedVolatility': 0.20, 'expiration': '2026-06-20'},
    ])
    df = calculate_gex(df, SPOT)
    levels = find_levels(df, SPOT)
    assert levels['regime'] == 'POSITIVE (low vol)'


def test_find_levels_regime_negative():
    """With heavy put OI, total GEX should be negative."""
    df = pd.DataFrame([
        {'strike': 545.0, 'type': 'call', 'openInterest': 100,   'impliedVolatility': 0.20, 'expiration': '2026-06-20'},
        {'strike': 530.0, 'type': 'put',  'openInterest': 50000, 'impliedVolatility': 0.20, 'expiration': '2026-06-20'},
    ])
    df = calculate_gex(df, SPOT)
    levels = find_levels(df, SPOT)
    assert levels['regime'] == 'NEGATIVE (high vol)'


def test_find_levels_no_calls_above_spot():
    """call_wall should be None when no calls exist above spot."""
    df = pd.DataFrame([
        {'strike': 530.0, 'type': 'put', 'openInterest': 5000, 'impliedVolatility': 0.20, 'expiration': '2026-06-20'},
    ])
    df = calculate_gex(df, SPOT)
    levels = find_levels(df, SPOT)
    assert levels['call_wall'] is None


def test_find_levels_no_puts_below_spot():
    """put_wall should be None when no puts exist below spot."""
    df = pd.DataFrame([
        {'strike': 550.0, 'type': 'call', 'openInterest': 5000, 'impliedVolatility': 0.20, 'expiration': '2026-06-20'},
    ])
    df = calculate_gex(df, SPOT)
    levels = find_levels(df, SPOT)
    assert levels['put_wall'] is None


def test_print_levels_contains_ticker(capsys):
    levels = {
        'spot': SPOT, 'call_wall': 545.0, 'put_wall': 535.0,
        'gamma_flip': 538.0, 'hvl': 530.0,
        'total_gex': 4.21e9, 'regime': 'POSITIVE (low vol)'
    }
    print_levels('SPY', levels)
    captured = capsys.readouterr()
    assert 'SPY' in captured.out


def test_print_levels_none_safe(capsys):
    """Must not raise when call_wall or put_wall is None."""
    levels = {
        'spot': SPOT, 'call_wall': None, 'put_wall': None,
        'gamma_flip': 538.0, 'hvl': 530.0,
        'total_gex': -1.5e9, 'regime': 'NEGATIVE (high vol)'
    }
    print_levels('QQQ', levels)
    captured = capsys.readouterr()
    assert 'N/A' in captured.out
    assert 'QQQ' in captured.out


def test_print_levels_shows_regime(capsys):
    levels = {
        'spot': SPOT, 'call_wall': 545.0, 'put_wall': 535.0,
        'gamma_flip': 538.0, 'hvl': 530.0,
        'total_gex': 2e9, 'regime': 'POSITIVE (low vol)'
    }
    print_levels('SPY', levels)
    captured = capsys.readouterr()
    assert 'POSITIVE' in captured.out


def test_print_levels_shows_gex_in_billions(capsys):
    levels = {
        'spot': SPOT, 'call_wall': 545.0, 'put_wall': 535.0,
        'gamma_flip': 538.0, 'hvl': 530.0,
        'total_gex': 4210000000.0, 'regime': 'POSITIVE (low vol)'
    }
    print_levels('SPY', levels)
    captured = capsys.readouterr()
    assert '4.21' in captured.out
