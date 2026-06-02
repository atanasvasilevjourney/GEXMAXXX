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
