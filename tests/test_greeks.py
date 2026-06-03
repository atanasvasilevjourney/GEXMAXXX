import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pandas as pd
import numpy as np
from greeks import calculate_gex, calculate_vanna, calculate_charm, calculate_all_greeks

SPOT = 540.0


def make_chain():
    """Synthetic options chain for testing. Spot = 540.0"""
    return pd.DataFrame([
        {'strike': 545.0, 'type': 'call', 'openInterest': 10000, 'impliedVolatility': 0.20, 'expiration': '2026-06-20'},
        {'strike': 550.0, 'type': 'call', 'openInterest': 5000,  'impliedVolatility': 0.18, 'expiration': '2026-06-20'},
        {'strike': 535.0, 'type': 'put',  'openInterest': 8000,  'impliedVolatility': 0.22, 'expiration': '2026-06-20'},
        {'strike': 530.0, 'type': 'put',  'openInterest': 3000,  'impliedVolatility': 0.25, 'expiration': '2026-06-20'},
    ])


def test_calculate_gex_calls_positive():
    result = calculate_gex(make_chain(), SPOT)
    assert (result[result['type'] == 'call']['gex'] > 0).all()


def test_calculate_gex_puts_negative():
    result = calculate_gex(make_chain(), SPOT)
    assert (result[result['type'] == 'put']['gex'] < 0).all()


def test_calculate_gex_stores_d1_d2():
    result = calculate_gex(make_chain(), SPOT)
    assert 'd1' in result.columns
    assert 'd2' in result.columns
    assert not result['d1'].isna().any()
    assert not result['d2'].isna().any()


def test_calculate_gex_iv_fallback():
    df = make_chain()
    df.loc[0, 'impliedVolatility'] = 0
    result = calculate_gex(df, SPOT)
    assert not np.isnan(result.loc[0, 'gamma'])
    assert result.loc[0, 'gamma'] > 0


def test_calculate_vanna_adds_vex():
    df = calculate_gex(make_chain(), SPOT)
    result = calculate_vanna(df, SPOT)
    assert 'vex' in result.columns
    assert not result['vex'].isna().any()


def test_calculate_vanna_puts_negated():
    df = calculate_gex(make_chain(), SPOT)
    result = calculate_vanna(df, SPOT)
    calls_vex = result[result['type'] == 'call']['vex'].values
    puts_vex = result[result['type'] == 'put']['vex'].values
    assert not (np.sign(calls_vex[0]) == np.sign(puts_vex[0]) and
                abs(calls_vex[0]) == abs(puts_vex[0])), "Put VEX must be negated vs call VEX at same strike"


def test_calculate_charm_adds_chex():
    df = calculate_gex(make_chain(), SPOT)
    result = calculate_charm(df, SPOT)
    assert 'chex' in result.columns
    assert not result['chex'].isna().any()


def test_calculate_all_greeks_adds_all_columns():
    result = calculate_all_greeks(make_chain(), SPOT)
    for col in ['d1', 'd2', 'gamma', 'gex', 'vanna', 'vex', 'charm', 'chex']:
        assert col in result.columns, f"Missing column: {col}"


def test_calculate_all_greeks_does_not_modify_input():
    df = make_chain()
    original_cols = set(df.columns)
    calculate_all_greeks(df, SPOT)
    assert set(df.columns) == original_cols, "Input DataFrame must not be mutated"
