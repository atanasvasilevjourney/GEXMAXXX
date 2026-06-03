import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pandas as pd
from datetime import date
from unittest.mock import patch
from greeks import calculate_all_greeks
from levels import find_levels, split_0dte, get_all_levels, add_futures_conversion

SPOT = 540.0


def make_processed_chain():
    df = pd.DataFrame([
        {'strike': 545.0, 'type': 'call', 'openInterest': 10000, 'impliedVolatility': 0.20, 'expiration': '2026-06-20'},
        {'strike': 550.0, 'type': 'call', 'openInterest': 5000,  'impliedVolatility': 0.18, 'expiration': '2026-06-20'},
        {'strike': 535.0, 'type': 'put',  'openInterest': 8000,  'impliedVolatility': 0.22, 'expiration': '2026-06-20'},
        {'strike': 530.0, 'type': 'put',  'openInterest': 3000,  'impliedVolatility': 0.25, 'expiration': '2026-06-20'},
    ])
    return calculate_all_greeks(df, SPOT)


def test_find_levels_call_wall_above_spot():
    levels = find_levels(make_processed_chain(), SPOT)
    assert levels['call_wall'] > SPOT


def test_find_levels_put_wall_below_spot():
    levels = find_levels(make_processed_chain(), SPOT)
    assert levels['put_wall'] < SPOT


def test_find_levels_put_wall_excludes_deep_otm():
    """Deep-OTM put (>20% below spot) with huge OI must not become put wall."""
    df = pd.DataFrame([
        {'strike': 545.0, 'type': 'call', 'openInterest': 10000, 'impliedVolatility': 0.20, 'expiration': '2026-06-20'},
        {'strike': 535.0, 'type': 'put',  'openInterest': 8000,  'impliedVolatility': 0.22, 'expiration': '2026-06-20'},
        # Deep OTM put at 30% below spot (like stale QQQ $450 vs $540 spot)
        {'strike': 378.0, 'type': 'put',  'openInterest': 500000, 'impliedVolatility': 0.50, 'expiration': '2026-06-20'},
    ])
    df = calculate_all_greeks(df, SPOT)
    levels = find_levels(df, SPOT)
    # Put wall must be within 20% of spot, not at 378
    assert levels['put_wall'] is None or levels['put_wall'] >= SPOT * 0.80


def test_find_levels_has_vanna_charm_keys():
    levels = find_levels(make_processed_chain(), SPOT)
    for key in ('vanna_wall', 'charm_wall', 'total_vex', 'total_chex'):
        assert key in levels, f"Missing key: {key}"


def test_find_levels_by_strike_has_greek_columns():
    levels = find_levels(make_processed_chain(), SPOT)
    assert 'by_strike' in levels
    bs = levels['by_strike']
    for col in ('strike', 'gex', 'vex', 'chex'):
        assert col in bs.columns, f"by_strike missing column: {col}"


def test_find_levels_returns_none_for_empty_df():
    result = find_levels(pd.DataFrame(), SPOT)
    assert result is None


def test_split_0dte_puts_today_in_0dte():
    today = date.today().isoformat()
    df = pd.DataFrame([
        {'strike': 540.0, 'type': 'call', 'openInterest': 100, 'impliedVolatility': 0.20, 'expiration': today},
        {'strike': 540.0, 'type': 'call', 'openInterest': 200, 'impliedVolatility': 0.20, 'expiration': '2026-07-20'},
    ])
    df_0dte, df_multi = split_0dte(df)
    assert len(df_0dte) == 1
    assert len(df_multi) == 1
    assert df_0dte.iloc[0]['expiration'] == today


def test_split_0dte_no_0dte_returns_empty():
    df = pd.DataFrame([
        {'strike': 540.0, 'type': 'call', 'openInterest': 100, 'impliedVolatility': 0.20, 'expiration': '2026-07-20'},
    ])
    df_0dte, df_multi = split_0dte(df)
    assert len(df_0dte) == 0
    assert len(df_multi) == 1


def test_get_all_levels_has_three_keys():
    result = get_all_levels(make_processed_chain(), SPOT)
    assert 'all' in result
    assert '0dte' in result
    assert 'multi' in result


def test_get_all_levels_0dte_is_none_when_no_0dte():
    result = get_all_levels(make_processed_chain(), SPOT)
    assert result['0dte'] is None  # test chain has no today-expiry options


def test_add_futures_conversion_qqq_symbol():
    levels = get_all_levels(make_processed_chain(), SPOT)
    with patch('levels.get_futures_price', return_value=21500.0):
        result = add_futures_conversion(levels, 'QQQ', SPOT)
    assert result['futures']['symbol'] == 'NQ'
    assert result['futures']['multiplier'] == round(21500.0 / SPOT, 2)


def test_add_futures_conversion_spy_symbol():
    levels = get_all_levels(make_processed_chain(), SPOT)
    with patch('levels.get_futures_price', return_value=5400.0):
        result = add_futures_conversion(levels, 'SPY', SPOT)
    assert result['futures']['symbol'] == 'ES'


def test_add_futures_conversion_has_level_fields():
    levels = get_all_levels(make_processed_chain(), SPOT)
    with patch('levels.get_futures_price', return_value=21500.0):
        result = add_futures_conversion(levels, 'QQQ', SPOT)
    for key in ('spot', 'call_wall', 'put_wall', 'gamma_flip', 'hvl'):
        assert key in result['futures'], f"Missing futures key: {key}"
