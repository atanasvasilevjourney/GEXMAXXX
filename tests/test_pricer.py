import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import pytest
from datetime import date, timedelta

from pricer import PricerStyle, detect_style, time_to_expiry, MIN_T_YEARS
from pricer.engines import ql_european_gamma, ql_american_gamma, ql_black76_gamma
from pricer import compute_gamma


# --- Style detection ---

def test_detect_style_european():
    assert detect_style('SPX')  == PricerStyle.EUROPEAN
    assert detect_style('^SPX') == PricerStyle.EUROPEAN
    assert detect_style('NDX')  == PricerStyle.EUROPEAN
    assert detect_style('^NDX') == PricerStyle.EUROPEAN


def test_detect_style_american():
    assert detect_style('QQQ') == PricerStyle.AMERICAN
    assert detect_style('SPY') == PricerStyle.AMERICAN
    assert detect_style('IWM') == PricerStyle.AMERICAN


def test_detect_style_black76():
    assert detect_style('ES=F') == PricerStyle.BLACK76
    assert detect_style('NQ=F') == PricerStyle.BLACK76
    assert detect_style('MNQ=F') == PricerStyle.BLACK76


# --- Time to expiry ---

def test_time_to_expiry_far_expiry():
    # 30 days out → T ≈ 30/365.25 ≈ 0.082
    future = (date.today() + timedelta(days=30)).strftime('%Y-%m-%d')
    T = time_to_expiry(future)
    assert 0.07 < T < 0.10


def test_time_to_expiry_floor():
    # Already-expired date → T must be MIN_T_YEARS (60-sec floor, not negative)
    T = time_to_expiry('2000-01-01')
    assert T == pytest.approx(MIN_T_YEARS)


# --- Engine tests ---

def test_european_gamma_known_value():
    # ATM, T=0.25yr, σ=0.20, r=0.05, S=K=100
    # d1 = (ln(1) + (0.05+0.02)*0.25) / (0.20*0.5) = 0.175
    # gamma = N'(0.175) / (100*0.20*0.5) ≈ 0.0393
    g = ql_european_gamma(100.0, 100.0, 0.25, 0.05, 0.20, 'call')
    assert g == pytest.approx(0.0393, abs=0.002)


def test_american_gamma_gte_european_put():
    # ATM put, 30-day: American put >= European put.
    # Puts always have nonzero early exercise premium.
    # Calls without dividends: American == European, so test uses put.
    S, K, T, r, sigma = 100.0, 100.0, 30 / 365.25, 0.05, 0.20
    g_eu = ql_european_gamma(S, K, T, r, sigma, 'put')
    g_am = ql_american_gamma(S, K, T, r, sigma, 'put')
    assert g_am >= g_eu


def test_black76_gamma_matches_formula():
    # Black-76 gamma: e^{-rT} * N'(d1) / (F * sigma * sqrt(T))
    # For ATM Black-76: d1 = 0.5 * sigma * sqrt(T)
    F, K, T, r, sigma = 100.0, 100.0, 0.25, 0.05, 0.20
    d1 = 0.5 * sigma * math.sqrt(T)
    expected = math.exp(-r * T) * math.exp(-0.5 * d1 ** 2) / math.sqrt(2 * math.pi) / (F * sigma * math.sqrt(T))
    g = ql_black76_gamma(F, K, T, r, sigma, 'call')
    assert g == pytest.approx(expected, rel=0.01)


def test_compute_gamma_dispatch_returns_positive():
    # Smoke test: compute_gamma returns a positive float for each style
    for style in PricerStyle:
        g = compute_gamma(100.0, 100.0, 0.25, 0.05, 0.20, 'call', style)
        assert g > 0, f"Expected positive gamma for style={style}"


def test_compute_gamma_unknown_style_raises():
    with pytest.raises((ValueError, AttributeError)):
        compute_gamma(100.0, 100.0, 0.25, 0.05, 0.20, 'call', 'invalid_style')
