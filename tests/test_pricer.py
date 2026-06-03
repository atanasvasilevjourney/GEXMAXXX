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


# --- Engine tests (added in Task 2) ---
# DO NOT add engine tests here yet.
