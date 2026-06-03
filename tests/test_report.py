import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pandas as pd
from unittest.mock import patch
from greeks import calculate_all_greeks
from levels import get_all_levels, add_futures_conversion
from report import build_report, save_report

SPOT = 540.0


def make_symbol_data(ticker):
    df = pd.DataFrame([
        {'strike': 545.0, 'type': 'call', 'openInterest': 10000, 'impliedVolatility': 0.20, 'expiration': '2026-06-20'},
        {'strike': 535.0, 'type': 'put',  'openInterest': 8000,  'impliedVolatility': 0.22, 'expiration': '2026-06-20'},
    ])
    df = calculate_all_greeks(df, SPOT)
    levels = get_all_levels(df, SPOT)
    futures_price = 21500.0 if ticker == 'QQQ' else 5400.0
    with patch('levels.get_futures_price', return_value=futures_price):
        levels = add_futures_conversion(levels, ticker, SPOT)
    levels['source'] = 'yfinance'
    levels['ticker'] = ticker
    return levels


def test_build_report_contains_spy():
    results = {'SPY': make_symbol_data('SPY'), 'QQQ': make_symbol_data('QQQ')}
    html = build_report(results, '2026-06-02')
    assert 'SPY' in html


def test_build_report_contains_qqq():
    results = {'SPY': make_symbol_data('SPY'), 'QQQ': make_symbol_data('QQQ')}
    html = build_report(results, '2026-06-02')
    assert 'QQQ' in html


def test_build_report_contains_date():
    results = {'SPY': make_symbol_data('SPY'), 'QQQ': make_symbol_data('QQQ')}
    html = build_report(results, '2026-06-02')
    assert '2026-06-02' in html


def test_build_report_contains_regime():
    results = {'SPY': make_symbol_data('SPY'), 'QQQ': make_symbol_data('QQQ')}
    html = build_report(results, '2026-06-02')
    assert 'POSITIVE' in html or 'NEGATIVE' in html


def test_build_report_contains_chart_js():
    results = {'SPY': make_symbol_data('SPY'), 'QQQ': make_symbol_data('QQQ')}
    html = build_report(results, '2026-06-02')
    assert 'chart.js' in html.lower()


def test_build_report_contains_futures_symbol():
    results = {'SPY': make_symbol_data('SPY'), 'QQQ': make_symbol_data('QQQ')}
    html = build_report(results, '2026-06-02')
    assert 'ES' in html
    assert 'NQ' in html


def test_build_report_handles_error_symbol():
    results = {
        'SPY': make_symbol_data('SPY'),
        'QQQ': {'ticker': 'QQQ', 'error': 'rate limited by yahoo'},
    }
    html = build_report(results, '2026-06-02')
    assert 'rate limited by yahoo' in html


def test_save_report_creates_file(tmp_path, monkeypatch):
    import report as rmod
    from pathlib import Path
    monkeypatch.setattr(rmod, '_reports_dir', lambda: tmp_path)
    results = {'SPY': make_symbol_data('SPY'), 'QQQ': make_symbol_data('QQQ')}
    html = build_report(results, '2026-06-02')
    path = save_report(html, '2026-06-02')
    assert os.path.exists(path)
    assert '2026-06-02' in path
