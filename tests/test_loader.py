import pytest
import pandas as pd
from pathlib import Path
from regime import Regime
from level_projection import FutLevel
from signal_gpr import MarketTick
from backtest.loader import load_nq_bars, static_snapshots_fn
from backtest.replay import replay


# ── CSV helpers ──────────────────────────────────────────────────────────────

def _write_csv_date_time(tmp_path: Path, rows: list[dict]) -> Path:
    """Date + Time columns (Kaggle NQ format A)."""
    path = tmp_path / "nq_dt.csv"
    lines = ["Date,Time,Open,High,Low,Close,Volume"]
    for i, r in enumerate(rows):
        lines.append(
            f"2020-01-{i+2:02d},09:30,"
            f"{r['open']},{r['high']},{r['low']},{r['close']},{r.get('vol', 100)}"
        )
    path.write_text("\n".join(lines))
    return path


def _write_csv_datetime(tmp_path: Path, rows: list[dict]) -> Path:
    """Single datetime column (format B)."""
    path = tmp_path / "nq_dts.csv"
    lines = ["datetime,open,high,low,close,volume"]
    for i, r in enumerate(rows):
        lines.append(
            f"2020-01-{i+2:02d} 09:30:00,"
            f"{r['open']},{r['high']},{r['low']},{r['close']},{r.get('vol', 100)}"
        )
    path.write_text("\n".join(lines))
    return path


def _normal_row(price: float = 8400.0) -> dict:
    """Normal bar — body fills most of range, no dominant wick."""
    return {"open": price - 2, "high": price + 5, "low": price - 5, "close": price + 2}


def _upper_wick_row(price: float = 8400.0) -> dict:
    """Upper wick = 9pt of 10pt range (0.9 > 0.6 threshold)."""
    return {"open": price, "high": price + 10, "low": price - 1, "close": price}


def _lower_wick_row(price: float = 8400.0) -> dict:
    """Lower wick dominant."""
    return {"open": price, "high": price + 1, "low": price - 10, "close": price}


# ── tests ──────────────────────────────────────────────────────────────────

def test_load_returns_correct_count_and_prices(tmp_path):
    rows = [_normal_row(8400.0 + i * 10) for i in range(20)]
    path = _write_csv_date_time(tmp_path, rows)
    bars = load_nq_bars(path)
    assert len(bars) == 20
    assert all(isinstance(b, MarketTick) for b in bars)
    assert bars[0].price == pytest.approx(8402.0)   # close = price + 2 = 8400 + 2


def test_atr_positive_for_all_bars(tmp_path):
    rows = [_normal_row(8400.0 + i * 5) for i in range(20)]
    path = _write_csv_date_time(tmp_path, rows)
    bars = load_nq_bars(path)
    assert all(b.atr > 0 for b in bars)


def test_in_event_window_always_false(tmp_path):
    rows = [_normal_row() for _ in range(10)]
    path = _write_csv_date_time(tmp_path, rows)
    bars = load_nq_bars(path)
    assert all(b.in_event_window is False for b in bars)


def test_rejection_bar_upper_wick(tmp_path):
    rows = [_normal_row() for _ in range(15)] + [_upper_wick_row()]
    path = _write_csv_date_time(tmp_path, rows)
    bars = load_nq_bars(path)
    assert bars[-1].has_rejection_bar is True


def test_rejection_bar_lower_wick(tmp_path):
    rows = [_normal_row() for _ in range(15)] + [_lower_wick_row()]
    path = _write_csv_date_time(tmp_path, rows)
    bars = load_nq_bars(path)
    assert bars[-1].has_rejection_bar is True


def test_normal_bar_not_rejection(tmp_path):
    rows = [_normal_row() for _ in range(20)]
    path = _write_csv_date_time(tmp_path, rows)
    bars = load_nq_bars(path)
    assert all(b.has_rejection_bar is False for b in bars)


def test_static_snapshots_fn_and_replay_integration(tmp_path):
    rows = [_normal_row(8400.0 + i * 5) for i in range(30)]
    path = _write_csv_datetime(tmp_path, rows)
    bars = load_nq_bars(path)

    regime = Regime(state='positive', distance_to_zero_gamma=50.0, conviction=0.8)
    levels = [FutLevel(fut_price=8600.0, tier=1, strength=0.8,
                       label='call_wall', source_strike=4100.0, basis_pts=50.0)]
    snap_fn = static_snapshots_fn(regime, levels)

    # same snapshot returned for any index
    r0, l0 = snap_fn(0)
    r99, l99 = snap_fn(99)
    assert r0 is regime
    assert l0 is levels
    assert r99 is regime

    # full pipeline runs without error
    trades = replay(bars, snap_fn)
    assert isinstance(trades, list)
