from __future__ import annotations
from pathlib import Path
from typing import Callable

import pandas as pd

from regime import Regime
from level_projection import FutLevel
from signal_gpr import MarketTick


def load_nq_bars(
    csv_path: str | Path,
    atr_period: int = 14,
    rej_wick_ratio: float = 0.6,
) -> list[MarketTick]:
    """
    Load a Kaggle NQ CSV and return list[MarketTick] for backtest.replay().

    Handles CSV formats:
      - Separate Date + Time columns (any case)
      - Single 'datetime' or 'timestamp' column

    price          = bar close
    atr            = Wilder ATR (atr_period bars); warmup NaNs back-filled
    has_rejection_bar = True if upper or lower wick > rej_wick_ratio * bar range
    in_event_window   = False (historical data — no live staleness)

    Args:
        csv_path:      path to CSV file
        atr_period:    ATR lookback period (default 14)
        rej_wick_ratio: wick fraction threshold (default 0.6)

    Returns:
        list[MarketTick]
    """
    df = _load_csv(csv_path)
    df['atr']     = _compute_atr(df, atr_period)
    df['has_rej'] = _detect_rejection_bars(df, rej_wick_ratio)

    return [
        MarketTick(
            price             = float(row['close']),
            atr               = float(row['atr']),
            has_rejection_bar = bool(row['has_rej']),
            in_event_window   = False,
        )
        for _, row in df.iterrows()
    ]


def static_snapshots_fn(
    regime: Regime,
    fut_levels: list[FutLevel],
) -> Callable[[int], tuple[Regime, list[FutLevel]]]:
    """
    Return a snapshots_fn that always yields the same (regime, fut_levels).

    Use this when you have a single current GEX snapshot and want to apply
    it uniformly across the entire backtest.

    Args:
        regime:     Regime dataclass
        fut_levels: list[FutLevel] — gamma walls projected to NQ price

    Returns:
        Callable[[int], tuple[Regime, list[FutLevel]]]
    """
    def _fn(_bar_index: int) -> tuple[Regime, list[FutLevel]]:
        return regime, fut_levels
    return _fn


# ── private helpers ──────────────────────────────────────────────────────────

def _load_csv(path: str | Path) -> pd.DataFrame:
    """Read CSV, normalise column names, parse datetime, sort by time."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    if 'date' in df.columns and 'time' in df.columns:
        df['datetime'] = pd.to_datetime(
            df['date'].astype(str) + ' ' + df['time'].astype(str)
        )
        df = df.drop(columns=['date', 'time'])
    elif 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
    elif 'timestamp' in df.columns:
        df['datetime'] = pd.to_datetime(df['timestamp'])
        df = df.drop(columns=['timestamp'])
    else:
        raise ValueError(
            f"No datetime columns found. Got columns: {list(df.columns)}"
        )

    df = df.set_index('datetime').sort_index()

    for col in ('open', 'high', 'low', 'close'):
        if col not in df.columns:
            raise ValueError(
                f"Required column '{col}' missing. Got: {list(df.columns)}"
            )

    return df


def _compute_atr(df: pd.DataFrame, period: int) -> pd.Series:
    """
    Wilder-style ATR via EWM (alpha=1/period, adjust=False).
    Any NaN values in the warmup period are back-filled with the first valid ATR.
    """
    prev_close = df['close'].shift(1)
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - prev_close).abs(),
        (df['low']  - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    return atr.bfill()


def _detect_rejection_bars(df: pd.DataFrame, wick_ratio: float) -> pd.Series:
    """
    True if upper or lower wick exceeds wick_ratio fraction of bar range.

    upper_wick = high - max(open, close)
    lower_wick = min(open, close) - low
    rejection  = (upper_wick / range > wick_ratio) OR (lower_wick / range > wick_ratio)
    """
    hl       = df['high'] - df['low']
    body_top = df[['open', 'close']].max(axis=1)
    body_bot = df[['open', 'close']].min(axis=1)
    upper    = df['high'] - body_top
    lower    = body_bot   - df['low']
    denom    = hl + 1e-9
    return (upper / denom > wick_ratio) | (lower / denom > wick_ratio)
