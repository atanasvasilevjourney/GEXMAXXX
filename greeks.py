import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime
from pricer import compute_gamma, detect_style, time_to_expiry


def calculate_gex(df: pd.DataFrame, spot: float, r: float = 0.05,
                  ticker: str = 'SPY') -> pd.DataFrame:
    """
    Compute GEX (gamma exposure) for each option in the chain.

    Uses the correct pricer per ticker:
      - European BS  for SPX, NDX (cash-settled, no early exercise)
      - American BAW for QQQ, SPY, IWM (ETF options, early exercise possible)
      - Black-76     for ES=F, NQ=F (futures options)

    Args:
        df:     options chain DataFrame with columns:
                  strike, type ('call'|'put'), openInterest,
                  impliedVolatility, expiration ('YYYY-MM-DD')
        spot:   current underlying price
        r:      risk-free rate (default 0.05)
        ticker: underlying ticker — used to select pricer style (default 'SPY')

    Returns:
        df with added columns: T, iv, d1, d2, gamma, gex
    """
    style = detect_style(ticker)
    df    = df.copy()

    # Exact T in years (60-second floor; replaces old clip(lower=0.001))
    df['T']  = df['expiration'].apply(time_to_expiry)
    df['iv'] = df['impliedVolatility'].replace(0, np.nan).fillna(0.20)

    # d1/d2 retained for vanna/charm downstream (calculate_vanna, calculate_charm)
    df['d1'] = (
        np.log(spot / df['strike']) + (r + 0.5 * df['iv'] ** 2) * df['T']
    ) / (df['iv'] * np.sqrt(df['T']))
    df['d2'] = df['d1'] - df['iv'] * np.sqrt(df['T'])

    # Gamma via correct pricer per style
    df['gamma'] = df.apply(
        lambda row: compute_gamma(
            spot, row['strike'], row['T'], r, row['iv'], row['type'], style
        ),
        axis=1,
    )

    # GEX: gamma * OI * 100 (multiplier) * spot^2 * 0.01 (per-1%-move convention)
    df['gex'] = df['gamma'] * df['openInterest'] * 100 * spot ** 2 * 0.01
    df.loc[df['type'] == 'put', 'gex'] *= -1
    return df


def calculate_vanna(df: pd.DataFrame, spot: float) -> pd.DataFrame:
    # Vanna = d(delta)/d(iv) = -norm.pdf(d1) * d2 / iv
    # Measures how dealer delta changes when IV moves (vol-driven hedging flows)
    df['vanna'] = -norm.pdf(df['d1']) * df['d2'] / df['iv']
    df['vex'] = df['vanna'] * df['openInterest'] * 100 * spot * 0.01
    df.loc[df['type'] == 'put', 'vex'] *= -1
    return df


def calculate_charm(df: pd.DataFrame, spot: float, r: float = 0.05) -> pd.DataFrame:
    # Charm = d(delta)/d(t) = time-decay rate of delta (theta of delta)
    # Measures how dealer delta changes as time passes (time-driven hedging flows)
    df['charm'] = -norm.pdf(df['d1']) * (
        2 * r * df['T'] - df['d2'] * df['iv'] * np.sqrt(df['T'])
    ) / (2 * df['T'] * df['iv'] * np.sqrt(df['T']))
    df['chex'] = df['charm'] * df['openInterest'] * 100
    df.loc[df['type'] == 'put', 'chex'] *= -1
    return df


def calculate_all_greeks(df: pd.DataFrame, spot: float, r: float = 0.05) -> pd.DataFrame:
    df = calculate_gex(df, spot, r)
    df = calculate_vanna(df, spot)
    df = calculate_charm(df, spot, r)
    return df
