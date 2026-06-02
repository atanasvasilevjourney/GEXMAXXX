import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime


def fetch_chain(ticker: str) -> tuple:
    raise NotImplementedError


def calculate_gex(df: pd.DataFrame, spot: float, r: float = 0.05) -> pd.DataFrame:
    df = df.copy()
    # Time to expiry in years, minimum 0.001 to avoid divide-by-zero
    df['T'] = (
        pd.to_datetime(df['expiration']) - datetime.now()
    ).dt.total_seconds() / (365.25 * 24 * 3600)
    df['T'] = df['T'].clip(lower=0.001)

    # IV fallback: replace 0 or NaN with 20%
    df['iv'] = df['impliedVolatility'].replace(0, np.nan).fillna(0.20)

    # Black-Scholes d1
    d1 = (
        np.log(spot / df['strike']) + (r + 0.5 * df['iv'] ** 2) * df['T']
    ) / (df['iv'] * np.sqrt(df['T']))

    # Analytical gamma
    df['gamma'] = norm.pdf(d1) / (spot * df['iv'] * np.sqrt(df['T']))

    # GEX per option: * 100 (contract multiplier) * spot^2 * 0.01 (per-1%-move convention)
    df['gex'] = df['gamma'] * df['openInterest'] * 100 * spot ** 2 * 0.01

    # Puts subtract from GEX (dealers are long puts = negative exposure)
    df.loc[df['type'] == 'put', 'gex'] *= -1

    return df


def find_levels(df: pd.DataFrame, spot: float) -> dict:
    # Aggregate GEX by strike
    by_strike = (
        df.groupby('strike')['gex']
        .sum()
        .reset_index()
        .sort_values('strike')
        .reset_index(drop=True)
    )

    above = by_strike[by_strike['strike'] > spot]
    below = by_strike[by_strike['strike'] < spot]

    # Call Wall: highest net GEX strike above spot
    call_wall = (
        above.loc[above['gex'].idxmax(), 'strike'] if len(above) > 0 else None
    )

    # Put Wall: most negative net GEX strike below spot
    put_wall = (
        below.loc[below['gex'].idxmin(), 'strike'] if len(below) > 0 else None
    )

    # HVL: strike with most negative GEX on entire chain
    hvl = by_strike.loc[by_strike['gex'].idxmin(), 'strike']

    # Gamma Flip: cumulative GEX crosses zero (strike closest to the sign change)
    by_strike['cum_gex'] = by_strike['gex'].cumsum()
    gamma_flip = by_strike.loc[by_strike['cum_gex'].abs().idxmin(), 'strike']

    total_gex = float(df['gex'].sum())
    regime = 'POSITIVE (low vol)' if total_gex > 0 else 'NEGATIVE (high vol)'

    return {
        'spot': spot,
        'call_wall': call_wall,
        'put_wall': put_wall,
        'gamma_flip': float(gamma_flip),
        'hvl': float(hvl),
        'total_gex': total_gex,
        'regime': regime,
    }


def print_levels(ticker: str, levels: dict) -> None:
    raise NotImplementedError


def main():
    raise NotImplementedError


if __name__ == "__main__":
    main()
