import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime


def calculate_gex(df: pd.DataFrame, spot: float, r: float = 0.05) -> pd.DataFrame:
    df = df.copy()
    df['T'] = (
        pd.to_datetime(df['expiration']) - datetime.now()
    ).dt.total_seconds() / (365.25 * 24 * 3600)
    df['T'] = df['T'].clip(lower=0.001)

    df['iv'] = df['impliedVolatility'].replace(0, np.nan).fillna(0.20)

    df['d1'] = (
        np.log(spot / df['strike']) + (r + 0.5 * df['iv'] ** 2) * df['T']
    ) / (df['iv'] * np.sqrt(df['T']))
    df['d2'] = df['d1'] - df['iv'] * np.sqrt(df['T'])

    df['gamma'] = norm.pdf(df['d1']) / (spot * df['iv'] * np.sqrt(df['T']))
    # GEX per option: * 100 (contract multiplier) * spot^2 * 0.01 (per-1%-move convention)
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
