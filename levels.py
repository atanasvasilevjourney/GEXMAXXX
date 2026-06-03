import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date


def find_levels(df: pd.DataFrame, spot: float) -> dict:
    if df.empty:
        return None

    by_strike = (
        df.groupby('strike')[['gex', 'vex', 'chex']]
        .sum()
        .reset_index()
        .sort_values('strike')
        .reset_index(drop=True)
    )

    above = by_strike[by_strike['strike'] > spot]
    below = by_strike[by_strike['strike'] < spot]

    call_wall = float(above.loc[above['gex'].idxmax(), 'strike']) if len(above) > 0 else None
    put_wall  = float(below.loc[below['gex'].idxmin(), 'strike']) if len(below) > 0 else None
    hvl       = float(by_strike.loc[by_strike['gex'].idxmin(), 'strike'])

    by_strike['cum_gex'] = by_strike['gex'].cumsum()
    sign_changes = by_strike[by_strike['cum_gex'].shift(1) * by_strike['cum_gex'] < 0]
    if len(sign_changes) > 0:
        idx = (sign_changes['strike'] - spot).abs().idxmin()
        gamma_flip = float(sign_changes.loc[idx, 'strike'])
    else:
        gamma_flip = float(by_strike.loc[by_strike['cum_gex'].abs().idxmin(), 'strike'])

    vanna_wall = float(by_strike.loc[by_strike['vex'].abs().idxmax(), 'strike'])
    charm_wall = float(by_strike.loc[by_strike['chex'].abs().idxmax(), 'strike'])

    total_gex  = float(df['gex'].sum())
    total_vex  = float(df['vex'].sum())
    total_chex = float(df['chex'].sum())
    regime = 'POSITIVE (low vol)' if total_gex > 0 else 'NEGATIVE (high vol)'

    return {
        'spot':       spot,
        'call_wall':  call_wall,
        'put_wall':   put_wall,
        'gamma_flip': gamma_flip,
        'hvl':        hvl,
        'vanna_wall': vanna_wall,
        'charm_wall': charm_wall,
        'total_gex':  total_gex,
        'total_vex':  total_vex,
        'total_chex': total_chex,
        'regime':     regime,
        'by_strike':  by_strike,
    }


def split_0dte(df: pd.DataFrame) -> tuple:
    today = date.today().isoformat()
    df_0dte  = df[df['expiration'] == today].copy()
    df_multi = df[df['expiration'] != today].copy()
    return df_0dte, df_multi


def get_all_levels(df: pd.DataFrame, spot: float) -> dict:
    df_0dte, df_multi = split_0dte(df)
    return {
        'all':   find_levels(df, spot),
        '0dte':  find_levels(df_0dte, spot) if len(df_0dte) > 0 else None,
        'multi': find_levels(df_multi, spot),
    }


def get_futures_price(symbol: str) -> float:
    ticker = yf.Ticker(symbol)
    price = ticker.fast_info.last_price
    if price is None:
        price = ticker.info.get('regularMarketPrice')
    if price is None:
        price = ticker.history(period='1d')['Close'].iloc[-1]
    return float(price)


def add_futures_conversion(levels: dict, ticker: str, spot: float) -> dict:
    futures_symbol = 'NQ=F' if ticker == 'QQQ' else 'ES=F'
    futures_name   = 'NQ'   if ticker == 'QQQ' else 'ES'

    try:
        futures_price = get_futures_price(futures_symbol)
        mult = futures_price / spot

        def conv(val):
            return round(val * mult) if val is not None else None

        all_lvl = levels['all']
        levels['futures'] = {
            'symbol':     futures_name,
            'price':      round(futures_price),
            'multiplier': round(mult, 2),
            'spot':       round(spot * mult),
            'call_wall':  conv(all_lvl['call_wall']),
            'put_wall':   conv(all_lvl['put_wall']),
            'gamma_flip': conv(all_lvl['gamma_flip']),
            'hvl':        conv(all_lvl['hvl']),
        }
    except Exception as e:
        levels['futures'] = {'symbol': futures_name, 'error': str(e)}

    return levels
