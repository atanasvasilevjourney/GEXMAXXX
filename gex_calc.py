import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime


def fetch_chain(ticker: str) -> tuple:
    stock = yf.Ticker(ticker)
    spot = stock.fast_info.last_price

    if spot is None:
        spot = stock.info.get('regularMarketPrice')
    if spot is None:
        spot = stock.history(period='1d')['Close'].iloc[-1]

    expirations = stock.options[:4]  # nearest 4 expirations
    if not expirations:
        raise ValueError(f"No options expirations found for {ticker}")

    frames = []
    for exp in expirations:
        chain = stock.option_chain(exp)

        calls = chain.calls[['strike', 'openInterest', 'impliedVolatility']].copy()
        calls['type'] = 'call'

        puts = chain.puts[['strike', 'openInterest', 'impliedVolatility']].copy()
        puts['type'] = 'put'

        combined = pd.concat([calls, puts], ignore_index=True)
        combined['expiration'] = exp
        frames.append(combined)

    df = pd.concat(frames, ignore_index=True)
    return df, float(spot)


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

    # Gamma Flip: cumulative GEX crosses zero (the actual flip)
    by_strike['cum_gex'] = by_strike['gex'].cumsum()
    # Find the first strike where cumulative GEX changes sign (the actual flip)
    sign_changes = by_strike[by_strike['cum_gex'].shift(1) * by_strike['cum_gex'] < 0]
    if len(sign_changes) > 0:
        # Pick the sign change closest to spot (not the lowest-strike one)
        idx = (sign_changes['strike'] - spot).abs().idxmin()
        gamma_flip = float(sign_changes.loc[idx, 'strike'])
    else:
        # No sign change in chain: use strike closest to zero cumulative GEX as fallback
        gamma_flip = float(by_strike.loc[by_strike['cum_gex'].abs().idxmin(), 'strike'])

    total_gex = float(df['gex'].sum())
    regime = 'POSITIVE (low vol)' if total_gex > 0 else 'NEGATIVE (high vol)'

    return {
        'spot': spot,
        'call_wall': float(call_wall) if call_wall is not None else None,
        'put_wall': float(put_wall) if put_wall is not None else None,
        'gamma_flip': gamma_flip,
        'hvl': float(hvl),
        'total_gex': total_gex,
        'regime': regime,
    }


def print_levels(ticker: str, levels: dict) -> None:
    date_str = datetime.now().strftime('%Y-%m-%d')
    spot = levels['spot']
    width = 37  # separator bar width

    def fmt_strike(val) -> str:
        if val is None:
            return '  N/A'
        pct = (val - spot) / spot * 100
        sign = '+' if pct >= 0 else ''
        return f"  {val:>8.2f}   ({sign}{pct:.2f}%)"

    gex_b = levels['total_gex'] / 1e9
    gex_sign = '+' if gex_b >= 0 else ''

    print(f"\n{'=' * width}")
    print(f"  {ticker} GEX LEVELS  -- {date_str}")
    print(f"{'=' * width}")
    print(f"  Spot         :  {spot:>8.2f}")
    print(f"  Call Wall    :{fmt_strike(levels['call_wall'])}")
    print(f"  Put Wall     :{fmt_strike(levels['put_wall'])}")
    print(f"  Gamma Flip   :{fmt_strike(levels['gamma_flip'])}")
    print(f"  HVL          :{fmt_strike(levels['hvl'])}")
    print(f"  Total GEX    :  {gex_sign}${gex_b:.2f}B")
    print(f"  Regime       :  {levels['regime']}")
    print(f"{'=' * width}")


def fetch_nq_price() -> float:
    """Fetch current NQ front-month futures price."""
    nq = yf.Ticker("NQ=F")
    price = nq.fast_info.last_price
    if price is None:
        price = nq.info.get('regularMarketPrice')
    if price is None:
        price = nq.history(period='1d')['Close'].iloc[-1]
    return float(price)


def print_nq_conversion(levels: dict, nq_multiplier: float) -> None:
    """Print NQ equivalent levels derived from QQQ GEX levels."""
    width = 37

    def to_nq(val) -> str:
        if val is None:
            return '  N/A'
        return f"  {val * nq_multiplier:>8.0f}"

    nq_spot = levels['spot'] * nq_multiplier
    print(f"\n{'=' * width}")
    print(f"  NQ EQUIVALENT  (QQQ x {nq_multiplier:.2f})")
    print(f"{'=' * width}")
    print(f"  NQ Spot      :  {nq_spot:>8.0f}")
    print(f"  NQ Call Wall :{to_nq(levels['call_wall'])}")
    print(f"  NQ Put Wall  :{to_nq(levels['put_wall'])}")
    print(f"  NQ Flip      :{to_nq(levels['gamma_flip'])}")
    print(f"  NQ HVL       :{to_nq(levels['hvl'])}")
    print(f"{'=' * width}")


def main():
    qqq_levels = None

    for ticker in ["SPY", "QQQ"]:
        try:
            df, spot = fetch_chain(ticker)
            df = calculate_gex(df, spot)
            levels = find_levels(df, spot)
            print_levels(ticker, levels)
            if ticker == "QQQ":
                qqq_levels = levels
        except Exception as e:
            print(f"\n[ERROR] {ticker}: {e}")

    if qqq_levels is not None:
        try:
            nq_price = fetch_nq_price()
            nq_multiplier = nq_price / qqq_levels['spot']
            print_nq_conversion(qqq_levels, nq_multiplier)
        except Exception as e:
            print(f"\n[ERROR] NQ conversion: {e}")


if __name__ == "__main__":
    main()
