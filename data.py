import yfinance as yf
import pandas as pd
import requests
from datetime import datetime
from io import StringIO

from freshness import assess_snapshot, SnapshotQuality


def fetch_chain(ticker: str) -> tuple:
    """Fetch options chain from Yahoo Finance (primary source, 15-min delayed)."""
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
    quality = assess_snapshot("yfinance")
    return df, float(spot), quality


def fetch_chain_cboe(ticker: str) -> tuple:
    """Fetch options chain from CBOE delayed quote table (fallback, free, no account)."""
    url = f"https://www.cboe.com/delayed_quotes/{ticker.lower()}/quote_table/"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()

    tables = pd.read_html(StringIO(resp.text))
    if len(tables) < 2:
        raise ValueError(f"CBOE page for {ticker} did not return expected tables")

    # CBOE renders calls (table 0) and puts (table 1)
    def parse_table(tbl, option_type):
        # CBOE column names vary; normalize common variants
        col_map = {}
        for col in tbl.columns:
            c = str(col).lower()
            if 'strike' in c:
                col_map[col] = 'strike'
            elif 'open' in c and 'int' in c:
                col_map[col] = 'openInterest'
            elif 'iv' in c or 'impl' in c:
                col_map[col] = 'impliedVolatility'
        tbl = tbl.rename(columns=col_map)
        needed = [c for c in ('strike', 'openInterest', 'impliedVolatility') if c in tbl.columns]
        tbl = tbl[needed].copy()
        tbl['type'] = option_type
        return tbl

    calls = parse_table(tables[0], 'call')
    puts  = parse_table(tables[1], 'put')

    # Use yfinance only for spot price
    stock = yf.Ticker(ticker)
    spot = float(stock.fast_info.last_price or stock.info.get('regularMarketPrice', 0))

    # IV from CBOE may be a percentage string like "25.3%" — normalize to decimal
    for df in (calls, puts):
        if 'impliedVolatility' in df.columns:
            df['impliedVolatility'] = (
                pd.to_numeric(
                    df['impliedVolatility'].astype(str).str.replace('%', '', regex=False),
                    errors='coerce'
                ) / 100
            ).fillna(0.20)
        if 'openInterest' in df.columns:
            df['openInterest'] = pd.to_numeric(df['openInterest'], errors='coerce').fillna(0).astype(int)

    # CBOE quote table is front-month only — use today as expiration placeholder
    exp = datetime.now().strftime('%Y-%m-%d')
    calls['expiration'] = exp
    puts['expiration']  = exp

    df = pd.concat([calls, puts], ignore_index=True)
    quality = assess_snapshot("cboe")
    return df, spot, quality


def fetch_chain_tradier(ticker: str, api_key: str) -> tuple:
    """Tradier real-time options chain. Stub -- configure API key first.
    Returns: (df, spot, SnapshotQuality) where quality.oi_source == OISource.LIVE
    """
    raise NotImplementedError(
        "Tradier not configured. Sign up at tradier.com (free brokerage account), "
        "get API key, then implement this function in data.py."
    )
