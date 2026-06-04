"""
GPR Real Backtest
=================
Downloads 60 days of NQ=F 5-min bars via yfinance, fetches live GEX levels
for QQQ, converts to NQ price space, then runs the full backtest pipeline.

Usage:
    python run_backtest.py
"""
from __future__ import annotations
import os
import sys
import tempfile
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import pandas as pd
import yfinance as yf

# Project imports
sys.path.insert(0, os.path.dirname(__file__))
from data import fetch_chain
from greeks import calculate_all_greeks
from levels import get_all_levels
from level_projection import FutLevel
from regime import Regime
from backtest import load_nq_bars, static_snapshots_fn, replay, compute_stats, format_report


# ── 1. Fetch live GEX snapshot ───────────────────────────────────────────────

print("Fetching QQQ options chain...")
try:
    df_opts, qqq_spot, quality = fetch_chain("QQQ")
except Exception as e:
    print(f"  yfinance failed ({e}), trying CBOE...")
    from data import fetch_chain_cboe
    df_opts, qqq_spot, quality = fetch_chain_cboe("QQQ")

# Filter: ±30% of spot, valid IV (0.01–5.0), positive open interest
from datetime import date
today = date.today().isoformat()
df_opts = df_opts[
    (df_opts['strike'] >= qqq_spot * 0.70) &
    (df_opts['strike'] <= qqq_spot * 1.30) &
    (df_opts['impliedVolatility'] > 0.01) &
    (df_opts['impliedVolatility'] < 5.0) &
    (df_opts['openInterest'] > 0) &
    (df_opts['expiration'] > today)        # exclude expired / 0-DTE
].copy()
print(f"  Options after filter: {len(df_opts)} rows")
df_opts = calculate_all_greeks(df_opts, qqq_spot)
all_levels = get_all_levels(df_opts, qqq_spot)
snap = all_levels['all']
print(f"  QQQ spot: {qqq_spot:.2f}")
print(f"  call_wall: {snap.get('call_wall')}, put_wall: {snap.get('put_wall')}, gamma_flip: {snap.get('gamma_flip'):.2f}")
print(f"  regime: {snap['regime']}")
print(f"  staleness: is_intraday_stale={quality.is_intraday_stale}")


# ── 2. Get NQ=F price for QQQ -> NQ conversion ───────────────────────────────

print("\nFetching NQ=F price...")
nq_ticker = yf.Ticker("NQ=F")
nq_price = nq_ticker.fast_info.last_price
if nq_price is None or nq_price < 1000:
    hist = nq_ticker.history(period="1d")
    nq_price = float(hist['Close'].iloc[-1]) if not hist.empty else None

if nq_price is None:
    print("ERROR: Could not fetch NQ price. Exiting.")
    sys.exit(1)

mult = nq_price / qqq_spot
print(f"  NQ=F price: {nq_price:.2f}  |  multiplier QQQ->NQ: {mult:.3f}")


# ── 3. Build FutLevel objects (tier-1 gamma levels in NQ price space) ────────

fut_levels: list[FutLevel] = []
basis_pts = nq_price - qqq_spot  # additive basis (large, but stored for reference)

level_keys = [
    ('call_wall',  'call_wall',  0.9),
    ('put_wall',   'put_wall',   0.9),
    ('gamma_flip', 'zero_gamma', 0.7),
    ('hvl',        'hvl',        0.6),
]
for key, label, strength in level_keys:
    val = snap.get(key)
    if val is not None:
        fut_price = float(val) * mult   # multiplicative QQQ->NQ conversion
        fut_levels.append(FutLevel(
            fut_price    = round(fut_price, 2),
            tier         = 1,
            strength     = strength,
            label        = label,
            source_strike= float(val),
            basis_pts    = basis_pts,
        ))

print(f"\nGamma levels in NQ space:")
for lv in sorted(fut_levels, key=lambda x: x.fut_price):
    print(f"  {lv.label:12s}  {lv.fut_price:>10.2f}  (tier={lv.tier}, strength={lv.strength:.2f})")


# ── 4. Build Regime ──────────────────────────────────────────────────────────

total_gex = snap['total_gex']
gamma_flip = snap.get('gamma_flip', qqq_spot)
state = 'positive' if total_gex > 0 else 'negative'
dist  = (qqq_spot - gamma_flip) * mult   # convert to NQ pts
regime = Regime(state=state, distance_to_zero_gamma=dist, conviction=min(abs(dist) / 50.0, 1.0))
print(f"\nRegime: state={regime.state}, distance={regime.distance_to_zero_gamma:.1f} NQ pts, conviction={regime.conviction:.2f}")


# ── 5. Download NQ=F 5-min historical bars (60 days) ────────────────────────

print("\nDownloading NQ=F 5-min bars (60 days)...")
nq_hist = yf.download("NQ=F", period="60d", interval="5m", auto_adjust=True, progress=False)

# Flatten MultiIndex columns if present (yfinance >= 0.2)
if isinstance(nq_hist.columns, pd.MultiIndex):
    nq_hist.columns = [col[0].lower() for col in nq_hist.columns]
else:
    nq_hist.columns = [c.lower() for c in nq_hist.columns]

nq_hist = nq_hist.dropna(subset=['close'])
print(f"  Downloaded {len(nq_hist)} bars  |  {nq_hist.index[0]}  ->  {nq_hist.index[-1]}")

# Save to temp CSV (loader expects datetime index)
tmp = tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w', newline='')
nq_hist.index.name = 'datetime'
nq_hist[['open', 'high', 'low', 'close', 'volume']].to_csv(tmp)
tmp.close()


# ── 6. Load bars + run backtest ──────────────────────────────────────────────

print("\nLoading bars into MarketTick format...")
bars = load_nq_bars(tmp.name)
os.unlink(tmp.name)
print(f"  {len(bars)} MarketTick objects")

print("\nRunning replay...")
trades = replay(bars, static_snapshots_fn(regime, fut_levels))
print(f"  Completed trades: {len(trades)}")

if len(trades) < 2:
    print("\nWARNING:  Too few trades to run stats (need >= 2).")
    print("   Likely cause: static snapshot levels don't align with 60-day price range.")
    print("   NQ price range in data:")
    closes = [b.price for b in bars]
    print(f"   min={min(closes):.0f}  max={max(closes):.0f}  current={nq_price:.0f}")
    print("\n   Level prices:")
    for lv in sorted(fut_levels, key=lambda x: x.fut_price):
        in_range = min(closes) <= lv.fut_price <= max(closes)
        print(f"   {lv.label:12s}  {lv.fut_price:>10.2f}  {'IN RANGE' if in_range else 'OUT OF RANGE'}")
    sys.exit(0)


# ── 7. Stats + validation ────────────────────────────────────────────────────

print("\nComputing statistics...")
result = compute_stats(trades, run_validation=True)

print("\n" + "="*60)
print(format_report(result))
print("="*60)

print(f"\nMonte Carlo p-value:  {result.validation.mc_pvalue:.4f}  ({'PASS' if result.validation.mc_passed else 'FAIL'})")
print(f"Kupiec POF p-value:   {result.validation.kupiec_pvalue:.4f}  ({'PASS' if result.validation.kupiec_passed else 'FAIL'})")
print(f"Validation:           {'PASSED' if result.validation.passed else 'FAILED'}")
print(f"\nsignal_only = {result.signal_only}")
if result.signal_only:
    print("  -> Edge NOT confirmed. Run in signal-only mode (no live execution).")
else:
    print("  -> Edge confirmed on this data. Ready for paper trading.")

print(f"\n{'='*60}")
print("NOTE: This backtest uses a STATIC GEX snapshot (today's levels)")
print("applied uniformly across 60 days. Results are directionally")
print("informative but not rigorous — historical GEX data would be needed")
print("for a proper walk-forward test.")
print("="*60)
