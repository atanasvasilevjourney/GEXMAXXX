"""
GEX Snapshot Logger
===================
Fetches the current QQQ GEX snapshot and appends it to data/snapshots.jsonl.
One JSON object per line. Idempotent — skips if today's snapshot already exists.

Usage:
    python log_snapshot.py           # logs today's snapshot
    python log_snapshot.py --force   # overwrite today's entry if it exists

Schedule daily (after market close, e.g. 17:00 ET):
    Windows: schtasks (see bottom of this file)
    Linux/Mac: cron  0 17 * * 1-5  cd /path/to/GEX && python log_snapshot.py
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import warnings
from datetime import date, datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import yfinance as yf

sys.path.insert(0, os.path.dirname(__file__))
from data import fetch_chain, fetch_chain_cboe
from greeks import calculate_all_greeks
from levels import get_all_levels

SNAPSHOTS_FILE = Path(__file__).parent / "data" / "snapshots.jsonl"
ERRORS_FILE    = Path(__file__).parent / "data" / "snapshots_errors.log"


def _log_error(msg: str) -> None:
    ts = datetime.now(timezone.utc).isoformat()
    with open(ERRORS_FILE, "a") as f:
        f.write(f"{ts}  {msg}\n")
    print(f"ERROR: {msg}", file=sys.stderr)


def _load_today_dates() -> set[str]:
    """Return set of dates already in snapshots.jsonl."""
    if not SNAPSHOTS_FILE.exists():
        return set()
    dates = set()
    with open(SNAPSHOTS_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    dates.add(json.loads(line)["date"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return dates


def fetch_snapshot() -> dict:
    """Fetch QQQ options, compute GEX, return snapshot dict."""
    # --- options chain ---
    try:
        df_opts, qqq_spot, quality = fetch_chain("QQQ")
    except Exception as e:
        print(f"  yfinance failed ({e}), trying CBOE...")
        df_opts, qqq_spot, quality = fetch_chain_cboe("QQQ")

    today = date.today().isoformat()
    df_opts = df_opts[
        (df_opts['strike'] >= qqq_spot * 0.70) &
        (df_opts['strike'] <= qqq_spot * 1.30) &
        (df_opts['impliedVolatility'] > 0.01) &
        (df_opts['impliedVolatility'] < 5.0) &
        (df_opts['openInterest'] > 0) &
        (df_opts['expiration'] > today)
    ].copy()

    df_opts = calculate_all_greeks(df_opts, qqq_spot)
    snap    = get_all_levels(df_opts, qqq_spot)['all']

    # --- NQ futures price ---
    nq_ticker = yf.Ticker("NQ=F")
    nq_price  = nq_ticker.fast_info.last_price
    if nq_price is None or nq_price < 1000:
        hist     = nq_ticker.history(period="1d")
        nq_price = float(hist['Close'].iloc[-1]) if not hist.empty else None
    if nq_price is None:
        raise RuntimeError("Could not fetch NQ=F price")

    mult      = nq_price / qqq_spot
    basis_pts = nq_price - qqq_spot

    # --- fut_levels ---
    fut_levels = []
    for key, label, strength in [
        ('call_wall',  'call_wall',  0.9),
        ('put_wall',   'put_wall',   0.9),
        ('gamma_flip', 'zero_gamma', 0.7),
        ('hvl',        'hvl',        0.6),
    ]:
        val = snap.get(key)
        if val is not None:
            fut_levels.append({
                "fut_price":     round(float(val) * mult, 2),
                "tier":          1,
                "strength":      strength,
                "label":         label,
                "source_strike": float(val),
                "basis_pts":     round(basis_pts, 4),
            })

    total_gex  = snap['total_gex']
    gamma_flip = snap.get('gamma_flip', qqq_spot)

    return {
        "date":              today,
        "fetched_at":        datetime.now(timezone.utc).isoformat(),
        "qqq_spot":          round(float(qqq_spot), 4),
        "nq_price":          round(float(nq_price), 2),
        "regime_state":      "positive" if total_gex > 0 else "negative",
        "gamma_flip_qqq":    round(float(gamma_flip), 4),
        "call_wall_qqq":     round(float(snap['call_wall']), 4) if snap.get('call_wall') else None,
        "put_wall_qqq":      round(float(snap['put_wall']), 4)  if snap.get('put_wall')  else None,
        "hvl_qqq":           round(float(snap['hvl']), 4)       if snap.get('hvl')       else None,
        "total_gex":         round(float(total_gex), 2),
        "fut_levels":        fut_levels,
        "is_intraday_stale": bool(quality.is_intraday_stale),
        "oi_source":         quality.oi_source.value,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true',
                        help='Log even if today already has an entry')
    args = parser.parse_args()

    today = date.today().isoformat()

    if not args.force:
        existing = _load_today_dates()
        if today in existing:
            print(f"Snapshot for {today} already exists. Use --force to overwrite.")
            return

    print(f"Fetching GEX snapshot for {today}...")
    try:
        record = fetch_snapshot()
    except Exception as e:
        _log_error(f"fetch_snapshot failed on {today}: {e}")
        sys.exit(1)

    # If --force, rewrite file without today's old entry, then append
    if args.force and SNAPSHOTS_FILE.exists():
        lines = SNAPSHOTS_FILE.read_text().splitlines()
        kept  = [l for l in lines if l.strip() and json.loads(l).get("date") != today]
        SNAPSHOTS_FILE.write_text("\n".join(kept) + ("\n" if kept else ""))

    SNAPSHOTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SNAPSHOTS_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")

    print(f"  Logged: regime={record['regime_state']}, "
          f"QQQ={record['qqq_spot']}, NQ={record['nq_price']}, "
          f"call_wall={record['call_wall_qqq']}, put_wall={record['put_wall_qqq']}, "
          f"gamma_flip={record['gamma_flip_qqq']}")
    print(f"  Saved to: {SNAPSHOTS_FILE}")


if __name__ == "__main__":
    main()


# ── Windows Task Scheduler setup ─────────────────────────────────────────────
#
# Run once in an admin PowerShell to schedule daily at 17:15 (after market close):
#
#   $python = (Get-Command python).Source
#   $script = "D:\2026\GEX\log_snapshot.py"
#   schtasks /create /tn "GEX_Snapshot_Logger" /tr "$python $script" `
#            /sc daily /st 17:15 /ru SYSTEM /f
#
# To verify:
#   schtasks /query /tn "GEX_Snapshot_Logger"
#
# To run manually right now:
#   schtasks /run /tn "GEX_Snapshot_Logger"
#
# To delete:
#   schtasks /delete /tn "GEX_Snapshot_Logger" /f
