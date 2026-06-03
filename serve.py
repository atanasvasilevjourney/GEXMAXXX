import asyncio
import os
import webbrowser
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

from data import fetch_chain, fetch_chain_cboe
from greeks import calculate_all_greeks
from levels import get_all_levels, add_futures_conversion
from report import _prepare_results

REFRESH_INTERVAL = int(os.getenv("GEX_REFRESH_SECONDS", "300"))
STALE_THRESHOLD  = int(os.getenv("GEX_STALE_SECONDS",   "900"))
HOST             = os.getenv("GEX_HOST", "127.0.0.1")
PORT             = int(os.getenv("GEX_PORT", "8080"))
SYMBOLS          = ["SPY", "QQQ"]

_cache: dict = {"snapshot": None, "last_updated": None, "next_refresh_ts": None}
_lock = asyncio.Lock()


def _json_safe(obj):
    """Recursively convert numpy types to Python native for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _run_pipeline_sync() -> dict:
    """Synchronous GEX pipeline -- called in thread executor to avoid blocking the event loop."""
    results = {}
    for ticker in SYMBOLS:
        try:
            try:
                df, spot = fetch_chain(ticker)
                source = "yfinance"
            except Exception as e_yf:
                print(f"[{ticker}] yfinance failed ({e_yf}), trying CBOE...")
                df, spot = fetch_chain_cboe(ticker)
                source = "cboe"
            df = calculate_all_greeks(df, spot)
            levels = get_all_levels(df, spot)
            levels = add_futures_conversion(levels, ticker, spot)
            levels["source"] = source
            levels["ticker"] = ticker
            results[ticker] = levels
        except Exception as e:
            print(f"[{ticker}] ERROR: {e}")
            results[ticker] = {"ticker": ticker, "error": str(e)}
    return results


async def _refresh():
    """Refresh snapshot cache. On failure, keep last-good snapshot."""
    print(f"[GEX] Refreshing at {datetime.now().strftime('%H:%M:%S')}...")
    try:
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(None, _run_pipeline_sync)
        prepared = _prepare_results(results)
        safe = _json_safe(prepared)
        async with _lock:
            _cache["snapshot"] = safe
            _cache["last_updated"] = datetime.now(timezone.utc)
        print("[GEX] Snapshot updated.")
    except Exception as e:
        print(f"[GEX] Refresh failed: {e} -- keeping last-good snapshot.")


async def _background_loop():
    """Refresh once on startup, then repeat every REFRESH_INTERVAL seconds."""
    await _refresh()
    while True:
        async with _lock:
            _cache["next_refresh_ts"] = datetime.now(timezone.utc).timestamp() + REFRESH_INTERVAL
        await asyncio.sleep(REFRESH_INTERVAL)
        await _refresh()


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(_background_loop())
    await asyncio.sleep(1.0)
    webbrowser.open(f"http://{HOST}:{PORT}")
    yield
    task.cancel()


app = FastAPI(title="GEX Live Dashboard", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    html = (Path(__file__).parent / "templates" / "dashboard.html").read_text(encoding="utf-8")
    return HTMLResponse(content=html)


@app.get("/api/gex")
async def api_gex():
    async with _lock:
        snapshot = _cache["snapshot"]
        last_updated = _cache["last_updated"]

    if snapshot is None:
        return JSONResponse({"error": "initializing -- snapshot not ready yet"}, status_code=503)

    now = datetime.now(timezone.utc)
    staleness_ms = int((now - last_updated).total_seconds() * 1000)

    return {
        "ts": last_updated.isoformat(),
        "staleness_ms": staleness_ms,
        "stale": staleness_ms > STALE_THRESHOLD * 1000,
        **snapshot,
    }


@app.get("/api/status")
async def api_status():
    async with _lock:
        last_updated = _cache["last_updated"]
        next_refresh_ts = _cache["next_refresh_ts"]

    now = datetime.now(timezone.utc)
    staleness_ms = int((now - last_updated).total_seconds() * 1000) if last_updated else 0
    next_refresh_in_s = max(0, int(next_refresh_ts - now.timestamp())) if next_refresh_ts else REFRESH_INTERVAL

    return {
        "last_updated": last_updated.isoformat() if last_updated else None,
        "staleness_ms": staleness_ms,
        "stale": staleness_ms > STALE_THRESHOLD * 1000,
        "next_refresh_in_s": next_refresh_in_s,
        "refresh_interval_s": REFRESH_INTERVAL,
        "stale_threshold_s": STALE_THRESHOLD,
    }


if __name__ == "__main__":
    uvicorn.run("serve:app", host=HOST, port=PORT, reload=False)
