# GEX Live Dashboard — Design Spec
**Date:** 2026-06-03
**Status:** Approved

## Overview

Add a lightweight local web server (`serve.py`) to the existing GEX Ultimate project. The server caches a GEX snapshot (computed by the existing pipeline) and serves a live-updating browser dashboard. The dashboard stays open all session; Chart.js charts update in-place every 5 minutes without a page reload. A staleness indicator warns when data is older than 15 minutes. This is a manual advisory tool — no automated trading, no external connections beyond yfinance.

## Goals

- `python serve.py` → fetches live data, opens `http://localhost:8080` in browser
- Dashboard auto-refreshes every 5 minutes (configurable via env var)
- Charts update in-place — no page flash, no scroll-position loss
- Staleness warning if data > 15 min old (yellow border + badge)
- If yfinance fails mid-session, last-good snapshot stays alive with rising staleness
- Existing `run.py` (HTML report) and `gex_calc.py` (CLI) remain unchanged

---

## File Structure

```
D:/2026/GEX/
├── serve.py                  ← NEW: FastAPI app + background scheduler
├── templates/
│   ├── report.html           (existing — unchanged)
│   └── dashboard.html        ← NEW: live client-side dashboard
└── requirements.txt          ← add: fastapi>=0.110, uvicorn>=0.27
```

No new modules. `serve.py` reuses `data.py`, `greeks.py`, `levels.py` directly.

---

## Module Specifications

### `serve.py`

**Configuration (env vars with defaults):**
```
GEX_REFRESH_SECONDS = 300   # recompute interval
GEX_STALE_SECONDS   = 900   # staleness threshold (15 min)
GEX_PORT            = 8080
GEX_HOST            = "127.0.0.1"
```

**Startup sequence:**
1. Run full pipeline immediately: `fetch_chain → calculate_all_greeks → get_all_levels → add_futures_conversion` for SPY and QQQ
2. Cache result in `_snapshot` (module-level dict, thread-safe via `asyncio.Lock`)
3. Launch background `asyncio` task that repeats on `GEX_REFRESH_SECONDS` cadence
4. Open `http://localhost:8080` in browser via `webbrowser.open`
5. Start uvicorn

**Pipeline failure handling:**
- If refresh fails: log error, keep last-good `_snapshot`, increment `staleness_ms`
- If startup fails for one symbol: store `{"error": str(e)}` for that ticker (same as `run.py`)
- Never fabricate data; always surface staleness

**Endpoints:**

| Method | Path | Description |
|---|---|---|
| GET | `/` | Serves `dashboard.html` |
| GET | `/api/gex` | Returns cached snapshot as JSON |
| GET | `/api/status` | Returns health info |

**`/api/gex` response schema:**
```json
{
  "ts": "2026-06-03T09:32:11Z",
  "staleness_ms": 1200,
  "stale": false,
  "SPY": {
    "all":     { "spot": 592.1, "call_wall": 593, "put_wall": 589, "gamma_flip": 591,
                 "hvl": 588, "vanna_wall": 590, "charm_wall": 590,
                 "total_gex": 4.2e9, "total_vex": 1.1e9, "total_chex": 8.2e7,
                 "regime": "POSITIVE (low vol)",
                 "chart_strikes": [...], "chart_gex": [...], "chart_vex": [...], "chart_chex": [...] },
    "0dte":    { ... } or null,
    "futures": { "symbol": "ES", "price": 5920, "multiplier": 10.01,
                 "spot": 5921, "call_wall": 5930, "put_wall": 5890,
                 "gamma_flip": 5910, "hvl": 5880 },
    "source":  "yfinance"
  },
  "QQQ": { ... }
}
```

**`/api/status` response:**
```json
{
  "last_updated": "2026-06-03T09:32:11Z",
  "staleness_ms": 1200,
  "stale": false,
  "next_refresh_in_s": 263,
  "refresh_interval_s": 300,
  "stale_threshold_s": 900
}
```

---

### `templates/dashboard.html`

Self-contained HTML. No Jinja2 — pure client-side JavaScript. No external font imports (system fonts). Chart.js via CDN (same as `report.html`).

**Layout:**
```
┌──────────────────────────────────────────────────────────────┐
│  GEX LIVE  ·  [yfinance]  ·  Last: 09:32:11  ·  Next: 4:23  │
│  ● LIVE  (or ⚠ STALE with yellow pulsing border)             │
├───────────────────────────────┬──────────────────────────────┤
│  SPY  [POSITIVE (low vol)]    │  QQQ  [NEGATIVE (high vol)]  │
│  ┌──────────┬────────┬──────┐ │  ┌──────────┬────────┬────┐  │
│  │ All Chain│ 0DTE  │  ES  │ │  │ All Chain│ 0DTE  │  NQ │  │
│  │ Spot     │ Spot  │Spot  │ │  │ Spot     │ Spot  │Spot │  │
│  │ Call Wall│Call W │Call W│ │  │ ...                      │  │
│  │ Put Wall │Put W  │Put W │ │  │                          │  │
│  │ Flip     │Flip   │Flip  │ │  │                          │  │
│  │ HVL      │HVL    │HVL   │ │  │                          │  │
│  └──────────┴────────┴──────┘ │  └──────────┴────────┴────┘  │
│  GEX by Strike [bar chart]    │  GEX by Strike [bar chart]   │
│  VEX by Strike [bar chart]    │  VEX by Strike [bar chart]   │
│  CHEX by Strike [bar chart]   │  CHEX by Strike [bar chart]  │
└───────────────────────────────┴──────────────────────────────┘
```

**JavaScript behaviour:**
- On load: immediately `fetch('/api/gex')` and render
- `setInterval` calls `fetch('/api/gex')` every `REFRESH_INTERVAL` seconds (read from `/api/status` on load)
- Charts update in-place: `chart.data.labels = strikes; chart.data.datasets[0].data = values; chart.update('none')`
- Level text nodes updated via `element.textContent = value`
- Regime badge colour: green for POSITIVE, red for NEGATIVE
- Status bar: countdown ticks every second with a separate 1s `setInterval`
- Staleness: if `stale: true` → yellow pulsing CSS border on both symbol cards + "⚠ STALE" badge in header

**Error handling in browser:**
- If `/api/gex` fetch fails (server down): show "⚠ CONNECTION LOST" banner, keep last rendered data, retry every 10s
- If ticker has `error` key: render error card with message text (same pattern as `report.html`)

---

## Run

```bash
cd D:/2026/GEX
python serve.py
# → opens http://localhost:8080 automatically
# → ctrl+C to stop
```

Override defaults:
```bash
GEX_REFRESH_SECONDS=120 GEX_PORT=9090 python serve.py
```

---

## Dependencies Added

```
fastapi>=0.110
uvicorn>=0.27
```

All other dependencies already in `requirements.txt`.

---

## Testing

No unit tests for `serve.py` (thin HTTP layer). Validated by:
1. Running `python serve.py` and confirming dashboard opens
2. Confirming `/api/gex` returns valid JSON with expected keys
3. Confirming charts update in-place after one refresh cycle
4. Confirming staleness indicator appears when `GEX_STALE_SECONDS=10` (test with short threshold)

Existing 45-test suite covers all underlying math modules — no regression risk.

---

## Relationship to Other Sub-projects

| Sub-project | Status | Notes |
|---|---|---|
| 1 — GEX Ultimate (HTML report) | In progress | Must complete first; `serve.py` reuses all its modules |
| 2 — GEX Live Dashboard | This spec | Builds on top of Sub-project 1 |
| 3 — Real-time Data Layer | Future | Swap `data.py` adapter; `serve.py` unchanged |
| 4+ — Advanced features | Future | All additive; API schema is forward-compatible |
