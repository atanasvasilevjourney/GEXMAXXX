# GEX Live Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a local FastAPI server (`serve.py`) that caches the GEX snapshot and serves a live-updating browser dashboard with in-place Chart.js updates every 5 minutes and a staleness indicator.

**Architecture:** `serve.py` runs the existing `data → greeks → levels → report._prepare_results` pipeline in a thread executor on startup and every N minutes. Three endpoints: `/` (HTML), `/api/gex` (cached JSON), `/api/status` (health). `dashboard.html` is pure client-side JS — it loads once and uses `setInterval` + `fetch('/api/gex')` to update charts and level values in-place without a page reload.

**Tech Stack:** Python 3.11+, FastAPI 0.110+, uvicorn 0.27+, Chart.js 4.4 (CDN). All GEX math modules (data.py, greeks.py, levels.py, report.py) are reused unchanged.

**Prerequisite:** Sub-project 1 (GEX Ultimate) must be complete — `data.py`, `greeks.py`, `levels.py`, and `report.py` must all exist and the 45-test suite must be passing before starting this plan.

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `D:/2026/GEX/requirements.txt` | Modify | Add fastapi, uvicorn |
| `D:/2026/GEX/serve.py` | Create | FastAPI app + background scheduler + endpoints |
| `D:/2026/GEX/templates/dashboard.html` | Create | Live client-side dashboard |
| `D:/2026/GEX/data.py` | **Unchanged** | |
| `D:/2026/GEX/greeks.py` | **Unchanged** | |
| `D:/2026/GEX/levels.py` | **Unchanged** | |
| `D:/2026/GEX/report.py` | **Unchanged** | `_prepare_results` reused by serve.py |

---

## Task 1: Update requirements.txt and install FastAPI/uvicorn

**Files:**
- Modify: `D:/2026/GEX/requirements.txt`

- [ ] **Step 1: Update requirements.txt**

Replace `D:/2026/GEX/requirements.txt` with:

```
yfinance>=0.2.40
pandas>=2.0
numpy>=1.26
scipy>=1.12
pytest>=8.0
requests>=2.31
jinja2>=3.1
fastapi>=0.110
uvicorn>=0.27
```

- [ ] **Step 2: Install the new dependencies**

```bash
cd D:/2026/GEX && pip install "fastapi>=0.110" "uvicorn>=0.27"
```

Expected: installs without errors.

- [ ] **Step 3: Verify existing tests still pass**

```bash
cd D:/2026/GEX && pytest tests/ -q
```

Expected: `45 passed` (all GEX Ultimate tests green, zero regressions).

- [ ] **Step 4: Commit**

```bash
cd D:/2026/GEX && git add requirements.txt && git commit -m "chore: add fastapi and uvicorn for live dashboard"
```

---

## Task 2: Implement `serve.py`

**Files:**
- Create: `D:/2026/GEX/serve.py`

- [ ] **Step 1: Create `serve.py` with this exact content**

Create `D:/2026/GEX/serve.py`:

```python
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
    """Synchronous GEX pipeline — called in thread executor to avoid blocking the event loop."""
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
        print(f"[GEX] Refresh failed: {e} — keeping last-good snapshot.")


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
    await asyncio.sleep(1.0)  # let first refresh start before opening browser
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
        return JSONResponse({"error": "initializing — snapshot not ready yet"}, status_code=503)

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
```

- [ ] **Step 2: Verify imports resolve**

```bash
cd D:/2026/GEX && python -c "import serve; print('imports OK')"
```

Expected: `imports OK` (no ImportError). If you see `ModuleNotFoundError: No module named 'data'` it means Sub-project 1 is not yet complete — stop and complete it first.

- [ ] **Step 3: Verify existing tests still pass**

```bash
cd D:/2026/GEX && pytest tests/ -q
```

Expected: `45 passed`

- [ ] **Step 4: Commit**

```bash
cd D:/2026/GEX && git add serve.py && git commit -m "feat: implement serve.py - FastAPI live dashboard server with background GEX refresh"
```

---

## Task 3: Implement `templates/dashboard.html`

**Files:**
- Create: `D:/2026/GEX/templates/dashboard.html`

- [ ] **Step 1: Create `templates/dashboard.html` with this exact content**

Create `D:/2026/GEX/templates/dashboard.html`:

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>GEX Live</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #0f1117; color: #e2e8f0; font-family: Consolas, Monaco, 'Courier New', monospace; font-size: 13px; }

.status-bar { display: flex; align-items: center; gap: 16px; padding: 10px 20px; background: #070909; border-bottom: 1px solid #1a1d2e; position: sticky; top: 0; z-index: 100; flex-wrap: wrap; }
.app-title { font-size: 14px; font-weight: 700; letter-spacing: 3px; color: #f1f5f9; }
.status-item { color: #475569; font-size: 11px; }
.status-item strong { color: #94a3b8; }

.badge { font-size: 10px; font-weight: 700; padding: 2px 8px; border-radius: 3px; letter-spacing: 0.5px; text-transform: uppercase; }
.badge-src  { background: #0f2847; color: #60a5fa; border: 1px solid #1e4080; }
.badge-live { background: #0a2e1a; color: #4ade80; border: 1px solid #166534; }
.badge-stale{ background: #2e1a00; color: #fbbf24; border: 1px solid #92400e; }
.badge-pos  { background: #0a2e1a; color: #4ade80; border: 1px solid #166534; }
.badge-neg  { background: #2e0a0a; color: #f87171; border: 1px solid #7f1d1d; }

.conn-banner { display: none; background: #7f1d1d; color: #fca5a5; text-align: center; padding: 8px; font-size: 12px; font-weight: 700; }

.symbols-grid { display: grid; grid-template-columns: 1fr 1fr; }
@media (max-width: 1100px) { .symbols-grid { grid-template-columns: 1fr; } }

.symbol-card { border-right: 1px solid #252840; }
.symbol-card:last-child { border-right: none; }
.symbol-card.stale { animation: pulse-stale 2s infinite; }
@keyframes pulse-stale {
  0%, 100% { box-shadow: inset 0 0 0 2px #fbbf24; }
  50%       { box-shadow: inset 0 0 0 3px #f59e0b; }
}

.symbol-header { display: flex; align-items: center; gap: 12px; padding: 14px 20px; background: #1a1d2e; border-bottom: 1px solid #252840; }
.symbol-name { font-size: 22px; font-weight: 700; color: #f1f5f9; letter-spacing: -0.5px; }
.error-card { padding: 20px; color: #f87171; font-size: 12px; }

.levels-grid { display: grid; grid-template-columns: repeat(3, 1fr); border-bottom: 1px solid #252840; }
.levels-col { padding: 14px 16px; border-right: 1px solid #252840; }
.levels-col:last-child { border-right: none; }
.col-title { font-size: 10px; color: #475569; letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 10px; font-weight: 700; }
.lrow { display: flex; justify-content: space-between; padding: 3px 0; }
.lkey { color: #64748b; font-size: 11px; }
.lval { font-weight: 700; font-size: 11px; color: #f1f5f9; }
.lval.g { color: #4ade80; }
.lval.r { color: #f87171; }
.lval.y { color: #fbbf24; }
.lval.b { color: #60a5fa; }
.no-data { color: #334155; font-size: 11px; padding: 8px 0; font-style: italic; }

.totals-bar { display: flex; gap: 20px; padding: 8px 16px; background: #0f1117; border-bottom: 1px solid #252840; flex-wrap: wrap; }
.total-item { font-size: 11px; color: #475569; }
.total-item b { color: #94a3b8; margin-left: 4px; }

.charts { padding: 16px; }
.chart-block { margin-bottom: 20px; }
.chart-title { font-size: 10px; color: #475569; letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 6px; font-weight: 700; }
.chart-wrap { position: relative; height: 150px; }

.loading-msg { padding: 20px; color: #334155; font-size: 11px; font-style: italic; }
</style>
</head>
<body>

<div class="status-bar">
  <span class="app-title">GEX LIVE</span>
  <span id="src-badge" class="badge badge-src">--</span>
  <span id="live-badge" class="badge badge-live">● LIVE</span>
  <span class="status-item">Last: <strong id="last-updated">--:--:--</strong></span>
  <span class="status-item">Next: <strong id="next-refresh">--:--</strong></span>
</div>

<div id="conn-banner" class="conn-banner">⚠ CONNECTION LOST -- retrying in 10s...</div>

<div class="symbols-grid">
  <div id="card-SPY" class="symbol-card"><div class="loading-msg">Loading SPY...</div></div>
  <div id="card-QQQ" class="symbol-card"><div class="loading-msg">Loading QQQ...</div></div>
</div>

<script>
const TICKERS = ['SPY', 'QQQ'];
const charts = {};
let refreshInterval = 300;
let countdown = 300;

// ── Formatters ────────────────────────────────────────────────────────────────

function fmt(v, dec) {
  dec = dec == null ? 0 : dec;
  if (v == null) return 'N/A';
  return Number(v).toLocaleString('en-US', { minimumFractionDigits: dec, maximumFractionDigits: dec });
}

function fmtGex(v) {
  if (v == null) return 'N/A';
  const b = v / 1e9;
  return (b >= 0 ? '+$' : '-$') + Math.abs(b).toFixed(2) + 'B';
}

function fmtChex(v) {
  if (v == null) return 'N/A';
  return (v >= 0 ? '+' : '') + (v / 1e6).toFixed(0) + 'M';
}

function fmtTime(iso) {
  if (!iso) return '--:--:--';
  return new Date(iso).toLocaleTimeString('en-US', { hour12: false });
}

function fmtCountdown(s) {
  const m = Math.floor(s / 60), sec = s % 60;
  return m + ':' + String(sec).padStart(2, '0');
}

// ── Charts ────────────────────────────────────────────────────────────────────

const baseChartOpts = {
  responsive: true, maintainAspectRatio: false, animation: false,
  plugins: { legend: { display: false } },
  scales: {
    x: { ticks: { color: '#334155', font: { size: 9 }, maxTicksLimit: 15 }, grid: { color: '#131620' } },
    y: { ticks: { color: '#334155', font: { size: 9 } }, grid: { color: '#1a1d2e' } }
  }
};

function colors(arr, pos, neg) {
  return arr.map(v => v >= 0 ? pos : neg);
}

function upsertChart(id, strikes, values, posColor, negColor) {
  if (charts[id]) {
    charts[id].data.labels = strikes;
    charts[id].data.datasets[0].data = values;
    charts[id].data.datasets[0].backgroundColor = colors(values, posColor, negColor);
    charts[id].update('none');
  } else {
    const canvas = document.getElementById(id);
    if (!canvas) return;
    charts[id] = new Chart(canvas, {
      type: 'bar',
      data: {
        labels: strikes,
        datasets: [{ data: values, backgroundColor: colors(values, posColor, negColor), borderWidth: 0 }]
      },
      options: baseChartOpts
    });
  }
}

function destroyCharts(ticker) {
  ['gex', 'vex', 'chex'].forEach(t => {
    const key = t + '_' + ticker;
    if (charts[key]) { charts[key].destroy(); delete charts[key]; }
  });
}

// ── Card HTML builders ────────────────────────────────────────────────────────

function lv(v, cls) {
  return `<span class="lval ${cls || ''}">${v != null ? fmt(v) : 'N/A'}</span>`;
}

function buildLevelsCol(title, lvl, isFutures) {
  if (!lvl) {
    return `<div class="levels-col">
      <div class="col-title">${title}</div>
      <div class="no-data">No 0DTE today</div>
    </div>`;
  }
  const rows = [
    ['Spot',      fmt(lvl.spot, 2), ''],
    ['Call Wall', lvl.call_wall != null ? fmt(lvl.call_wall) : 'N/A', 'g'],
    ['Put Wall',  lvl.put_wall  != null ? fmt(lvl.put_wall)  : 'N/A', 'r'],
    ['Flip',      fmt(lvl.gamma_flip), 'y'],
    ['HVL',       fmt(lvl.hvl), 'r'],
  ];
  if (!isFutures) {
    rows.push(['Vanna Wall', fmt(lvl.vanna_wall), 'b']);
    rows.push(['Charm Wall', fmt(lvl.charm_wall), 'b']);
  } else if (lvl.multiplier) {
    rows.push(['Mult', lvl.multiplier + 'x', '']);
  }
  const rowsHtml = rows.map(([k, v, cls]) =>
    `<div class="lrow"><span class="lkey">${k}</span><span class="lval ${cls}">${v}</span></div>`
  ).join('');
  return `<div class="levels-col"><div class="col-title">${title}</div>${rowsHtml}</div>`;
}

function buildCardHtml(ticker, d) {
  if (d.error) {
    return `
      <div class="symbol-header">
        <span class="symbol-name">${ticker}</span>
        <span class="badge badge-neg">error</span>
      </div>
      <div class="error-card">${d.error}</div>`;
  }

  const all  = d.all || {};
  const dte0 = d['0dte'];
  const fut  = d.futures || {};
  const isPos = (all.regime || '').includes('POSITIVE');
  const futTitle = fut.symbol ? `${fut.symbol} Equiv` : 'Futures';
  const futLvl = fut.price != null ? {
    spot: fut.spot, call_wall: fut.call_wall, put_wall: fut.put_wall,
    gamma_flip: fut.gamma_flip, hvl: fut.hvl, multiplier: fut.multiplier
  } : null;

  return `
    <div class="symbol-header">
      <span class="symbol-name">${ticker}</span>
      <span class="badge ${isPos ? 'badge-pos' : 'badge-neg'}">${all.regime || '--'}</span>
    </div>
    <div class="levels-grid">
      ${buildLevelsCol('All Chain', all, false)}
      ${buildLevelsCol('0DTE', dte0, false)}
      ${buildLevelsCol(futTitle, futLvl, true)}
    </div>
    <div class="totals-bar">
      <div class="total-item">GEX<b id="tgex-${ticker}">${fmtGex(all.total_gex)}</b></div>
      <div class="total-item">VEX<b id="tvex-${ticker}">${fmtGex(all.total_vex)}</b></div>
      <div class="total-item">CHEX<b id="tchex-${ticker}">${fmtChex(all.total_chex)}</b></div>
    </div>
    <div class="charts">
      <div class="chart-block">
        <div class="chart-title">GEX by Strike</div>
        <div class="chart-wrap"><canvas id="gex_${ticker}"></canvas></div>
      </div>
      <div class="chart-block">
        <div class="chart-title">Vanna Exposure (VEX)</div>
        <div class="chart-wrap"><canvas id="vex_${ticker}"></canvas></div>
      </div>
      <div class="chart-block">
        <div class="chart-title">Charm Exposure (CHEX)</div>
        <div class="chart-wrap"><canvas id="chex_${ticker}"></canvas></div>
      </div>
    </div>`;
}

// ── Render ────────────────────────────────────────────────────────────────────

function renderCard(ticker, d) {
  const card = document.getElementById('card-' + ticker);
  if (!card) return;
  destroyCharts(ticker);
  card.innerHTML = buildCardHtml(ticker, d);
  if (!d.error) drawCharts(ticker, d.all || {});
}

function updateInPlace(ticker, d) {
  const card = document.getElementById('card-' + ticker);
  // If card has no .symbol-header (loading/error state), full re-render
  if (!card || !card.querySelector('.symbol-header') || d.error) {
    renderCard(ticker, d);
    return;
  }
  // Update totals text
  const tgex = document.getElementById('tgex-' + ticker);
  if (tgex) tgex.textContent = fmtGex((d.all || {}).total_gex);
  const tvex = document.getElementById('tvex-' + ticker);
  if (tvex) tvex.textContent = fmtGex((d.all || {}).total_vex);
  const tchex = document.getElementById('tchex-' + ticker);
  if (tchex) tchex.textContent = fmtChex((d.all || {}).total_chex);
  // Update charts in-place
  drawCharts(ticker, d.all || {});
}

function drawCharts(ticker, all) {
  const s = all.chart_strikes || [];
  upsertChart('gex_'  + ticker, s, all.chart_gex  || [], 'rgba(74,222,128,0.75)',  'rgba(248,113,113,0.75)');
  upsertChart('vex_'  + ticker, s, all.chart_vex  || [], 'rgba(96,165,250,0.75)',  'rgba(251,191,36,0.75)');
  upsertChart('chex_' + ticker, s, all.chart_chex || [], 'rgba(167,139,250,0.75)', 'rgba(251,146,60,0.75)');
}

// ── Data fetch ────────────────────────────────────────────────────────────────

let retryTimeout = null;

async function fetchAndRender(isFirstLoad) {
  try {
    const resp = await fetch('/api/gex');
    if (!resp.ok) throw new Error('HTTP ' + resp.status);
    const data = await resp.json();

    // Hide connection lost banner
    document.getElementById('conn-banner').style.display = 'none';
    if (retryTimeout) { clearTimeout(retryTimeout); retryTimeout = null; }

    // Status bar
    document.getElementById('last-updated').textContent = fmtTime(data.ts);
    const firstOk = TICKERS.find(t => data[t] && !data[t].error);
    if (firstOk) document.getElementById('src-badge').textContent = data[firstOk].source || 'yfinance';

    const liveBadge = document.getElementById('live-badge');
    if (data.stale) {
      liveBadge.textContent = '⚠ STALE';
      liveBadge.className = 'badge badge-stale';
      TICKERS.forEach(t => document.getElementById('card-' + t)?.classList.add('stale'));
    } else {
      liveBadge.textContent = '● LIVE';
      liveBadge.className = 'badge badge-live';
      TICKERS.forEach(t => document.getElementById('card-' + t)?.classList.remove('stale'));
    }

    // Render symbol cards
    TICKERS.forEach(t => {
      if (!data[t]) return;
      isFirstLoad ? renderCard(t, data[t]) : updateInPlace(t, data[t]);
    });

    countdown = refreshInterval;
  } catch (e) {
    console.error('Fetch failed:', e);
    document.getElementById('conn-banner').style.display = 'block';
    retryTimeout = setTimeout(() => fetchAndRender(false), 10000);
  }
}

// ── Init ──────────────────────────────────────────────────────────────────────

async function init() {
  // Read refresh interval from status endpoint
  try {
    const s = await fetch('/api/status').then(r => r.json());
    refreshInterval = s.refresh_interval_s || 300;
    countdown = s.next_refresh_in_s || refreshInterval;
  } catch (_) {}

  // First full render
  await fetchAndRender(true);

  // Data refresh on cadence
  setInterval(() => fetchAndRender(false), refreshInterval * 1000);

  // Countdown ticks every second
  setInterval(() => {
    countdown = Math.max(0, countdown - 1);
    document.getElementById('next-refresh').textContent = fmtCountdown(countdown);
  }, 1000);
}

init();
</script>
</body>
</html>
```

- [ ] **Step 2: Verify the file exists**

```bash
ls D:/2026/GEX/templates/
```

Expected: shows both `report.html` and `dashboard.html`

- [ ] **Step 3: Commit**

```bash
cd D:/2026/GEX && git add templates/dashboard.html && git commit -m "feat: implement dashboard.html - live GEX dashboard with in-place Chart.js updates"
```

---

## Task 4: End-to-end validation and push

**Files:** None (validation only)

- [ ] **Step 1: Start the server**

```bash
cd D:/2026/GEX && python serve.py
```

Expected console output (within ~30 seconds):
```
[GEX] Refreshing at HH:MM:SS...
INFO:     Started server process [...]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8080
[GEX] Snapshot updated.
```

Expected: browser opens automatically to `http://localhost:8080` showing the dashboard with SPY and QQQ sections.

- [ ] **Step 2: Check /api/gex returns valid JSON**

In a second terminal (leave the server running):

```bash
curl http://localhost:8080/api/gex | python -m json.tool | head -30
```

Expected: JSON with `ts`, `staleness_ms`, `stale`, `SPY`, `QQQ` keys. `SPY.all.chart_strikes` should be a non-empty list of numbers.

- [ ] **Step 3: Check /api/status returns valid JSON**

```bash
curl http://localhost:8080/api/status
```

Expected:
```json
{"last_updated": "2026-06-03T...", "staleness_ms": ..., "stale": false, "next_refresh_in_s": ..., "refresh_interval_s": 300, "stale_threshold_s": 900}
```

- [ ] **Step 4: Verify staleness indicator**

Stop the server (Ctrl+C). Restart with a short stale threshold:

```bash
cd D:/2026/GEX && GEX_STALE_SECONDS=10 python serve.py
```

Wait 15 seconds after the first snapshot loads. Expected: the "● LIVE" badge changes to "⚠ STALE" and the symbol cards get a yellow pulsing border.

Stop the server (Ctrl+C) and restart normally when done testing.

- [ ] **Step 5: Verify existing tests still pass**

```bash
cd D:/2026/GEX && pytest tests/ -q
```

Expected: `45 passed`

- [ ] **Step 6: Commit and push**

```bash
cd D:/2026/GEX && git add . && git commit -m "test: validate live dashboard end-to-end" && git push
```

---

## Self-Review (completed inline)

**Spec coverage:**
- [x] `python serve.py` → opens `http://localhost:8080` — startup lifespan + webbrowser.open
- [x] Auto-refresh every 5 min (configurable via GEX_REFRESH_SECONDS) — background loop + setInterval
- [x] Charts update in-place — `chart.update('none')` in `upsertChart`, no page reload
- [x] Staleness warning if data > 15 min — `stale` flag in /api/gex, yellow pulsing CSS animation
- [x] Last-good snapshot on yfinance failure — `_refresh()` catches exceptions, keeps `_cache["snapshot"]`
- [x] Existing run.py and gex_calc.py unchanged — serve.py only imports from them
- [x] Connection lost banner — shown on fetch failure, hidden on recovery, 10s auto-retry

**No placeholders:** All code is complete.

**Type consistency:**
- `_prepare_results(results)` from `report.py` — returns dict with `chart_strikes`, `chart_gex`, `chart_vex`, `chart_chex` lists per ticker per level set. Dashboard JS reads `data[ticker].all.chart_strikes` etc. — matches.
- `futures` key from `add_futures_conversion`: contains `symbol`, `price`, `spot`, `call_wall`, `put_wall`, `gamma_flip`, `hvl`, `multiplier`. Dashboard reads all of these — matches.
- `_json_safe()` converts numpy types before storing in `_cache["snapshot"]` — prevents JSON serialization errors on `/api/gex`.
