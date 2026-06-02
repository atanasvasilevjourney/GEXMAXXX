# GEX Ultimate — Design Spec
**Date:** 2026-06-02
**Status:** Approved

## Overview

Upgrade the existing GEX calculator into a modular package that produces a self-contained daily HTML report. Covers SPY and QQQ with full Greek exposure (GEX + Vanna + Charm), 0DTE vs multi-day split, and ES/NQ futures conversion. Data source: yfinance primary, CBOE delayed scrape as automatic fallback.

## Goals

- `python run.py` → fetches live data, computes all Greeks, saves `reports/YYYY-MM-DD.html`, opens in browser
- Existing `gex_calc.py` CLI remains untouched and working
- Clean module boundaries so swapping data source (e.g. adding Tradier) requires editing only `data.py`

---

## File Structure

```
D:/2026/GEX/
├── gex_calc.py           # existing CLI (unchanged)
├── run.py                # new entry point
├── data.py               # data fetching layer
├── greeks.py             # all Greek calculations
├── levels.py             # level detection + 0DTE split + futures conversion
├── report.py             # HTML report builder
├── templates/
│   └── report.html       # Jinja2 HTML template with Chart.js
├── reports/              # generated HTML output (gitignored)
├── tests/
│   ├── __init__.py
│   ├── test_greeks.py
│   ├── test_levels.py
│   └── test_report.py
└── requirements.txt      # add: requests>=2.31, jinja2>=3.1
```

---

## Module Specifications

### `data.py`

**`fetch_chain(ticker: str) -> tuple[pd.DataFrame, float]`**
- Uses `yfinance.Ticker(ticker)`
- Spot via `fast_info.last_price` with fallback chain: `info['regularMarketPrice']` → `history(period='1d')`
- Fetches nearest 4 expirations via `stock.options[:4]`
- Raises `ValueError` if no expirations found
- Returns DataFrame with columns: `strike, type, openInterest, impliedVolatility, expiration`
- Returns `(df, float(spot))`

**`fetch_chain_cboe(ticker: str) -> tuple[pd.DataFrame, float]`**
- Scrapes `https://www.cboe.com/delayed_quotes/{ticker.lower()}/quote_table/` via `requests`
- Parses HTML table to extract same columns as `fetch_chain`
- Returns same `(df, float(spot))` interface
- Called automatically by `run.py` if `fetch_chain` raises

**`fetch_chain_tradier(ticker: str, api_key: str) -> tuple[pd.DataFrame, float]`**
- Stub only: raises `NotImplementedError("Configure Tradier API key first")`
- Ready for future implementation, zero changes needed elsewhere

**Provider selection in `run.py`:**
```python
try:
    df, spot = fetch_chain(ticker)
    source = 'yfinance'
except Exception:
    df, spot = fetch_chain_cboe(ticker)
    source = 'cboe'
```

---

### `greeks.py`

All calculations vectorized with NumPy/SciPy. Common inputs: `df` (options DataFrame), `spot: float`, `r: float = 0.05`.

**`calculate_all_greeks(df, spot, r=0.05) -> pd.DataFrame`**
Calls all three functions below in sequence. Returns enriched DataFrame with `T`, `iv`, `d1`, `d2`, `gamma`, `gex`, `vanna`, `vex`, `charm`, `chex` columns.

**`calculate_gex(df, spot, r) -> pd.DataFrame`**
- Same as existing implementation
- `T` = time to expiry in years, clipped at 0.001
- `iv` = impliedVolatility, 0/NaN replaced with 0.20
- `d1 = (ln(S/K) + (r + σ²/2)*T) / (σ*√T)`
- `d2 = d1 - σ*√T`
- `gamma = norm.pdf(d1) / (S * σ * √T)`
- `gex = gamma * OI * 100 * S² * 0.01`
- puts: `gex *= -1`

**`calculate_vanna(df, spot) -> pd.DataFrame`**
- Requires `d1`, `d2`, `iv` columns (added by `calculate_gex`)
- `vanna = -norm.pdf(d1) * d2 / iv`
- `vex = vanna * OI * 100 * spot * 0.01`
- puts: `vex *= -1`

**`calculate_charm(df, spot) -> pd.DataFrame`**
- Requires `d1`, `d2`, `T`, `iv` columns
- `charm = -norm.pdf(d1) * (2*r*T - d2*iv*√T) / (2*T*iv*√T)`
- `chex = charm * OI * 100`
- puts: `chex *= -1`

---

### `levels.py`

**`find_levels(df: pd.DataFrame, spot: float) -> dict`**
- Same algorithm as existing `gex_calc.py`
- Aggregates `gex` by strike
- Returns: `{spot, call_wall, put_wall, gamma_flip, hvl, total_gex, regime}`
- Also aggregates `vex` and `chex` by strike, adds to return dict:
  - `vanna_wall`: strike with highest absolute VEX
  - `charm_wall`: strike with highest absolute CHEX
  - `total_vex`, `total_chex`

**`split_0dte(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]`**
- Splits by expiration date vs today's date
- Returns `(df_0dte, df_multi)`
- If no 0DTE options exist today, `df_0dte` is empty DataFrame

**`get_all_levels(df: pd.DataFrame, spot: float) -> dict`**
Calls `find_levels` three times:
```python
{
  'all':   find_levels(df, spot),
  '0dte':  find_levels(df_0dte, spot) if len(df_0dte) > 0 else None,
  'multi': find_levels(df_multi, spot),
}
```

**`get_futures_price(symbol: str) -> float`**
- Fetches live price for `NQ=F` or `ES=F` via yfinance
- Same fallback chain as `fetch_chain` spot fetching

**`add_futures_conversion(symbol_levels: dict, ticker: str, spot: float) -> dict`**
- For QQQ: fetches NQ=F, computes `nq_mult = NQ_price / spot`, converts all levels
- For SPY: fetches ES=F, computes `es_mult = ES_price / spot`, converts all levels
- Adds `futures` key to symbol_levels dict:
  ```python
  'futures': {
    'symbol': 'NQ',  # or 'ES'
    'price': 21850.0,
    'multiplier': 29.36,
    'spot': 21850,
    'call_wall': 21900,
    'put_wall': 21700,
    'gamma_flip': 21820,
    'hvl': 21700,
  }
  ```

---

### `report.py`

**`build_report(spy_data: dict, qqq_data: dict, date_str: str) -> str`**
- Loads `templates/report.html` via Jinja2
- Renders with spy_data, qqq_data, date_str
- Returns rendered HTML string

**`save_report(html: str, date_str: str) -> str`**
- Creates `reports/` directory if needed
- Saves to `reports/{date_str}.html`
- Returns file path

**`open_report(path: str) -> None`**
- Calls `webbrowser.open(f"file://{path}")`

---

### `templates/report.html`

Self-contained HTML (no external files). Includes:
- Chart.js via CDN for bar charts
- Inline CSS (dark theme matching existing `spy_options_strategy_dashboard.html` style)
- Per-symbol sections:
  - Header: ticker, date, data source badge, regime badge
  - 3-column key levels table: All / 0DTE / Futures equivalent
  - GEX by strike bar chart (calls green, puts red, net line)
  - Vanna exposure bar chart
  - Charm exposure bar chart
- No external font imports — system fonts only (avoids CORS issues on local file)

---

### `run.py`

```python
SYMBOLS = ["SPY", "QQQ"]

for ticker in SYMBOLS:
    1. fetch chain (yfinance → CBOE fallback)
    2. calculate_all_greeks(df, spot)
    3. split_0dte(df) → df_0dte, df_multi
    4. get_all_levels(df, spot)
    5. add_futures_conversion(levels, ticker, spot)
    6. store in results dict

build_report(results['SPY'], results['QQQ'], date_str)
save_report(html, date_str)
open_report(path)
```

Per-symbol try/except: if one fails, the other still runs and the report shows an error card for the failed symbol.

---

## HTML Report Layout (per symbol)

```
╔═══════════════════════════════════════════════════╗
║  SPY  2026-06-02  [yfinance]  POSITIVE (low vol)  ║
╠══════════════╦════════════════╦═══════════════════╣
║  ALL CHAIN   ║   0DTE ONLY    ║   ES EQUIVALENT   ║
║  Spot 760.22 ║  Spot 760.22   ║  ES Spot 5412     ║
║  Call  761   ║  Call  761     ║  ES Call  5419    ║
║  Put   759   ║  Put   758     ║  ES Put   5405    ║
║  Flip  760   ║  Flip  760     ║  ES Flip  5412    ║
║  HVL   759   ║  HVL   758     ║  ES HVL   5405    ║
║  GEX +5.42B  ║  GEX  +1.2B   ║                   ║
╠══════════════╩════════════════╩═══════════════════╣
║  GEX BY STRIKE ▐bar chart▌                        ║
║  VANNA EXPOSURE ▐bar chart▌                        ║
║  CHARM EXPOSURE ▐bar chart▌                        ║
╚═══════════════════════════════════════════════════╝
```

---

## Error Handling

- Per-symbol try/except in `run.py` — failed symbol shows error card in HTML
- `fetch_chain` raises → automatic CBOE fallback, source badge changes to `cboe`
- Empty 0DTE (weekdays with no same-day expiry) → 0DTE column shows "No 0DTE today"
- Futures fetch failure → futures column shows "N/A" (non-blocking)

---

## Testing

- `tests/test_greeks.py` — unit tests for calculate_gex, calculate_vanna, calculate_charm using synthetic DataFrames
- `tests/test_levels.py` — unit tests for find_levels, split_0dte, get_futures_price (mocked)
- `tests/test_report.py` — smoke test: build_report with synthetic data produces valid HTML containing expected strings
- No tests for `data.py` (network) or `run.py` (integration entry point)

---

## Dependencies Added

```
requests>=2.31      # CBOE fallback scraping
jinja2>=3.1         # HTML templating
```

All other dependencies already in requirements.txt (yfinance, pandas, numpy, scipy).
