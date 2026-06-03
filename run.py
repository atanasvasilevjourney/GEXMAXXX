from datetime import datetime
from data import fetch_chain, fetch_chain_cboe
from greeks import calculate_all_greeks
from levels import get_all_levels, add_futures_conversion
from report import build_report, save_report, open_report

SYMBOLS = ['SPY', 'QQQ']


def main():
    date_str = datetime.now().strftime('%Y-%m-%d')
    results = {}

    for ticker in SYMBOLS:
        try:
            # Primary: yfinance
            try:
                df, spot = fetch_chain(ticker)
                source = 'yfinance'
            except Exception as e_yf:
                print(f"[{ticker}] yfinance failed ({e_yf}), trying CBOE fallback...")
                df, spot = fetch_chain_cboe(ticker)
                source = 'cboe'

            print(f"[{ticker}] fetched {len(df)} options rows, spot={spot:.2f} (source: {source})")

            df = calculate_all_greeks(df, spot)
            levels = get_all_levels(df, spot)
            levels = add_futures_conversion(levels, ticker, spot)
            levels['source'] = source
            levels['ticker'] = ticker
            results[ticker] = levels

        except Exception as e:
            print(f"[{ticker}] ERROR: {e}")
            results[ticker] = {'ticker': ticker, 'error': str(e)}

    print(f"\nBuilding report for {date_str}...")
    html = build_report(results, date_str)
    path = save_report(html, date_str)
    print(f"Report saved: {path}")
    open_report(path)
    print("Report opened in browser.")


if __name__ == '__main__':
    main()
