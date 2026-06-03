from .models import BacktestResult


def format_report(result: BacktestResult) -> str:
    lines = [
        '=== GPR Backtest Report ===',
        f'Total trades:           {result.total_trades}',
        f'  Positive regime:      {result.positive_regime_trades}'
        f'  (pnl: {result.positive_regime_pnl:+.1f} pts,'
        f' sharpe: {result.positive_regime_sharpe:.2f})',
        f'  Negative regime:      {result.negative_regime_trades}'
        f'  (pnl: {result.negative_regime_pnl:+.1f} pts,'
        f' sharpe: {result.negative_regime_sharpe:.2f})',
        '',
        f'OOS positive-regime pnl: {result.oos_positive_pnl:+.1f} pts',
        f'Regime split significant: {"Yes" if result.regime_split_significant else "No"}',
        '',
        f'*** SIGNAL_ONLY: {result.signal_only} ***',
    ]
    if result.signal_only:
        lines.append('*** LIVE EXECUTION BLOCKED — regime edge not validated ***')
    else:
        lines.append('*** EDGE VALIDATED — execution permitted ***')
    return '\n'.join(lines)
