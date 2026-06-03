from __future__ import annotations
import statistics
from .models import TradeRecord, BacktestResult


def _sharpe(pnls: list[float]) -> float:
    if len(pnls) < 2:
        return 0.0
    mean = statistics.mean(pnls)
    std  = max(statistics.stdev(pnls), 0.01)
    return mean / std


def compute_stats(trades: list[TradeRecord], oos_start_pct: float = 0.70) -> BacktestResult:
    if not trades:
        return BacktestResult(
            total_trades=0, positive_regime_trades=0, negative_regime_trades=0,
            positive_regime_pnl=0.0, negative_regime_pnl=0.0,
            positive_regime_sharpe=0.0, negative_regime_sharpe=0.0,
            oos_positive_pnl=0.0, regime_split_significant=False, signal_only=True,
        )

    max_bar = max(t.entry_bar for t in trades)
    oos_cutoff = int(max_bar * oos_start_pct)

    is_trades  = [t for t in trades if t.entry_bar <= oos_cutoff]
    oos_trades = [t for t in trades if t.entry_bar >  oos_cutoff]

    # All trades by regime
    pos_trades = [t for t in trades if t.regime_at_entry == 'positive']
    neg_trades = [t for t in trades if t.regime_at_entry == 'negative']

    # In-sample trades by regime (for sharpe calculation)
    pos_is = [t for t in is_trades if t.regime_at_entry == 'positive']
    neg_is = [t for t in is_trades if t.regime_at_entry == 'negative']

    pos_pnl = sum(t.pnl_pts for t in pos_trades)
    neg_pnl = sum(t.pnl_pts for t in neg_trades)

    pos_sharpe = _sharpe([t.pnl_pts for t in pos_is])
    neg_sharpe = _sharpe([t.pnl_pts for t in neg_is])

    regime_split = pos_pnl > 0 and pos_sharpe > neg_sharpe

    oos_pos_pnl = sum(t.pnl_pts for t in oos_trades if t.regime_at_entry == 'positive')

    signal_only = not (regime_split and oos_pos_pnl > 0)

    return BacktestResult(
        total_trades=len(trades),
        positive_regime_trades=len(pos_trades),
        negative_regime_trades=len(neg_trades),
        positive_regime_pnl=pos_pnl,
        negative_regime_pnl=neg_pnl,
        positive_regime_sharpe=pos_sharpe,
        negative_regime_sharpe=neg_sharpe,
        oos_positive_pnl=oos_pos_pnl,
        regime_split_significant=regime_split,
        signal_only=signal_only,
    )
