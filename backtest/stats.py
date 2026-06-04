from __future__ import annotations
import statistics
from .models import TradeRecord, BacktestResult
from .validation import validate


def _sharpe(pnls: list[float]) -> float:
    if len(pnls) < 2:
        return 0.0
    mean = statistics.mean(pnls)
    std  = max(statistics.stdev(pnls), 0.01)
    return mean / std


def compute_stats(
    trades: list[TradeRecord],
    oos_start_pct: float = 0.70,
    run_validation: bool = True,
) -> BacktestResult:
    """
    Compute backtest statistics from a list of completed trades.

    Args:
        trades:          output of replay()
        oos_start_pct:   fraction of bars used as in-sample (default 0.70)
        run_validation:  if True (default), runs bootstrap MC + Kupiec tests
                         and tightens signal_only accordingly.
                         Set False in unit tests that don't need statistical tests.

    Returns:
        BacktestResult
    """
    if not trades:
        return BacktestResult(
            total_trades=0, positive_regime_trades=0, negative_regime_trades=0,
            positive_regime_pnl=0.0, negative_regime_pnl=0.0,
            positive_regime_sharpe=0.0, negative_regime_sharpe=0.0,
            oos_positive_pnl=0.0, regime_split_significant=False, signal_only=True,
            validation=None,
        )

    max_bar    = max(t.entry_bar for t in trades)
    oos_cutoff = int(max_bar * oos_start_pct)

    is_trades  = [t for t in trades if t.entry_bar <= oos_cutoff]
    oos_trades = [t for t in trades if t.entry_bar >  oos_cutoff]

    # All trades by regime (for trade counts)
    pos_trades = [t for t in trades if t.regime_at_entry == 'positive']
    neg_trades = [t for t in trades if t.regime_at_entry == 'negative']

    # In-sample trades by regime (for PnL sums and Sharpe)
    pos_is = [t for t in is_trades if t.regime_at_entry == 'positive']
    neg_is = [t for t in is_trades if t.regime_at_entry == 'negative']

    pos_pnl = sum(t.pnl_pts for t in pos_is)
    neg_pnl = sum(t.pnl_pts for t in neg_is)

    pos_sharpe = _sharpe([t.pnl_pts for t in pos_is])
    neg_sharpe = _sharpe([t.pnl_pts for t in neg_is])

    regime_split = pos_pnl > 0 and pos_sharpe > neg_sharpe

    oos_pos_pnl = sum(t.pnl_pts for t in oos_trades if t.regime_at_entry == 'positive')

    # Statistical validation gate
    vr = validate(trades) if run_validation else None
    validation_passed = vr.passed if vr is not None else True

    signal_only = not (regime_split and oos_pos_pnl > 0 and validation_passed)

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
        validation=vr,
    )
