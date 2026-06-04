from __future__ import annotations
import math
import random
import statistics
from dataclasses import dataclass
from .models import TradeRecord


@dataclass
class ValidationResult:
    mc_pvalue:      float   # fraction of shuffles with Sharpe > observed; lower is better
    mc_passed:      bool    # True if mc_pvalue < 0.05
    kupiec_pvalue:  float   # chi-squared p-value; higher = better-calibrated VaR
    kupiec_passed:  bool    # True if kupiec_pvalue >= 0.05 (or n < 10 → auto-pass)
    n_permutations: int     # number of MC shuffles run
    passed:         bool    # True if mc_passed AND kupiec_passed


def _sharpe(pnls: list[float]) -> float:
    """Mean / stdev Sharpe ratio. Returns 0.0 if fewer than 2 trades."""
    if len(pnls) < 2:
        return 0.0
    mean = statistics.mean(pnls)
    std  = max(statistics.stdev(pnls), 1e-9)
    return mean / std


def monte_carlo_pvalue(
    pnls: list[float],
    n_permutations: int = 1000,
    seed: int = 42,
) -> float:
    """
    Permutation test: fraction of random orderings whose Sharpe STRICTLY EXCEEDS
    the observed Sharpe.

    A low p-value means the observed ordering achieves a Sharpe that random
    orderings cannot beat — the strategy's timing adds value.

    Uses strict > (not >=) so ties (identical PnLs) do not count as beats,
    ensuring an all-winning strategy scores p = 0.0.

    Args:
        pnls:           per-trade PnL values (from TradeRecord.pnl_pts)
        n_permutations: shuffle count (default 1000)
        seed:           RNG seed for reproducibility

    Returns:
        float in [0, 1]; p < 0.05 passes
    """
    if len(pnls) < 2:
        return 1.0   # inconclusive → conservative fail
    observed = _sharpe(pnls)
    rng = random.Random(seed)
    buf = list(pnls)
    beats = 0
    for _ in range(n_permutations):
        rng.shuffle(buf)
        if _sharpe(buf) > observed:   # strict > — ties don't count
            beats += 1
    return beats / n_permutations


def kupiec_pof(pnls: list[float], confidence: float = 0.95) -> float:
    """
    Kupiec Proportion of Failures test — one-sided at the stated confidence level.

    Tests whether the observed loss-exceedance rate exceeds the expected rate
    (1 - confidence). One-sided: auto-passes when p_hat <= alpha (fewer losses
    than expected is good news, not a rejection reason).

    Returns 1.0 (auto-pass) when n < 10 (inconclusive) or p_hat <= alpha.

    Args:
        pnls:       per-trade PnL values
        confidence: VaR confidence level (default 0.95)

    Returns:
        float in [0, 1]; p >= 0.05 passes
    """
    n = len(pnls)
    if n < 10:
        return 1.0   # too few trades — inconclusive, auto-pass

    alpha = 1.0 - confidence                         # expected exceedance rate = 0.05
    sorted_pnls = sorted(pnls)
    var_threshold = sorted_pnls[max(0, int(n * alpha) - 1)]   # 5th-percentile PnL

    x = sum(1 for p in pnls if p < var_threshold)   # observed exceedances
    p_hat = x / n

    # One-sided: only test if observed rate EXCEEDS expected rate
    if p_hat <= alpha:
        return 1.0   # fewer losses than expected → don't reject

    # Kupiec LR statistic: -2 * log(L0 / L1)
    if x == n:
        lr = -2.0 * (n * math.log(alpha))
    else:
        lr = -2.0 * (
            x * math.log(alpha / p_hat) +
            (n - x) * math.log((1.0 - alpha) / (1.0 - p_hat))
        )

    return _chi2_sf(lr, df=1)


def _chi2_sf(x: float, df: int) -> float:
    """
    Chi-squared survival function (1 - CDF), df=1.
    Uses scipy.stats when available (always true in this project);
    falls back to math.erfc for environments without scipy.
    """
    try:
        from scipy.stats import chi2
        return float(chi2.sf(x, df))
    except ImportError:
        # chi2(1) SF = erfc(sqrt(x/2))
        return math.erfc(math.sqrt(x / 2.0))


def validate(trades: list[TradeRecord], n_permutations: int = 1000) -> ValidationResult:
    """
    Run both statistical tests on completed trades.

    Args:
        trades:         output of backtest.replay()
        n_permutations: MC shuffle count (default 1000)

    Returns:
        ValidationResult — .passed is True only if both tests pass
    """
    pnls = [t.pnl_pts for t in trades]

    mc_p  = monte_carlo_pvalue(pnls, n_permutations=n_permutations)
    kup_p = kupiec_pof(pnls)

    mc_passed  = mc_p  < 0.05
    kup_passed = kup_p >= 0.05

    return ValidationResult(
        mc_pvalue      = mc_p,
        mc_passed      = mc_passed,
        kupiec_pvalue  = kup_p,
        kupiec_passed  = kup_passed,
        n_permutations = n_permutations,
        passed         = mc_passed and kup_passed,
    )
