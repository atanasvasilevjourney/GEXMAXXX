import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from backtest.validation import monte_carlo_pvalue, kupiec_pof, validate, ValidationResult
from backtest.models import TradeRecord


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_trades(pnls: list[float]) -> list[TradeRecord]:
    return [
        TradeRecord(
            entry_bar=i, exit_bar=i + 1, direction='short',
            entry_price=100.0, exit_price=100.0 - pnl,
            pnl_pts=pnl, regime_at_entry='positive',
            level_tier=1, level_strength=0.9,
        )
        for i, pnl in enumerate(pnls)
    ]


# ── Monte Carlo ───────────────────────────────────────────────────────────────

def test_mc_pvalue_all_winners():
    # All identical PnLs → every bootstrap sample has same Sharpe → all > 0 → p = 0.0
    pnls = [10.0] * 50
    p = monte_carlo_pvalue(pnls, n_permutations=1000, seed=42)
    assert p == pytest.approx(0.0)


def test_mc_pvalue_random_not_significant():
    # Alternating +1/-1 → bootstrap Sharpes near 0 → ~50% non-positive → p ≈ 0.5 >> 0.05
    pnls = [1.0 if i % 2 == 0 else -1.0 for i in range(50)]
    p = monte_carlo_pvalue(pnls, n_permutations=1000, seed=42)
    assert p > 0.05


def test_mc_pvalue_too_few_trades_returns_1():
    # <2 trades → inconclusive → conservative fail (p=1.0)
    p = monte_carlo_pvalue([5.0], n_permutations=100, seed=42)
    assert p == pytest.approx(1.0)


# ── Kupiec POF ────────────────────────────────────────────────────────────────

def test_kupiec_too_few_trades_auto_pass():
    # n < 10 → inconclusive → auto-pass (p=1.0)
    p = kupiec_pof([1.0, -2.0, 3.0], confidence=0.95)
    assert p == pytest.approx(1.0)


def test_kupiec_few_exceedances_passes():
    # 97 winning trades + 3 losing trades → loss rate = 3% < 5% → one-sided auto-pass
    pnls = [10.0] * 97 + [-100.0, -200.0, -300.0]
    p = kupiec_pof(pnls, confidence=0.95)
    assert p >= 0.05


def test_kupiec_too_many_exceedances_fails():
    # 76 winning trades + 24 losing trades → loss rate = 24% >> 5% → Kupiec rejects
    pnls = [10.0] * 76 + [-100.0] * 4 + [-200.0] * 20
    p = kupiec_pof(pnls, confidence=0.95)
    assert p < 0.05


from backtest import compute_stats


# ── validate() end-to-end ─────────────────────────────────────────────────────

def test_validate_consistent_winners_passes():
    # 50 trades all +10 pts → MC p=0.0, Kupiec auto-pass (0% loss rate) → passed=True
    trades = _make_trades([10.0] * 50)
    vr = validate(trades)
    assert vr.mc_passed     is True
    assert vr.kupiec_passed is True
    assert vr.passed        is True
    assert vr.n_permutations == 1000


def test_validate_random_pnls_mc_fails():
    # Alternating +1/-1 → MC p > 0.05 → mc_passed=False → passed=False
    pnls = [1.0 if i % 2 == 0 else -1.0 for i in range(50)]
    trades = _make_trades(pnls)
    vr = validate(trades)
    assert vr.mc_passed is False
    assert vr.passed    is False
