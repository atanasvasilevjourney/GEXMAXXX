from __future__ import annotations
from .models import Level, FutLevel


def measure_basis(future_price: float, index_value: float) -> float:
    """
    Additive basis in index points.

    basis_pts = future_price - index_value

    Works for ES/SPX and NQ/NDX (both track 1:1 in points).
    No hardcoded multiplier — basis is always measured, never assumed.
    """
    return future_price - index_value


def project(levels: list[Level], basis_pts: float) -> list[FutLevel]:
    """
    Project each index-space Level to a futures price.

    fut_price = strike + basis_pts

    Args:
        levels:    list of Level objects (from select_levels)
        basis_pts: measured basis from measure_basis()

    Returns:
        list of FutLevel objects in the same order as input levels.
    """
    return [
        FutLevel(
            fut_price=level.strike + basis_pts,
            tier=level.tier,
            strength=level.strength,
            label=level.label,
            source_strike=level.strike,
            basis_pts=basis_pts,
        )
        for level in levels
    ]
