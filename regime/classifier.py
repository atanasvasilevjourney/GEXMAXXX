from __future__ import annotations
from .models import Regime


def classify(snapshot: dict, conviction_scale: float = 0.05) -> Regime:
    """
    Classify the current gamma regime from a find_levels() snapshot.

    Args:
        snapshot:         dict from find_levels() with 'total_gex', 'spot', 'gamma_flip'
        conviction_scale: fraction of spot that equals full conviction distance (default 5%)

    Returns:
        Regime with state ('positive'|'negative'), distance_to_zero_gamma, conviction (0-1)
    """
    spot       = float(snapshot['spot'])
    gamma_flip = float(snapshot['gamma_flip'])
    total_gex  = float(snapshot['total_gex'])

    if spot == 0.0:
        raise ValueError("spot cannot be zero")

    state = 'positive' if total_gex > 0 else 'negative'

    distance_to_zero_gamma = spot - gamma_flip

    full_conviction_distance = spot * conviction_scale
    conviction = min(1.0, abs(distance_to_zero_gamma) / full_conviction_distance)

    return Regime(
        state=state,
        distance_to_zero_gamma=distance_to_zero_gamma,
        conviction=conviction,
    )
