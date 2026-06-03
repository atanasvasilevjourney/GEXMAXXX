from __future__ import annotations
import pandas as pd
from .models import Level


def select_levels(snapshot: dict, pct_threshold: float = 0.30) -> list[Level]:
    """
    Select significant gamma strikes from a find_levels() snapshot.

    Always includes (Tier-1): call_wall, put_wall, pin_strike, zero_gamma.
    Adds (Tier-2): any strike where |gex| >= pct_threshold * max|gex|.

    Args:
        snapshot: dict returned by levels.find_levels()
        pct_threshold: minimum |gex| fraction of max to qualify as Tier-2

    Returns:
        List of Level objects sorted by strike ascending.
    """
    by_strike: pd.DataFrame = snapshot.get('by_strike', pd.DataFrame())
    if by_strike.empty:
        return []

    by_strike = by_strike.dropna(subset=['gex'])
    if by_strike.empty:
        return []

    max_abs_gex = by_strike['gex'].abs().max()
    if max_abs_gex == 0:
        return []

    call_wall  = snapshot.get('call_wall')
    put_wall   = snapshot.get('put_wall')
    gamma_flip = snapshot.get('gamma_flip')
    pin_strike = float(by_strike.loc[by_strike['gex'].abs().idxmax(), 'strike'])

    tier1: dict[float, str] = {}
    if call_wall is not None:
        tier1[float(call_wall)] = 'call_wall'
    if put_wall is not None:
        tier1[float(put_wall)] = 'put_wall'
    if gamma_flip is not None:
        tier1[float(gamma_flip)] = 'zero_gamma'
    if pin_strike not in tier1:
        tier1[pin_strike] = 'pin'

    def _gex_at(strike: float) -> float:
        row = by_strike[by_strike['strike'] == strike]
        return float(row.iloc[0]['gex']) if not row.empty else 0.0

    levels: list[Level] = []

    for strike, label in tier1.items():
        gex = _gex_at(strike)
        strength = abs(gex) / max_abs_gex
        levels.append(Level(strike=strike, gex=gex, tier=1, strength=strength, label=label))

    tier1_strikes = set(tier1.keys())

    for _, row in by_strike.iterrows():
        strike = float(row['strike'])
        if strike in tier1_strikes:
            continue
        gex = float(row['gex'])
        strength = abs(gex) / max_abs_gex
        if strength >= pct_threshold:
            levels.append(Level(strike=strike, gex=gex, tier=2, strength=strength, label='cluster'))

    levels.sort(key=lambda lv: lv.strike)
    return levels
