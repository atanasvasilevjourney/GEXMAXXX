from __future__ import annotations


def compute_stop(entry: float, direction: str, atr: float,
                 k_atr: float = 0.75, min_ticks: float = 12.0) -> float:
    """ATR-scaled stop with floor. buffer = max(k_atr * atr, min_ticks)."""
    buffer = max(k_atr * atr, min_ticks)
    return entry - buffer if direction == 'long' else entry + buffer


def select_target(entry: float, direction: str, fut_levels: list) -> float | None:
    """Nearest Tier-1 FutLevel in trade direction."""
    if direction == 'long':
        candidates = [lv for lv in fut_levels if lv.tier == 1 and lv.fut_price > entry]
        return min(candidates, key=lambda lv: lv.fut_price).fut_price if candidates else None
    else:
        candidates = [lv for lv in fut_levels if lv.tier == 1 and lv.fut_price < entry]
        return max(candidates, key=lambda lv: lv.fut_price).fut_price if candidates else None


def stop_hit(tick, position) -> bool:
    """Returns True if current price has crossed the position stop."""
    if position.direction == 'long':
        return tick.price <= position.stop
    return tick.price >= position.stop
