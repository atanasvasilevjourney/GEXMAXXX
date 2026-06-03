from .models import MarketTick


class ProxyConfirm:
    """
    Automation proxy for order-flow confirmation (paper/replay mode).
    Fires on rejection-reclaim bar: price wicks through level, closes back inside.
    Signalled by MarketTick.has_rejection_bar in replay mode.
    """
    def confirmed(self, tick: MarketTick) -> bool:
        return tick.has_rejection_bar
