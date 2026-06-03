from enum import Enum


class PricerStyle(str, Enum):
    EUROPEAN = "european"  # SPX, NDX — cash-settled index options, no early exercise
    AMERICAN = "american"  # QQQ, SPY, IWM — ETF options, early exercise possible
    BLACK76  = "black76"   # ES=F, NQ=F, MNQ=F — futures options


def detect_style(ticker: str) -> PricerStyle:
    """
    Detect the correct pricing model for an options chain by ticker.

    Args:
        ticker: e.g. 'SPX', '^SPX', 'QQQ', 'ES=F'

    Returns:
        PricerStyle.EUROPEAN, AMERICAN, or BLACK76
    """
    t = ticker.upper().lstrip('^')
    if t in ('SPX', 'NDX', 'XSP'):
        return PricerStyle.EUROPEAN
    if t.endswith('=F'):
        return PricerStyle.BLACK76
    return PricerStyle.AMERICAN  # QQQ, SPY, IWM and any unknown ETF
