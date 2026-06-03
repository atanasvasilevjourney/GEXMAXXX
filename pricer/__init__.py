from .style import PricerStyle, detect_style
from .time import time_to_expiry, MIN_T_YEARS
from .engines import ql_european_gamma, ql_american_gamma, ql_black76_gamma


def compute_gamma(S: float, K: float, T: float, r: float, sigma: float,
                  option_type: str, style: PricerStyle) -> float:
    """
    Dispatch to the correct pricer engine by PricerStyle.

    Args:
        S:           spot price (or futures price for BLACK76)
        K:           strike
        T:           time to expiry in years (use time_to_expiry() to compute)
        r:           risk-free rate
        sigma:       implied volatility
        option_type: 'call' | 'put'
        style:       PricerStyle.EUROPEAN | AMERICAN | BLACK76

    Returns:
        gamma (float, always positive)

    Raises:
        ValueError: if style is unrecognised
    """
    if style == PricerStyle.EUROPEAN:
        return ql_european_gamma(S, K, T, r, sigma, option_type)
    if style == PricerStyle.AMERICAN:
        return ql_american_gamma(S, K, T, r, sigma, option_type)
    if style == PricerStyle.BLACK76:
        return ql_black76_gamma(S, K, T, r, sigma, option_type)
    raise ValueError(f"Unknown PricerStyle: {style!r}")


__all__ = [
    'compute_gamma', 'PricerStyle', 'detect_style',
    'time_to_expiry', 'MIN_T_YEARS',
    'ql_european_gamma', 'ql_american_gamma', 'ql_black76_gamma',
]
